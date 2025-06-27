import sys
from google_play_scraper import Sort, reviews
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from datetime import datetime
import time
import jieba
from wordcloud import WordCloud
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
from langdetect import detect
import langdetect
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import torch
import warnings
from functools import lru_cache
warnings.filterwarnings('ignore')  # 忽略警告

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def extract_app_id(url):
    """Extract application ID from Google Play URL"""
    try:
        # Handle escaped characters
        url = url.replace('\\?', '?').replace('\\=', '=')
        print(f"Processed URL: {url}")  # Debug info
        
        # Try multiple pattern matching
        patterns = [
            r'id=([^&\s]+)',  # Standard format
            r'details/([^/?&\s]+)',  # Alternative format
            r'apps/([^/?&\s]+)'  # Short format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                app_id = match.group(1)
                print(f"Found app ID: {app_id}")  # Debug info
                return app_id
        
        raise ValueError("Could not extract app ID from URL")
    except Exception as e:
        print(f"Error processing URL: {str(e)}")  # Debug info
        raise ValueError(f"Invalid Google Play URL: {str(e)}")

# Cache for language detection
@lru_cache(maxsize=1000)
def is_english(text):
    """Check if text is in English"""
    try:
        return detect(text) == 'en'
    except (langdetect.lang_detect_exception.LangDetectException, Exception):
        return False

def fetch_reviews_batch(app_id, lang, country, country_name, continuation_token=None):
    """Fetch a batch of reviews"""
    try:
        response, next_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.MOST_RELEVANT,
            count=200,  # Increased batch size
            continuation_token=continuation_token
        )
        
        # Process reviews in bulk
        filtered_reviews = [
            {**review, 'country': country_name}
            for review in response
            if review['score'] <= 3 and is_english(review['content'])
        ]
        
        return filtered_reviews, next_token
        
    except Exception as e:
        print(f"Error fetching reviews from {country_name}: {str(e)}")
        return [], None

def get_reviews_for_country(app_id, lang, country, country_name, result_queue, min_reviews_per_country=3000):
    """Get reviews for a single country"""
    local_results = []
    continuation_token = None
    retry_count = 0
    max_retries = 5
    max_attempts = 50
    attempts = 0
    
    # Use set for faster duplicate checking
    seen_contents = set()
    
    while attempts < max_attempts and len(local_results) < min_reviews_per_country:
        try:
            time.sleep(1)  # Reduced delay
            
            filtered_reviews, continuation_token = fetch_reviews_batch(
                app_id, lang, country, country_name, continuation_token
            )
            
            # Efficient deduplication using set
            new_reviews = []
            for review in filtered_reviews:
                content = review['content']
                if content not in seen_contents:
                    seen_contents.add(content)
                    new_reviews.append(review)
            
            local_results.extend(new_reviews)
            print(f"{country_name}: Added {len(new_reviews)} English reviews, Total: {len(local_results)}")
            
            if not continuation_token or not filtered_reviews:
                print(f"No more reviews available for {country_name}")
                break
                
            attempts += 1
            
        except Exception as e:
            retry_count += 1
            print(f"Error getting reviews for {country_name} (Attempt {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                print(f"Maximum retry attempts reached for {country_name}")
                break
                
            time.sleep(2)  # Reduced retry delay
    
    result_queue.put((country_name, local_results))
    return len(local_results)

def get_reviews(app_id, min_reviews=10000):
    """Fetch reviews from multiple countries in parallel"""
    # Target country and language pairs (all using English)
    lang_country_pairs = [
        ('en_US', 'us', 'United States'),
        ('en_GB', 'gb', 'United Kingdom'),
        ('en_CA', 'ca', 'Canada'),
        ('en_AU', 'au', 'Australia'),
        ('en_IN', 'in', 'India'),
        ('en_PH', 'ph', 'Philippines'),
    ]
    
    result_queue = Queue()
    min_reviews_per_country = min_reviews // len(lang_country_pairs)
    
    print(f"Starting parallel review collection, targeting {min_reviews_per_country} reviews per country...")
    
    # Increased thread pool size for better parallelization
    with ThreadPoolExecutor(max_workers=len(lang_country_pairs) * 2) as executor:
        futures = [
            executor.submit(
                get_reviews_for_country,
                app_id,
                lang,
                country,
                country_name,
                result_queue,
                min_reviews_per_country
            )
            for lang, country, country_name in lang_country_pairs
        ]
        
        # Process results as they complete
        for future in as_completed(futures):
            try:
                review_count = future.result()
            except Exception as e:
                print(f"Error in review collection: {str(e)}")
    
    # Efficient collection and deduplication
    all_reviews = []
    seen_contents = set()
    
    while not result_queue.empty():
        _, reviews = result_queue.get()
        for review in reviews:
            content = review['content']
            if content not in seen_contents:
                seen_contents.add(content)
                all_reviews.append(review)
    
    # Sort by rating
    all_reviews.sort(key=lambda x: x['score'])
    
    print(f"\nReview collection completed! Total: {len(all_reviews)} reviews")
    if len(all_reviews) < min_reviews:
        print(f"Warning: Could not reach target review count ({min_reviews}), only got {len(all_reviews)} reviews")
    
    return all_reviews

def preprocess_text(text):
    """Preprocess text for analysis"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ' '.join(text.split())
    return text

def analyze_reviews(reviews_data):
    """Analyze review content using BERTopic for topic modeling"""
    # Rating statistics
    ratings = [review['score'] for review in reviews_data]
    rating_counts = Counter(ratings)
    
    # Prepare and preprocess review texts
    review_texts = [preprocess_text(review['content']) for review in reviews_data]
    
    # Check dataset size
    if len(review_texts) < 10:
        print("Too few reviews for topic modeling, falling back to keyword analysis...")
        return analyze_reviews_keywords(reviews_data)
    
    try:
        print("Initializing topic model...")
        
        # Check for GPU availability and set batch size
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 32 if device == "cuda" else 16
        print(f"Using device: {device}")
        
        # Initialize sentence transformer model with optimized settings
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Configure BERTopic with optimized settings
        topic_model = BERTopic(
            embedding_model=sentence_model,
            min_topic_size=3,
            nr_topics="auto",
            language="english",
            verbose=True,
            calculate_probabilities=False,
            low_memory=True,
            batch_size=batch_size,
            n_gram_range=(1, 2),  # Consider bigrams
            min_df=5,  # Minimum document frequency
            top_n_words=10  # Increase for better topic representation
        )
        
        print("Performing topic modeling analysis...")
        
        # Filter and chunk texts for efficient processing
        valid_texts = [text for text in review_texts if len(text.split()) >= 3]
        if not valid_texts:
            print("No reviews of sufficient length, falling back to keyword analysis...")
            return analyze_reviews_keywords(reviews_data)
        
        # Process in batches if dataset is large
        if len(valid_texts) > 1000:
            chunk_size = 1000
            topics = []
            for i in range(0, len(valid_texts), chunk_size):
                chunk = valid_texts[i:i + chunk_size]
                chunk_topics, _ = topic_model.fit_transform(chunk)
                topics.extend(chunk_topics)
        else:
            topics, _ = topic_model.fit_transform(valid_texts)
        
        # Get topic information efficiently
        topic_info = topic_model.get_topic_info()
        all_topics_docs = topic_model.get_representative_docs()
        
        # Process topics efficiently
        topic_details = {}
        topic_counts = defaultdict(int)
        for t in topics:
            if t != -1:
                topic_counts[t] += 1
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id != -1 and topic_counts[topic_id] > 0:
                try:
                    topic_words = topic_model.get_topic(topic_id)
                    if not topic_words:
                        continue
                    
                    representative_docs = all_topics_docs.get(topic_id, [])
                    if not representative_docs:
                        continue
                    
                    topic_details[int(topic_id)] = {
                        'keywords': [word for word, _ in topic_words[:5]],
                        'count': topic_counts[topic_id],
                        'examples': representative_docs[:2]
                    }
                except Exception as e:
                    print(f"Error processing topic {topic_id}: {str(e)}")
                    continue
        
        if not topic_details:
            print("Topic modeling did not produce valid results, falling back to keyword analysis...")
            return analyze_reviews_keywords(reviews_data)
        
        return rating_counts, topic_details
        
    except Exception as e:
        print(f"Error during topic modeling: {str(e)}")
        print("Falling back to keyword analysis...")
        return analyze_reviews_keywords(reviews_data)

def analyze_reviews_keywords(reviews_data):
    """Use keyword analysis as fallback method"""
    # Rating statistics
    ratings = [review['score'] for review in reviews_data]
    rating_counts = Counter(ratings)
    
    # Define issue keywords (English)
    problem_keywords = {
        'Technical Issues': [
            'crash', 'crashes', 'crashing', 'force close', 'stop working', 'not working',
            'black screen', 'freeze', 'frozen', 'stuck', 'loading', 'error', 'bug',
            'glitch', 'broken', 'malfunction'
        ],
        'Performance': [
            'slow', 'lag', 'laggy', 'stutter', 'fps', 'frame rate', 'battery drain',
            'overheat', 'overheating', 'performance', 'optimization', 'heavy'
        ],
        'User Interface': [
            'interface', 'ui', 'ux', 'design', 'layout', 'confusing', 'difficult',
            'hard to use', 'complicated', 'unintuitive', 'clunky', 'messy',
            'user friendly', 'user unfriendly', 'navigation'
        ],
        'Advertisements': [
            'ad', 'ads', 'advert', 'advertising', 'commercial', 'popup', 'pop-up',
            'video ad', 'forced ad', 'too many ads', 'intrusive', 'annoying ads'
        ],
        'Game Mechanics': [
            'gameplay', 'difficult', 'too hard', 'too easy', 'unfair', 'balance',
            'mechanics', 'controls', 'unplayable', 'impossible', 'challenging'
        ],
        'Content & Features': [
            'feature', 'content', 'missing', 'need', 'should add', 'suggestion',
            'limited', 'lack of', 'more levels', 'more content', 'repetitive',
            'boring', 'short'
        ],
        'Account & Login': [
            'account', 'login', 'sign in', 'register', 'authentication', 'password',
            'facebook', 'google', 'connection', 'offline', 'online', 'server'
        ],
        'Monetization': [
            'price', 'expensive', 'cost', 'pay', 'payment', 'money', 'coin',
            'premium', 'subscription', 'purchase', 'microtransaction', 'pay to win'
        ],
        'Updates & Compatibility': [
            'update', 'version', 'latest', 'old', 'newer', 'compatibility',
            'android', 'ios', 'phone', 'device', 'support', 'incompatible'
        ],
        'Storage & Data': [
            'storage', 'space', 'size', 'memory', 'large', 'data', 'download',
            'cache', 'mb', 'gb', 'huge', 'backup', 'save'
        ]
    }
    
    # Preprocess reviews once
    preprocessed_reviews = [(review['content'].lower(), review['content']) 
                          for review in reviews_data]
    
    # Count occurrences efficiently
    topic_details = {}
    for category, keywords in problem_keywords.items():
        # Create keyword pattern once
        pattern = '|'.join(map(re.escape, keywords))
        regex = re.compile(pattern)
        
        matching_reviews = []
        count = 0
        
        for lower_content, original_content in preprocessed_reviews:
            if regex.search(lower_content):
                count += 1
                if len(matching_reviews) < 2:  # Collect up to 2 examples
                    matching_reviews.append(original_content)
        
        if count > 0:
            topic_details[category] = {
                'keywords': keywords[:5],
                'count': count,
                'examples': matching_reviews
            }
    
    return rating_counts, topic_details

def create_word_cloud(reviews_data, output_path):
    """Create word cloud visualization"""
    # Combine and preprocess all review text
    text = ' '.join(preprocess_text(review['content']) for review in reviews_data)
    
    # Use jieba for word segmentation with cache
    words = jieba.cut(text)
    word_space_split = ' '.join(words)
    
    # Create WordCloud object with optimized settings
    wc = WordCloud(
        font_path='/System/Library/Fonts/Arial Unicode.ttf',
        width=1000,
        height=700,
        background_color='white',
        max_words=200,
        max_font_size=100,
        random_state=42,
        prefer_horizontal=0.7,
        collocations=True,
        normalize_plurals=False
    )
    
    # Generate word cloud
    wc.generate(word_space_split)
    
    # Create and save figure
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(rating_counts, topic_details, output_path):
    """Create visualization charts"""
    # Set style once
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Plot rating distribution
    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())
    bars1 = ax1.bar(ratings, counts, color='#2ecc71')
    ax1.set_title('Rating Distribution', fontsize=12, pad=15)
    ax1.set_xlabel('Rating', fontsize=10)
    ax1.set_ylabel('Number of Reviews', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Plot topic distribution
    topics = list(topic_details.keys())
    topic_counts = [details['count'] for details in topic_details.values()]
    
    # Prepare topic names
    if all(isinstance(topic, int) for topic in topics):
        topic_names = [f"Topic {tid}\n({', '.join(details['keywords'][:3])})"
                      for tid, details in topic_details.items()]
    else:
        topic_names = topics
    
    # Sort by count
    sorted_indices = np.argsort(topic_counts)[::-1]
    topic_names = [topic_names[i] for i in sorted_indices]
    topic_counts = [topic_counts[i] for i in sorted_indices]
    
    bars2 = ax2.bar(range(len(topic_counts)), topic_counts, color='#3498db')
    ax2.set_title('Issue Distribution', fontsize=12, pad=15)
    ax2.set_xlabel('Issues', fontsize=10)
    ax2.set_ylabel('Number of Reviews', fontsize=10)
    ax2.set_xticks(range(len(topic_names)))
    ax2.set_xticklabels(topic_names, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(app_url):
    """Main function"""
    try:
        # Download NLTK resources
        download_nltk_resources()
        
        # Get app ID
        app_id = extract_app_id(app_url)
        print(f"Starting review collection for app {app_id}...")
        
        # Get reviews
        print("Fetching reviews...")
        reviews_data = get_reviews(app_id, min_reviews=10000)
        
        if not reviews_data:
            print("No 3-star or lower English reviews found")
            return
            
        print(f"Successfully retrieved {len(reviews_data)} reviews, starting analysis...")
        
        # Create DataFrame
        df = pd.DataFrame(reviews_data)
        
        # Analyze reviews
        print("Analyzing reviews...")
        rating_counts, topic_details = analyze_reviews(reviews_data)
        
        # Generate timestamp and filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        formatted_app_id = app_id.replace('.', '-')
        
        excel_output = f'{formatted_app_id}_reviews_{timestamp}.xlsx'
        chart_output = f'{formatted_app_id}_analysis_{timestamp}.png'
        wordcloud_output = f'{formatted_app_id}_wordcloud_{timestamp}.png'
        
        # Export data
        print(f"Exporting to Excel: {excel_output}")
        if len(df) > 0:
            # Add topic information efficiently
            topic_keywords = defaultdict(str)
            for topic_id, details in topic_details.items():
                keywords = ', '.join(details['keywords'])
                for example in details['examples']:
                    topic_keywords[example] = keywords
            
            df['topic_keywords'] = df['content'].map(topic_keywords)
            
            # Export to Excel
            df.to_excel(excel_output, index=False, columns=[
                'score', 'content', 'thumbsUpCount', 'reviewCreatedVersion',
                'at', 'country', 'topic_keywords'
            ])
            
            # Create visualizations
            print(f"Generating analysis charts: {chart_output}")
            create_visualizations(rating_counts, topic_details, chart_output)
            
            print(f"Generating word cloud: {wordcloud_output}")
            create_word_cloud(reviews_data, wordcloud_output)
            
            print("\nAnalysis completed!")
            print(f"Review data saved to: {excel_output}")
            print(f"Analysis charts saved to: {chart_output}")
            print(f"Word cloud saved to: {wordcloud_output}")
            
            # Print topic analysis results
            print("\nMain Issue Topics Analysis:")
            for topic_id, details in topic_details.items():
                print(f"\nTopic {topic_id}:")
                print(f"Keywords: {', '.join(details['keywords'])}")
                print(f"Review count: {details['count']}")
                print("Example reviews:")
                for i, example in enumerate(details['examples'], 1):
                    print(f"{i}. {example[:200]}...")
                
        else:
            print("Warning: No data to export")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print("Detailed error information:")
        print(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_app_reviews.py <Google Play App URL>")
        sys.exit(1)
    
    main(sys.argv[1]) 
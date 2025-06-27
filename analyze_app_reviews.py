import sys
from google_play_scraper import Sort, reviews
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from langdetect import detect
import langdetect
import warnings
from functools import lru_cache
from wordcloud import WordCloud
import numpy as np
warnings.filterwarnings('ignore')

def extract_app_id(app_url):
    """Extract app ID from Google Play Store URL"""
    try:
        if 'id=' in app_url:
            return app_url.split('id=')[1].split('&')[0]
        else:
            raise ValueError("Invalid URL format")
    except Exception as e:
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
            count=100,  # Reduced batch size to avoid duplicates
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

def get_reviews_for_country(app_id, lang, country, country_name, result_queue, total_counter, min_reviews_per_country=3000):
    """Get reviews for a single country"""
    local_results = []
    continuation_token = None
    retry_count = 0
    max_retries = 5
    max_attempts = 50
    attempts = 0
    
    # Track review IDs to prevent duplicates
    seen_review_ids = set()
    
    while attempts < max_attempts and len(local_results) < min_reviews_per_country:
        try:
            time.sleep(1)  # Rate limiting
            
            filtered_reviews, continuation_token = fetch_reviews_batch(
                app_id, lang, country, country_name, continuation_token
            )
            
            # Efficient deduplication using review IDs
            new_reviews = []
            for review in filtered_reviews:
                review_id = f"{review['reviewId']}_{review['at']}"
                if review_id not in seen_review_ids:
                    seen_review_ids.add(review_id)
                    new_reviews.append(review)
            
            if new_reviews:
                local_results.extend(new_reviews)
                with total_counter.get_lock():
                    total_counter.value += len(new_reviews)
                    current_total = total_counter.value
                
                print(f"{country_name}: Added {len(new_reviews)} English reviews, Global Total: {current_total}")
            
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
                
            time.sleep(2)
    
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
    
    # Create a shared counter for total reviews
    from multiprocessing import Value
    total_counter = Value('i', 0)
    
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
                total_counter,
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
    
    final_total = len(all_reviews)
    print(f"\nReview collection completed!")
    print(f"Total reviews collected: {total_counter.value}")
    print(f"Final unique reviews after global deduplication: {final_total}")
    
    if final_total < min_reviews:
        print(f"Warning: Could not reach target review count ({min_reviews}), only got {final_total} unique reviews")
    
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
    """Analyze reviews using enhanced keyword analysis"""
    # Rating statistics
    ratings = [review['score'] for review in reviews_data]
    rating_counts = Counter(ratings)
    
    # Enhanced issue categories and keywords
    problem_keywords = {
        'Technical Issues': [
            'crash', 'crashes', 'crashing', 'force close', 'force stop', 'not working',
            'black screen', 'white screen', 'freeze', 'frozen', 'stuck', 'loading',
            'error', 'bug', 'glitch', 'broken', 'malfunction', 'not loading',
            'not responding', 'unresponsive', 'restart', 'shut down', 'close unexpectedly',
            'connection error', 'network error', 'server error', 'failed to load'
        ],
        'Performance Issues': [
            'slow', 'lag', 'laggy', 'stutter', 'stuttering', 'fps', 'frame rate',
            'frame drop', 'battery drain', 'battery life', 'overheat', 'overheating',
            'performance', 'optimization', 'heavy', 'memory', 'ram', 'cpu',
            'resource heavy', 'power consumption', 'slow loading', 'slow response',
            'delayed', 'choppy', 'sluggish', 'unoptimized'
        ],
        'User Interface': [
            'interface', 'ui', 'ux', 'design', 'layout', 'confusing', 'difficult',
            'hard to use', 'complicated', 'unintuitive', 'clunky', 'messy',
            'user friendly', 'user unfriendly', 'navigation', 'menu', 'buttons',
            'controls', 'touch response', 'touch screen', 'gesture', 'swipe',
            'tap', 'scroll', 'zoom', 'pinch', 'accessibility'
        ],
        'Advertisements': [
            'ad', 'ads', 'advert', 'advertising', 'commercial', 'popup', 'pop-up',
            'video ad', 'forced ad', 'too many ads', 'intrusive', 'annoying ads',
            'advertisement', 'sponsored', 'promotional', 'banner', 'interstitial',
            'skippable', 'unskippable', 'rewarded video', 'ad frequency',
            'aggressive ads', 'excessive ads', 'invasive ads'
        ],
        'Game Mechanics': [
            'gameplay', 'difficult', 'too hard', 'too easy', 'unfair', 'balance',
            'mechanics', 'controls', 'unplayable', 'impossible', 'challenging',
            'progression', 'level design', 'game design', 'reward system',
            'scoring', 'achievements', 'missions', 'quests', 'objectives',
            'tutorial', 'learning curve', 'difficulty spike', 'game balance',
            'power ups', 'abilities', 'skills'
        ],
        'Content & Features': [
            'feature', 'content', 'missing', 'need', 'should add', 'suggestion',
            'limited', 'lack of', 'more levels', 'more content', 'repetitive',
            'boring', 'short', 'incomplete', 'empty', 'bare', 'basic',
            'not enough', 'variety', 'diversity', 'options', 'customization',
            'personalization', 'game modes', 'multiplayer', 'single player'
        ],
        'Account & Login': [
            'account', 'login', 'sign in', 'register', 'authentication', 'password',
            'facebook', 'google', 'connection', 'offline', 'online', 'server',
            'profile', 'save', 'progress', 'cloud save', 'sync', 'synchronization',
            'data transfer', 'backup', 'restore', 'recovery', 'verification',
            'social features', 'friends list', 'social media'
        ],
        'Monetization': [
            'price', 'expensive', 'cost', 'pay', 'payment', 'money', 'coin',
            'premium', 'subscription', 'purchase', 'microtransaction', 'pay to win',
            'free to play', 'f2p', 'p2w', 'whale', 'gems', 'currency', 'bundle',
            'offer', 'deal', 'value', 'worth', 'pricing', 'in-app purchase',
            'paywall', 'monetization'
        ],
        'Updates & Compatibility': [
            'update', 'version', 'latest', 'old', 'newer', 'compatibility',
            'android', 'ios', 'phone', 'device', 'support', 'incompatible',
            'not compatible', 'system requirements', 'os version', 'upgrade',
            'downgrade', 'rollback', 'patch', 'bug fix', 'maintenance',
            'deprecated', 'legacy', 'outdated'
        ],
        'Storage & Data': [
            'storage', 'space', 'size', 'memory', 'large', 'data', 'download',
            'cache', 'mb', 'gb', 'huge', 'backup', 'save', 'data usage',
            'bandwidth', 'offline mode', 'storage space', 'file size',
            'installation size', 'update size', 'disk space', 'storage requirement',
            'cloud storage', 'local storage'
        ],
        'Social & Community': [
            'chat', 'community', 'social', 'friends', 'multiplayer', 'co-op',
            'cooperative', 'competitive', 'pvp', 'team', 'clan', 'guild',
            'communication', 'voice chat', 'text chat', 'emotes', 'messaging',
            'social features', 'player interaction', 'community features',
            'matchmaking', 'lobby', 'party system'
        ],
        'Customer Support': [
            'support', 'help', 'service', 'response', 'contact', 'feedback',
            'report', 'ticket', 'bug report', 'customer service', 'assistance',
            'unresponsive', 'no reply', 'ignored', 'poor support', 'help desk',
            'support team', 'technical support', 'customer care', 'service desk',
            'response time', 'resolution'
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
    text = ' '.join(review['content'].lower() for review in reviews_data)
    
    # Remove common English words and special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Create WordCloud object with optimized settings
    wc = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=200,
        max_font_size=100,
        random_state=42,
        prefer_horizontal=0.7,
        collocations=True,
        normalize_plurals=False
    )
    
    # Generate word cloud
    wc.generate(text)
    
    # Create and save figure
    plt.figure(figsize=(12, 8))
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
    
    # Create figure with adjusted size - make it taller
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot grid with different sizes
    gs = plt.GridSpec(1, 2, width_ratios=[1, 2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    
    fig.patch.set_facecolor('white')
    
    # Plot rating distribution
    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())
    bars1 = ax1.bar(ratings, counts, color='#2ecc71')
    ax1.set_title('Rating Distribution', fontsize=14, pad=20)
    ax1.set_xlabel('Rating', fontsize=12)
    ax1.set_ylabel('Number of Reviews', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11)
    
    # Plot topic distribution
    topics = list(topic_details.keys())
    topic_counts = [details['count'] for details in topic_details.values()]
    
    # Prepare topic names with better formatting
    if all(isinstance(topic, int) for topic in topics):
        # For BERTopic results, format as "Topic N: keyword1, keyword2"
        topic_names = [f"Topic {tid}: {', '.join(details['keywords'][:3])}"
                      for tid, details in topic_details.items()]
    else:
        # For keyword analysis results, keep category names as is
        topic_names = topics
    
    # Sort by count and get top 15 topics
    sorted_indices = np.argsort(topic_counts)[::-1][:15]  # Limit to top 15 topics
    topic_names = [topic_names[i] for i in sorted_indices]
    topic_counts = [topic_counts[i] for i in sorted_indices]
    
    # Create horizontal bars
    bars2 = ax2.barh(range(len(topic_counts)), topic_counts, color='#3498db')
    ax2.set_title('Top Issues Distribution', fontsize=14, pad=20)
    ax2.set_xlabel('Number of Reviews', fontsize=12)
    ax2.set_ylabel('Issues', fontsize=12)
    
    # Set y-ticks
    ax2.set_yticks(range(len(topic_names)))
    ax2.set_yticklabels(topic_names, fontsize=11)
    ax2.tick_params(axis='x', which='major', labelsize=11)
    
    # Add grid
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars2:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f' {int(width)}',
                ha='left', va='center', fontsize=11)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def analyze_demographics(reviews_data):
    """Analyze demographic information from reviews"""
    demographics = {
        'countries': Counter(),
        'versions': Counter(),
        'devices': Counter()
    }
    
    for review in reviews_data:
        demographics['countries'][review['country']] += 1
        if review.get('reviewCreatedVersion'):
            demographics['versions'][review['reviewCreatedVersion']] += 1
        if review.get('device'):
            demographics['devices'][review['device']] += 1
    
    return demographics

def create_demographics_chart(demographics, output_path):
    """Create pie charts for demographic information"""
    fig = plt.figure(figsize=(20, 6))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Colors
    colors = plt.cm.Pastel1(np.linspace(0, 1, 9))
    
    # Country distribution
    ax1 = fig.add_subplot(gs[0])
    country_data = dict(demographics['countries'].most_common(6))
    if len(demographics['countries']) > 6:
        other_count = sum(count for country, count in demographics['countries'].items() 
                         if country not in country_data)
        country_data['Others'] = other_count
    
    ax1.pie(country_data.values(), labels=country_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Review Distribution by Country', pad=20)
    
    # Version distribution
    ax2 = fig.add_subplot(gs[1])
    version_data = dict(demographics['versions'].most_common(6))
    if len(demographics['versions']) > 6:
        other_count = sum(count for version, count in demographics['versions'].items() 
                         if version not in version_data)
        version_data['Others'] = other_count
    
    ax2.pie(version_data.values(), labels=version_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Review Distribution by App Version', pad=20)
    
    # Device distribution
    ax3 = fig.add_subplot(gs[2])
    device_data = dict(demographics['devices'].most_common(6))
    if len(demographics['devices']) > 6:
        other_count = sum(count for device, count in demographics['devices'].items() 
                         if device not in device_data)
        device_data['Others'] = other_count
    
    ax3.pie(device_data.values(), labels=device_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax3.set_title('Review Distribution by Device', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

def main(app_url):
    """Main function"""
    try:
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
        demographics_output = f'{formatted_app_id}_demographics_{timestamp}.png'
        
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
            
            # Ensure all required columns exist
            required_columns = [
                'score', 'content', 'thumbsUpCount', 'reviewCreatedVersion',
                'at', 'country', 'topic_keywords'
            ]
            
            # Add missing columns with None values
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Export to Excel with only existing columns
            df.to_excel(excel_output, index=False, columns=required_columns)
            
            # Create visualizations
            print(f"Generating analysis charts: {chart_output}")
            create_visualizations(rating_counts, topic_details, chart_output)
            
            print(f"Generating word cloud: {wordcloud_output}")
            create_word_cloud(reviews_data, wordcloud_output)
            
            # Create demographic analysis
            print("Analyzing demographics...")
            demographics = analyze_demographics(reviews_data)
            print(f"Generating demographics charts: {demographics_output}")
            create_demographics_chart(demographics, demographics_output)
            
            print("\nAnalysis completed!")
            print(f"Review data saved to: {excel_output}")
            print(f"Analysis charts saved to: {chart_output}")
            print(f"Word cloud saved to: {wordcloud_output}")
            print(f"Demographics charts saved to: {demographics_output}")
            
            # Print topic analysis results
            print("\nMain Issue Topics Analysis:")
            for topic_id, details in topic_details.items():
                print(f"\nTopic {topic_id}:")
                print(f"Keywords: {', '.join(details['keywords'])}")
                print(f"Review count: {details['count']}")
                print("Example reviews:")
                for i, example in enumerate(details['examples'], 1):
                    print(f"{i}. {example[:200]}...")
            
            # Print demographic summary
            print("\nDemographic Summary:")
            print("\nTop Countries:")
            for country, count in demographics['countries'].most_common(5):
                print(f"{country}: {count} reviews ({count/len(reviews_data)*100:.1f}%)")
            
            print("\nTop App Versions:")
            for version, count in demographics['versions'].most_common(5):
                print(f"Version {version}: {count} reviews ({count/len(reviews_data)*100:.1f}%)")
            
            print("\nTop Devices:")
            for device, count in demographics['devices'].most_common(5):
                print(f"{device}: {count} reviews ({count/len(reviews_data)*100:.1f}%)")
                
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
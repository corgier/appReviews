import sys
from google_play_scraper import Sort, reviews
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
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

def download_nltk_resources():
    """下载必要的NLTK资源"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def extract_app_id(url):
    """从Google Play URL中提取应用ID"""
    try:
        # 处理转义字符
        url = url.replace('\\?', '?').replace('\\=', '=')
        print(f"处理后的URL: {url}")  # 调试信息
        
        # 尝试多种模式匹配
        patterns = [
            r'id=([^&\s]+)',  # 标准格式
            r'details/([^/?&\s]+)',  # 替代格式
            r'apps/([^/?&\s]+)'  # 简短格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                app_id = match.group(1)
                print(f"找到应用ID: {app_id}")  # 调试信息
                return app_id
        
        raise ValueError("无法从URL中提取应用ID")
    except Exception as e:
        print(f"URL处理过程中出错: {str(e)}")  # 调试信息
        raise ValueError(f"无效的Google Play URL: {str(e)}")

def is_english(text):
    """检查文本是否为英文"""
    try:
        return detect(text) == 'en'
    except (langdetect.lang_detect_exception.LangDetectException, Exception):
        return False

def fetch_reviews_batch(app_id, lang, country, country_name, continuation_token=None):
    """获取一批评论"""
    try:
        response, next_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.MOST_RELEVANT,
            count=150,  # 每批150条评论
            continuation_token=continuation_token
        )
        
        # 只保留3星及以下的英文评论
        filtered_reviews = []
        for review in response:
            if review['score'] <= 3 and is_english(review['content']):
                review['country'] = country_name  # 添加国家信息
                filtered_reviews.append(review)
                
        return filtered_reviews, next_token
        
    except Exception as e:
        print(f"获取{country_name}评论批次时出错: {str(e)}")
        return [], None

def get_reviews_for_country(app_id, lang, country, country_name, result_queue, min_reviews_per_country=3000):
    """获取单个国家的评论"""
    local_results = []
    continuation_token = None
    retry_count = 0
    max_retries = 5
    max_attempts = 50
    attempts = 0
    
    while attempts < max_attempts and len(local_results) < min_reviews_per_country:
        try:
            time.sleep(2)  # 延迟以避免限制
            
            filtered_reviews, continuation_token = fetch_reviews_batch(
                app_id, lang, country, country_name, continuation_token
            )
            
            # 去重（基于评论内容）
            existing_contents = {r['content'] for r in local_results}
            unique_reviews = [r for r in filtered_reviews if r['content'] not in existing_contents]
            
            local_results.extend(unique_reviews)
            print(f"{country_name}: 新增 {len(unique_reviews)} 条英文评论，当前: {len(local_results)} 条")
            
            if not continuation_token or not filtered_reviews:
                print(f"{country_name}没有更多评论可获取")
                break
                
            attempts += 1
            
        except Exception as e:
            retry_count += 1
            print(f"获取{country_name}评论时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
            
            if retry_count >= max_retries:
                print(f"{country_name}达到最大重试次数")
                break
                
            time.sleep(3)
    
    # 将结果放入队列
    result_queue.put((country_name, local_results))
    return len(local_results)

def get_reviews(app_id, min_reviews=10000):
    """并行获取多个国家的评论"""
    # 目标国家和语言组合（都使用英语）
    lang_country_pairs = [
        ('en_US', 'us', 'United States'),
        ('en_GB', 'gb', 'United Kingdom'),
        ('en_CA', 'ca', 'Canada'),
        ('en_AU', 'au', 'Australia'),
        ('en_IN', 'in', 'India'),
        ('en_PH', 'ph', 'Philippines'),
    ]
    
    result_queue = Queue()
    all_results = []
    min_reviews_per_country = min_reviews // len(lang_country_pairs)
    
    print(f"开始并行获取英文评论，每个国家目标获取 {min_reviews_per_country} 条评论...")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=len(lang_country_pairs)) as executor:
        # 创建future对象
        future_to_country = {
            executor.submit(
                get_reviews_for_country,
                app_id,
                lang,
                country,
                country_name,
                result_queue,
                min_reviews_per_country
            ): country_name
            for lang, country, country_name in lang_country_pairs
        }
        
        # 等待所有任务完成
        for future in as_completed(future_to_country):
            country_name = future_to_country[future]
            try:
                review_count = future.result()
                print(f"{country_name}评论获取完成，获取到 {review_count} 条评论")
            except Exception as e:
                print(f"{country_name}评论获取失败: {str(e)}")
    
    # 从队列中收集所有结果
    while not result_queue.empty():
        country_name, reviews_data = result_queue.get()
        all_results.extend(reviews_data)
    
    # 全局去重
    unique_reviews = []
    seen_contents = set()
    for review in all_results:
        if review['content'] not in seen_contents:
            seen_contents.add(review['content'])
            unique_reviews.append(review)
    
    # 按评分排序
    unique_reviews.sort(key=lambda x: x['score'])
    
    print(f"\n评论获取完成！总计: {len(unique_reviews)} 条")
    if len(unique_reviews) < min_reviews:
        print(f"警告: 未能达到目标评论数量 ({min_reviews} 条)，仅获取到 {len(unique_reviews)} 条评论")
    
    return unique_reviews

def analyze_reviews(reviews_data):
    """Analyze review content and extract key issues"""
    # Rating statistics
    ratings = [review['score'] for review in reviews_data]
    rating_counts = Counter(ratings)
    
    # Define common issue keywords (English)
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
    
    # Count occurrences of each issue type
    problem_counts = {category: 0 for category in problem_keywords}
    
    for review in reviews_data:
        content = review['content'].lower()
        for category, keywords in problem_keywords.items():
            if any(keyword in content for keyword in keywords):
                problem_counts[category] += 1
    
    return rating_counts, problem_counts

def create_word_cloud(reviews_data, output_path):
    """创建词云图"""
    # 合并所有评论文本
    text = ' '.join([review['content'] for review in reviews_data])
    
    # 使用结巴分词
    words = jieba.cut(text)
    word_space_split = ' '.join(words)
    
    # 创建词云对象
    wc = WordCloud(
        font_path='/System/Library/Fonts/Arial Unicode.ttf',  # Use Arial Unicode font
        width=1000,
        height=700,
        background_color='white',
        max_words=200,
        max_font_size=100,
        random_state=42
    )
    
    # 生成词云
    wc.generate(word_space_split)
    
    # 创建图形
    plt.figure(figsize=(10, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    
    # 保存词云图
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(rating_counts, problem_counts, output_path):
    """Create visualization charts"""
    # Set font and style
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # Set colors
    bar_colors = ['#2ecc71', '#3498db']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Rating distribution chart
    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())
    bars1 = ax1.bar(ratings, counts, color=bar_colors[0])
    ax1.set_title('Rating Distribution', fontsize=12, pad=15)
    ax1.set_xlabel('Rating', fontsize=10)
    ax1.set_ylabel('Number of Reviews', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Problem type distribution chart
    problems = list(problem_counts.keys())
    problem_nums = list(problem_counts.values())
    bars2 = ax2.bar(problems, problem_nums, color=bar_colors[1])
    ax2.set_title('Problem Type Distribution', fontsize=12, pad=15)
    ax2.set_xlabel('Problem Type', fontsize=10)
    ax2.set_ylabel('Mentions', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main(app_url):
    """主函数"""
    try:
        # 下载NLTK资源
        download_nltk_resources()
        
        # 获取应用ID
        app_id = extract_app_id(app_url)
        print(f"开始获取应用 {app_id} 的评论...")
        
        # 获取评论
        print("正在获取评论...")
        reviews_data = get_reviews(app_id, min_reviews=10000)  # 设置最小评论数为10000
        
        if not reviews_data:
            print("未找到任何3星及以下的英文评论")
            return
            
        print(f"成功获取 {len(reviews_data)} 条评论，开始分析...")
        
        # 创建DataFrame
        df = pd.DataFrame(reviews_data)
        
        # 分析评论
        print("正在分析评论...")
        rating_counts, problem_counts = analyze_reviews(reviews_data)
        
        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建输出文件名
        excel_output = f'app_reviews_{timestamp}.xlsx'
        chart_output = f'review_analysis_{timestamp}.png'
        wordcloud_output = f'review_wordcloud_{timestamp}.png'
        
        # 导出到Excel
        print(f"正在导出到Excel: {excel_output}")
        if len(df) > 0:
            # 选择要导出的列，包括国家信息
            df.to_excel(excel_output, index=False, columns=['score', 'content', 'thumbsUpCount', 'reviewCreatedVersion', 'at', 'country'])
            
            # 创建可视化
            print(f"正在生成分析图表: {chart_output}")
            create_visualizations(rating_counts, problem_counts, chart_output)
            
            # 创建词云图
            print(f"正在生成词云图: {wordcloud_output}")
            create_word_cloud(reviews_data, wordcloud_output)
            
            print("\n分析完成！")
            print(f"评论数据已保存到: {excel_output}")
            print(f"分析图表已保存到: {chart_output}")
            print(f"词云图已保存到: {wordcloud_output}")
        else:
            print("警告: 没有数据可以导出")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python analyze_app_reviews.py <Google Play应用URL>")
        sys.exit(1)
    
    main(sys.argv[1]) 
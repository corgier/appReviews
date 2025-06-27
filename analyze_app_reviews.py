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

def get_reviews(app_id, max_retries=5, min_reviews=10000):
    """获取应用的所有3星及以下评论"""
    result = []
    page_size = 150
    max_attempts = 200  # 增加最大尝试次数以获取更多评论
    
    # 目标国家和语言组合
    lang_country_pairs = [
        ('en_US', 'us', 'United States'),
        # ('en_IN', 'in', 'India'),
        # ('en_PH', 'ph', 'Philippines'),
    ]
    
    try:
        # 遍历不同的国家
        for lang, country, country_name in lang_country_pairs:
            print(f"\n开始获取{country_name}的评论...")
            continuation_token = None
            local_attempts = 0
            retry_count = 0
            
            while local_attempts < max_attempts and len(result) < min_reviews:
                try:
                    # 添加延迟以避免被限制
                    time.sleep(2)
                    
                    response, continuation_token = reviews(
                        app_id,
                        lang=lang,
                        country=country,
                        sort=Sort.MOST_RELEVANT,  # 按相关性排序
                        count=page_size,
                        continuation_token=continuation_token
                    )
                    
                    local_attempts += 1
                    
                    # 检查响应是否有效
                    if not response:
                        print(f"警告: 获取到空响应 ({country_name})")
                        break
                    
                    # 只保留3星及以下的评论
                    filtered_reviews = [review for review in response if review['score'] <= 3]
                    
                    # 去重（基于评论内容）
                    existing_contents = {r['content'] for r in result}
                    unique_reviews = [r for r in filtered_reviews if r['content'] not in existing_contents]
                    
                    result.extend(unique_reviews)
                    print(f"{country_name}: 新增 {len(unique_reviews)} 条评论 (≤3星)，当前总计: {len(result)} 条")
                    
                    # 如果已经获取足够多的评论，就退出
                    if len(result) >= min_reviews:
                        print(f"已达到目标评论数量: {len(result)}")
                        break
                    
                    # 如果没有更多评论，就尝试下一个国家
                    if not continuation_token:
                        print(f"{country_name}没有更多评论可获取")
                        break
                except Exception as e:
                    retry_count += 1
                    print(f"获取{country_name}评论时出错 (尝试 {retry_count}/{max_retries}): {str(e)}")
                    
                    if retry_count >= max_retries:
                        print(f"达到最大重试次数，切换到下一个国家")
                        break
                    
                    time.sleep(3)  # 出错后等待更长时间
            
            print(f"{country_name}评论获取完成，当前总计: {len(result)} 条")
            
            if len(result) >= min_reviews:
                break
        
        print(f"\n评论获取完成！总计: {len(result)} 条")
        if len(result) < min_reviews:
            print(f"警告: 未能达到目标评论数量 ({min_reviews} 条)，仅获取到 {len(result)} 条评论")
        
        # 按评分排序
        result.sort(key=lambda x: x['score'])
        return result
        
    except Exception as e:
        print(f"获取评论过程中发生严重错误: {str(e)}")
        if result:  # 如果已经获取到一些评论，仍然返回
            print(f"返回已获取的 {len(result)} 条评论")
            return result
        raise

def analyze_reviews(reviews_data):
    """分析评论内容，提取关键问题"""
    # 基于评分统计
    ratings = [review['score'] for review in reviews_data]
    rating_counts = Counter(ratings)
    
    # 定义常见问题关键词（英文）
    problem_keywords = {
        'Crashes': ['crash', 'crashes', 'crashing', 'force close', 'stop working', 'not working'],
        'Performance': ['slow', 'lag', 'loading', 'stuck', 'freeze', 'frozen', 'performance'],
        'UI/UX': ['interface', 'ui', 'design', 'confusing', 'difficult', 'hard to use'],
        'Ads': ['ad', 'ads', 'advert', 'advertising', 'commercial', 'popup'],
        'Bugs': ['bug', 'glitch', 'issue', 'problem', 'broken', 'not working'],
        'Features': ['feature', 'missing', 'need', 'should add', 'suggestion'],
        'Account': ['account', 'login', 'sign in', 'register', 'authentication'],
        'Price': ['price', 'expensive', 'cost', 'pay', 'payment', 'money'],
        'Updates': ['update', 'version', 'latest', 'old', 'newer'],
        'Storage': ['storage', 'space', 'size', 'memory', 'large']
    }
    
    # 统计各类问题出现次数
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
        font_path='/System/Library/Fonts/PingFang.ttc',  # macOS 中文字体
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
    """创建可视化图表"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    plt.style.use('default')  # 使用默认样式
    
    # 设置颜色
    bar_colors = ['#2ecc71', '#3498db']
    
    # 创建一个新的图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')  # 设置图表背景为白色
    
    # 评分分布图
    ratings = list(rating_counts.keys())
    counts = list(rating_counts.values())
    bars1 = ax1.bar(ratings, counts, color=bar_colors[0])
    ax1.set_title('评分分布', fontsize=12, pad=15)
    ax1.set_xlabel('评分', fontsize=10)
    ax1.set_ylabel('评论数量', fontsize=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 为柱状图添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 问题类型分布图
    problems = list(problem_counts.keys())
    problem_nums = list(problem_counts.values())
    bars2 = ax2.bar(problems, problem_nums, color=bar_colors[1])
    ax2.set_title('问题类型分布', fontsize=12, pad=15)
    ax2.set_xlabel('问题类型', fontsize=10)
    ax2.set_ylabel('提及次数', fontsize=10)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 为柱状图添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # 调整布局，确保标签不被截断
    plt.tight_layout()
    
    # 保存图表
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
            print("未找到任何3星及以下评论")
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
            df.to_excel(excel_output, index=False, columns=['score', 'content', 'thumbsUpCount', 'reviewCreatedVersion', 'at'])
            
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
# appReviews
This Python script is designed to scrape negative reviews of a specified app from the Google Play Store.
# Google Play 应用评论分析工具

这是一个用于分析 Google Play 应用商店评论的工具，它可以帮助开发者了解用户的反馈和可能的改进点。

## 功能特点

- 自动抓取指定应用的所有非5星评论
- 将评论数据导出到Excel文件
- 生成评分分布和问题类型分布的可视化图表
- 支持中文评论分析
- 自动识别常见问题类型（如崩溃、性能、界面等）

## 安装要求

1. Python 3.7 或更高版本
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 从 Google Play 商店获取您想要分析的应用链接，格式类似：
   `https://play.google.com/store/apps/details?id=com.example.app`

2. 运行脚本：
```bash
python analyze_app_reviews.py <Google Play应用URL>
```

例如：
```bash
python analyze_app_reviews.py "https://play.google.com/store/apps/details?id=com.example.app"
```

## 输出文件

脚本会生成两个文件：

1. `app_reviews_[时间戳].xlsx`：包含所有非5星评论的Excel文件
   - 评分
   - 评论内容
   - 点赞数
   - 应用版本
   - 评论时间

2. `review_analysis_[时间戳].png`：包含两个图表
   - 评分分布图
   - 问题类型分布图

## 问题类型分类

工具会自动识别以下类型的问题：
- 崩溃：应用崩溃、闪退等问题
- 性能：卡顿、加载速度等问题
- 界面：UI设计、操作体验等问题
- 广告：广告相关投诉
- Bug：各类错误和故障
- 功能：功能建议和需求
- 账号：登录注册相关问题
- 价格：收费相关问题 
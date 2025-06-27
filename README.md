# Google Play Review Analyzer
A Python package for analyzing user reviews from the Google Play Store, helping developers understand user feedback and identify potential areas for improvement.

## Features

- Automatically fetches reviews with ratings ≤ 3 stars from multiple English-speaking countries
- Implements dual analysis methods:
  1. Advanced topic modeling using BERTopic (primary method)
  2. Keyword-based analysis (fallback method)
- Exports review data to Excel files with topic/issue information
- Generates visualization charts for rating distribution and issue analysis
- Creates word clouds from review content
- Supports parallel processing for faster data collection

## Requirements

1. Python 3.7 or higher
2. Required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Get the Google Play Store URL of the app you want to analyze. The URL format should be:
   `https://play.google.com/store/apps/details?id=com.example.app`

2. Run the script:
```bash
python analyze_app_reviews.py <Google Play App URL>
```

Example:
```bash
python analyze_app_reviews.py "https://play.google.com/store/apps/details?id=com.example.app"
```

## Output Files

The script generates three files with timestamps in the format `YYYYMMDD_HHMMSS`. Files are named using the app ID with dots replaced by hyphens (e.g., for `com.example.app`, files will be named `com-example-app_*`):

1. `[app-id]_reviews_[timestamp].xlsx`: Excel file containing all reviews with:
   - Rating
   - Review content
   - Likes count
   - App version
   - Review date
   - Country of origin
   - Topic/Issue keywords (automatically identified)

2. `[app-id]_analysis_[timestamp].png`: Analysis charts showing:
   - Rating distribution
   - Issue distribution with key terms

3. `[app-id]_wordcloud_[timestamp].png`: Word cloud visualization of common terms in reviews

## Analysis Methods

The tool implements a dual-analysis approach for maximum reliability:

### Primary Method: BERTopic Analysis
- Uses state-of-the-art topic modeling combining:
  - BERT embeddings for semantic understanding
  - UMAP for dimensionality reduction
  - HDBSCAN for clustering
- Automatically discovers topics without predefined categories
- Requires minimum 10 reviews and sufficient review length
- Provides:
  - Key terms that characterize each topic
  - Number of reviews in each topic
  - Representative review examples
  - Topic distribution visualization

### Fallback Method: Keyword Analysis
Automatically activates when:
- Review count is less than 10
- Reviews are too short (less than 3 words)
- Topic modeling fails to produce meaningful results
- System resources are insufficient

Uses predefined categories:
- Technical Issues
- Performance
- User Interface
- Advertisements
- Game Mechanics
- Content & Features
- Account & Login
- Monetization
- Updates & Compatibility
- Storage & Data

Each category includes:
- Relevant keywords
- Review count
- Example reviews

## Performance Optimizations

- Automatic GPU detection and utilization when available
- Low memory mode for resource-constrained environments
- Parallel processing for data collection
- Efficient text preprocessing and filtering
- Automatic handling of short or invalid reviews

## Supported Countries

The tool collects reviews from major English-speaking countries:
- United States
- United Kingdom
- Canada
- Australia
- India
- Philippines

## Note

- The tool uses parallel processing to speed up data collection
- Reviews are automatically filtered to include only English language content
- The target is to collect up to 10,000 reviews (distributed across countries)
- Duplicate reviews are automatically filtered out
- Topic modeling requires sufficient GPU memory for optimal performance
- CPU mode is automatically used if GPU is not available
- The analysis method automatically adapts to your data quality and quantity
- Invalid or too short reviews are automatically filtered out
- The tool provides detailed progress and error information during analysis 
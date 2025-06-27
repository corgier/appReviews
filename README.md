# Google Play Store Review Analysis Tool

A powerful Python tool for analyzing Google Play Store app reviews, providing insights into user feedback, sentiment analysis, and demographic information.

## Features

- **Automated Review Collection**
  - Collects reviews from multiple countries in parallel
  - Focuses on critical reviews (3-star and below)
  - Supports multiple languages with automatic English detection
  - Efficient deduplication and data validation

- **Advanced Analysis**
  - Topic modeling using BERTopic for intelligent issue categorization
  - Fallback to keyword-based analysis when needed
  - Rating distribution analysis
  - Word cloud generation for visual trend analysis
  - Demographic analysis (country, app version, device distribution)

- **Performance Optimizations**
  - GPU acceleration when available
  - Efficient batch processing for large datasets
  - Memory-optimized processing for large review sets
  - Parallel review collection from multiple regions

- **Comprehensive Visualizations**
  - Interactive charts for rating distribution
  - Topic/issue distribution visualization
  - Word cloud for common terms
  - Demographic pie charts showing:
    - Review distribution by country
    - App version distribution
    - Device distribution

- **Data Export**
  - Excel export with detailed review information
  - Organized file naming with timestamps
  - Comprehensive analysis results in both visual and text formats

## Requirements

```
google-play-scraper
pandas
matplotlib
nltk
jieba
wordcloud
numpy
langdetect
bertopic
sentence-transformers
torch
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the script with a Google Play Store app URL:

```bash
python analyze_app_reviews.py "https://play.google.com/store/apps/details?id=com.example.app"
```

## Output Files

The tool generates several output files with timestamps:

- `{app-id}_reviews_{timestamp}.xlsx`: Detailed review data
- `{app-id}_analysis_{timestamp}.png`: Rating and topic distribution charts
- `{app-id}_wordcloud_{timestamp}.png`: Word cloud visualization
- `{app-id}_demographics_{timestamp}.png`: Demographic analysis charts

## Analysis Results

The tool provides:

1. **Rating Analysis**
   - Distribution of ratings
   - Focus on critical reviews

2. **Topic Analysis**
   - Key issues identified through topic modeling
   - Common keywords and themes
   - Representative review examples

3. **Demographic Analysis**
   - Country distribution of reviews
   - App version distribution
   - Device distribution

4. **Detailed Reports**
   - Excel file with all review data
   - Console output with key findings
   - Visual representations of all analyses

## Notes

- The tool focuses on English reviews by default
- Requires internet connection for review collection
- GPU acceleration is automatically used if available
- Processing time depends on the number of reviews and available computing resources

## License

MIT License 
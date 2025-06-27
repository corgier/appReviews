# Google Play Review Analysis Tool
# appReviews
This Python script is designed to scrape negative reviews of a specified app from the Google Play Store.
A powerful tool for analyzing user reviews from the Google Play Store, helping developers understand user feedback and identify potential areas for improvement.

## Features

- Automatically fetches reviews with ratings ≤ 3 stars from multiple English-speaking countries
- Exports review data to Excel files
- Generates visualization charts for rating distribution and issue types
- Creates word clouds from review content
- Supports parallel processing for faster data collection
- Performs intelligent issue categorization and analysis

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

The script generates three files with timestamps:

1. `app_reviews_[timestamp].xlsx`: Excel file containing all reviews with:
   - Rating
   - Review content
   - Likes count
   - App version
   - Review date
   - Country of origin

2. `review_analysis_[timestamp].png`: Analysis charts showing:
   - Rating distribution
   - Issue type distribution

3. `review_wordcloud_[timestamp].png`: Word cloud visualization of common terms in reviews

## Issue Categories

The tool automatically categorizes reviews into the following issue types:

- Technical Issues: Crashes, freezes, and technical problems
- Performance: Speed, lag, and resource usage
- User Interface: UI/UX design and usability
- Advertisements: Ad-related complaints
- Game Mechanics: Gameplay and control issues
- Content & Features: Content availability and feature requests
- Account & Login: Authentication and account-related issues
- Monetization: Pricing and payment concerns
- Updates & Compatibility: Version updates and device compatibility
- Storage & Data: Storage space and data management

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
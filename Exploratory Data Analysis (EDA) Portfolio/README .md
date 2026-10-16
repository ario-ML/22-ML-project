# Netflix Content Exploratory Data Analysis (EDA)

## Overview
This project explores the Netflix Movies and TV Shows dataset to uncover patterns in content type, ratings, genres, countries of production, and release trends over time. The goal was to practice a full, real-world EDA workflow: cleaning messy data, handling missing values, engineering new features, and extracting meaningful insights through visualization.

## Dataset
- **Source:** [Netflix Movies and TV Shows (Kaggle)](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Size:** 8,807 titles, 12 original columns
- **Features:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in` (genres), `description`

## Questions Explored
- What is the balance between Movies and TV Shows on the platform?
- How has the volume of content added to Netflix changed over time?
- Which countries produce the most content?
- What content ratings dominate the platform?
- What are the most common genres?
- How does movie duration vary by content rating?
- Has the distribution of ratings shifted over the years?

## Tools Used
Python, Pandas, Matplotlib, Seaborn, Jupyter Notebook

## Data Cleaning & Feature Engineering
- **Missing values:** `director`, `cast`, and `country` were filled with `"Unknown"` instead of being dropped, preserving ~96% of the original rows. Rows with missing `date_added`, `rating`, or `duration` (fewer than 20 total) were dropped, since imputing them wasn't meaningful.
- **Date parsing:** `date_added` was converted from text to a proper `datetime` type, and `year_added` / `month_added` columns were extracted for time-based analysis.
- **Duration split:** The original `duration` column mixed two different units ("90 min" for movies, "2 Seasons" for TV shows). It was split into two clean numeric columns: `duration_minutes` and `duration_seasons`.
- **Multi-value columns:** `country` and `listed_in` (genre) often contained multiple comma-separated values per row (e.g., "United States, India"). These were split and exploded into individual values for accurate counting, rather than treating each combination as a single category.

## Key Findings

1. **Content type:** Netflix's library is dominated by Movies (6,126 titles, ~70%) over TV Shows (2,664 titles, ~30%).

2. **Growth trend:** Content additions grew steadily from 2015 to 2019, peaking in 2019 (~2,000 titles added). The apparent drop in 2021 is misleading — the dataset only covers content added through September 2021, not a real decline in output.

3. **Top producing countries:** The United States leads by a wide margin (3,681 titles), followed by India (1,046) and the United Kingdom (805). About 829 titles have no listed country.

4. **Ratings:** TV-MA is the most common content rating, indicating Netflix's library skews toward adult audiences. This pattern has remained consistent across years — the proportion of ratings hasn't meaningfully shifted over time, only the total volume of content has grown.

5. **Genres:** "International Movies" is the most frequent genre tag (2,752), reflecting Netflix's global content strategy, followed by Dramas (2,426) and Comedies (1,674).

6. **Duration patterns:** Most movies run 90–100 minutes, with a right-skewed distribution (a long tail of longer films). Adult-rated content (NC-17, R, TV-MA) tends to run longer than children's content (TV-Y, TV-Y7), likely reflecting differences in genre composition (e.g., short animated content vs. feature-length dramas).

## Project Structure
```
├── Untitled.ipynb        # Main analysis notebook
├── netflix_titles.csv    # Raw dataset (from Kaggle)
└── README.md             # This file
```

## How to Run
1. Install dependencies: `pip install pandas numpy matplotlib seaborn`
2. Download the dataset from Kaggle and place it in the project folder
3. Open the notebook in Jupyter and run all cells

## Next Steps
Potential extensions to this project include: analyzing genre trends by country, building a simple recommendation logic based on genre similarity, or comparing Netflix's catalog composition to other streaming platforms.

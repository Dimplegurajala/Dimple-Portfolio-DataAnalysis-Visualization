# Web Crawling and Analysis of Indian Unicorn Startups

## Overview
This project scrapes data about top Indian unicorn startups from [StartupTalky](https://startuptalky.com/top-unicorn-startups-india/) and performs statistical and predictive analyses. It demonstrates the end-to-end pipeline of **web crawling**, **data storage**, and **data analysis**.

## Features
- Web scrape startup details (name, industry, valuation, etc.) using BeautifulSoup
- Store data in **SQLite database** and export to **CSV**
- Visualize distributions and industry trends with Seaborn/Matplotlib
- Perform **linear regression** (valuation vs founding year/unicorn entry year)
- Perform **logistic regression** (predicting startup status)

## Tech Stack
- Python (requests, BeautifulSoup, pandas, seaborn, matplotlib)
- SQLite for local storage
- Statsmodels & scikit-learn for regression/classification

## Workflow
1. Crawl startup data from StartupTalky
2. Save to SQLite database (`unicorn_data.db`)
3. Export to CSV (`unicorn_data.csv`)
4. Perform descriptive analysis (histograms, count plots, heatmaps)
5. Conduct regression and classification modeling

## Visualizations
- Founding year distribution
- Industry-wise startup count
- Regression analysis: Founding year vs valuation
- Confusion matrix for classification

## Results
- Identified trends in unicorn formation by year and industry
- Regression revealed valuation patterns based on founding year
- Logistic regression achieved moderate classification accuracy for startup status


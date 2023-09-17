import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Uncomment the following lines if you haven't downloaded the NLTK data before
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# Load data from CSV file (replace 'news.csv' with your CSV file path)
data = pd.read_csv('news.csv')

# Sentiment analysis
sent = SentimentIntensityAnalyzer()

# Create new columns for sentiment score and sentiment label
data['SentimentScore'] = data['review'].apply(lambda x: sent.polarity_scores(x)['compound'])

def categorize_sentiment(compound):
    if compound > 0:
        return 'Positive'
    elif compound < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment'] = data['SentimentScore'].apply(categorize_sentiment)

# Print the first few rows of the DataFrame with the added columns
print(data[['review', 'Sentiment', 'SentimentScore']].head())

# You can save the modified DataFrame back to a CSV file if needed
data.to_csv('news_with_sentiment.csv', index=False)

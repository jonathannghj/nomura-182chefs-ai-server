from newsapi import NewsApiClient
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv



# NewsAPI client
newsapi = NewsApiClient(api_key='d55f2b26a0ec4273b6c34390b3eb1e74')

# # Set NLTK data path to the local directory
# nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.download('vader_lexicon')

# if nltk_data_path not in nltk.data.path:
#     nltk.data.path.append(nltk_data_path)

# VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()



def get_sentiment_score(ticker, number_of_days=7):
    """
    Get sentiment score for a stock ticker from news articles

    Parameters
    ==========
    * ticker (str): stock ticker
    * number_of_days (int): number of days to look back for news articles

    Returns
    =======
    * float: average sentiment score
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=number_of_days)
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    all_articles = newsapi.get_everything(q=ticker,
                                          from_param=from_date,
                                          to=to_date,
                                          language='en',
                                          sort_by='popularity')
    
    articles = pd.DataFrame(all_articles['articles'])

    def get_sentiment(text):
        if text:
            sentiment = sid.polarity_scores(text)
            return sentiment['compound']
        else:
            return 0

    articles['sentiment'] = articles['description'].apply(get_sentiment)
    average_sentiment = articles['sentiment'].mean()
    return average_sentiment

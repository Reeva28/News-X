import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import json
import re
from concurrent.futures import ThreadPoolExecutor
import time
import yfinance as yf

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')

class FinancialDataAgent:
    def __init__(self):
        # API Keys
        self.news_api_key = "NEWS_API_KEY"
        self.finnhub_api_key = "FINNHUB_API_KEY"
        self.alpha_vantage_api_key = "ALPHA_VANTAGE_API_KEY"  # Replace with your Alpha Vantage API key

        # Initialize sentiment analyzers
        self.sia = SentimentIntensityAnalyzer()
        self.nlp = pipeline("sentiment-analysis")

        # Data storage
        self.news_data = []
        self.stock_data = None
        self.sentiment_scores = {}
        self.ticker = None
        self.company_info = None

        # Company data dictionary (kept the same as before)
        self.company_data = {
            "AAPL": {"name": "Apple Inc.", "industry": "Consumer Electronics", "sector": "Technology"},
            "MSFT": {"name": "Microsoft Corporation", "industry": "Software", "sector": "Technology"},
            "AMZN": {"name": "Amazon.com, Inc.", "industry": "E-Commerce", "sector": "Consumer Cyclical"},
            "GOOGL": {"name": "Alphabet Inc.", "industry": "Internet Content & Information", "sector": "Technology"},
            "TSLA": {"name": "Tesla, Inc.", "industry": "Auto Manufacturers", "sector": "Consumer Cyclical"},
            "META": {"name": "Meta Platforms, Inc.", "industry": "Social Media", "sector": "Technology"},
            "NFLX": {"name": "Netflix, Inc.", "industry": "Entertainment", "sector": "Communication Services"},
            "NVDA": {"name": "NVIDIA Corporation", "industry": "Semiconductors", "sector": "Technology"},
            "JPM": {"name": "JPMorgan Chase & Co.", "industry": "Banks", "sector": "Financial Services"},
            "BAC": {"name": "Bank of America Corporation", "industry": "Banks", "sector": "Financial Services"},
            "DIS": {"name": "The Walt Disney Company", "industry": "Entertainment", "sector": "Communication Services"},
            "PFE": {"name": "Pfizer Inc.", "industry": "Pharmaceuticals", "sector": "Healthcare"},
            "JNJ": {"name": "Johnson & Johnson", "industry": "Pharmaceuticals", "sector": "Healthcare"},
            "KO": {"name": "The Coca-Cola Company", "industry": "Beverages", "sector": "Consumer Defensive"},
            "PEP": {"name": "PepsiCo, Inc.", "industry": "Beverages", "sector": "Consumer Defensive"},
            "WMT": {"name": "Walmart Inc.", "industry": "Retail", "sector": "Consumer Defensive"},
            "HD": {"name": "The Home Depot, Inc.", "industry": "Home Improvement Retail", "sector": "Consumer Cyclical"},
            "V": {"name": "Visa Inc.", "industry": "Credit Services", "sector": "Financial Services"},
            "MA": {"name": "Mastercard Incorporated", "industry": "Credit Services", "sector": "Financial Services"},
            "UNH": {"name": "UnitedHealth Group Incorporated", "industry": "Healthcare Plans", "sector": "Healthcare"},
            "INTC": {"name": "Intel Corporation", "industry": "Semiconductors", "sector": "Technology"},
            "IBM": {"name": "International Business Machines", "industry": "Information Technology", "sector": "Technology"},
            "GS": {"name": "Goldman Sachs Group Inc.", "industry": "Investment Banking", "sector": "Financial Services"},
            "MRK": {"name": "Merck & Co., Inc.", "industry": "Pharmaceuticals", "sector": "Healthcare"},
            "CRM": {"name": "Salesforce, Inc.", "industry": "Software", "sector": "Technology"}
        }

    def set_user_input(self, ticker=None):
        """Set the user input data"""
        self.ticker = ticker
        self.fetch_company_info()
        self.news_data = []
        self.stock_data = None
        self.sentiment_scores = {}
        return True

    def fetch_company_info(self):
        """Fetch company information"""
        if not self.ticker:
            return

        try:
            # Use yfinance as a fallback for company info
            ticker = yf.Ticker(self.ticker)
            info = ticker.info

            if info:
                self.company_info = {
                    "name": info.get('longName', f"Unknown ({self.ticker})"),
                    "industry": info.get('industry', 'Unknown'),
                    "sector": info.get('sector', 'Unknown')
                }
            else:
                self.company_info = {
                    "name": self.ticker,
                    "industry": "Unknown",
                    "sector": "Unknown"
                }
        except Exception as e:
            print(f"Error fetching company info: {str(e)}")
            self.company_info = {
                "name": self.ticker,
                "industry": "Unknown",
                "sector": "Unknown"
            }

    def fetch_finnhub_news(self):
        """Fetch news from Finnhub"""
        if not self.ticker:
            return []

        try:
            # Get news for the last week
            today = datetime.today().strftime('%Y-%m-%d')
            week_ago = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            url = f"https://finnhub.io/api/v1/company-news?symbol={self.ticker}&from={week_ago}&to={today}&token={self.finnhub_api_key}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                articles = response.json()
                
                # Process and format articles
                processed_articles = []
                for article in articles[:10]:  # Limit to 10 articles
                    # Convert timestamp to readable datetime
                    pub_datetime = datetime.fromtimestamp(article.get('datetime', time.time()))
                    
                    # Sentiment analysis
                    headline = article.get('headline', '')
                    sentiment = self.sia.polarity_scores(headline)
                    
                    processed_article = {
                        'title': headline,
                        'description': article.get('summary', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', 'Finnhub'),
                        'published_at': pub_datetime,
                        'sentiment_score': sentiment['compound']
                    }
                    processed_articles.append(processed_article)
                
                return processed_articles
            else:
                print(f"Failed to fetch news from Finnhub: {response.status_code}")
                return []
        
        except Exception as e:
            print(f"Error fetching Finnhub news: {str(e)}")
            return []

    def analyze_news_sentiment(self, news_data):
        """Analyze sentiment of news articles"""
        sentiments = []
        
        for article in news_data:
            # Use VADER for sentiment scoring
            headline = article.get('title', '')
            description = article.get('description', '')
            
            # Combine headline and description for more context
            text = headline + " " + description
            
            # Get sentiment score
            sentiment = self.sia.polarity_scores(text)
            sentiments.append(sentiment['compound'])
        
        # Calculate average sentiment
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            return avg_sentiment
        return 0

    def determine_market_sentiment(self, sentiment_score):
        """Determine market sentiment based on sentiment score"""
        if sentiment_score > 0.2:
            return "Strongly Positive"
        elif sentiment_score > 0:
            return "Slightly Positive"
        elif sentiment_score < -0.2:
            return "Strongly Negative"
        elif sentiment_score < 0:
            return "Slightly Negative"
        else:
            return "Neutral"

def get_stock_summary(ticker, api_key):
    """Fetch basic stock information using Alpha Vantage"""
    try:
        # Fetch overview information
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
        overview_response = requests.get(overview_url)
        overview_data = overview_response.json()

        # Fetch quote information
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()

        # Extract quote information
        quote_info = quote_data.get('Global Quote', {})

        # Validate and extract data
        def safe_get(dict_obj, key, default='N/A'):
            return dict_obj.get(key, default)

        return {
            'current_price': safe_get(quote_info, '05. price'),
            'open': safe_get(quote_info, '02. open'),
            'previous_close': safe_get(quote_info, '08. previous close'),
            'high': safe_get(quote_info, '03. high'),
            'low': safe_get(quote_info, '04. low'),
            '52_week_high': safe_get(overview_data, '52WeekHigh'),
            '52_week_low': safe_get(overview_data, '52WeekLow'),
            'sector': safe_get(overview_data, 'Sector', 'Unknown'),
            'industry': safe_get(overview_data, 'Industry', 'Unknown')
        }
    except Exception as e:
        st.error(f"Error fetching stock information for {ticker}: {e}")
        
        # Fallback to hardcoded company data
        agent = FinancialDataAgent()
        fallback_info = agent.company_data.get(ticker, {})
        
        return {
            'current_price': 'N/A',
            'open': 'N/A',
            'previous_close': 'N/A',
            'high': 'N/A',
            'low': 'N/A',
            '52_week_high': 'N/A',
            '52_week_low': 'N/A',
            'sector': fallback_info.get('sector', 'Unknown'),
            'industry': fallback_info.get('industry', 'Unknown')
        }

def determine_stock_trend(stock_data):
    """Determine stock trend based on moving averages"""
    if stock_data is None or stock_data.empty:
        return "Neutral"
    
    try:
        # Calculate moving averages
        stock_data['MA50'] = stock_data['Close'].rolling(window=50).mean()
        stock_data['MA200'] = stock_data['Close'].rolling(window=200).mean()
        
        # Get latest values
        latest_close = stock_data['Close'].iloc[-1]
        ma50 = stock_data['MA50'].iloc[-1]
        ma200 = stock_data['MA200'].iloc[-1]
        
        if ma50 > ma200 and latest_close > ma50:
            return "Strong Bullish"
        elif ma50 > ma200:
            return "Mild Bullish"
        elif ma50 < ma200 and latest_close < ma50:
            return "Strong Bearish"
        elif ma50 < ma200:
            return "Mild Bearish"
        else:
            return "Neutral"
    except Exception as e:
        st.warning(f"Unable to determine stock trend: {e}")
        return "Unable to Determine"
    
def investment_recommendation(trend, market_sentiment, volatility):
    """Generate investment recommendation"""
    if trend.startswith("Strong Bullish") and market_sentiment.startswith("Strongly Positive"):
        return "Strong Buy - Extremely Positive Indicators"
    elif trend.startswith("Bullish") and market_sentiment.startswith("Positive"):
        return "Buy - Positive Market Conditions"
    elif trend == "Neutral" and market_sentiment == "Neutral":
        return "Hold - Wait for Clearer Signals"
    elif trend.startswith("Bearish") or market_sentiment.startswith("Negative"):
        return "Sell or Caution - Negative Market Signals"
    else:
        return "Consult Financial Advisor - Complex Market Conditions"

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Financial News Analyst",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )
    
    # Custom dark blue theme
    st.markdown("""
    <style>
    .reportview-container {
        background: #0E1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #1E2130;
    }
    body {
        color: white;
    }
    .stTextInput>div>div>input {
        color: white;
        background-color: #1E2130;
    }
    .stButton>button {
        color: white;
        background-color: #4A5568;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🔍 Financial News & Stock Analysis")
    
    # Input for ticker or company name
    user_input = st.text_input("Enter Stock Ticker (e.g., AAPL) or Company Name", "")
    
    if user_input:
        # Ticker mapping
        ticker_map = {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Amazon": "AMZN",
            "Google": "GOOGL",
            "Meta": "META",
            "Tesla": "TSLA",
            "Nvdia": "NVDA"
        }
        
        # Determine ticker
        ticker = ticker_map.get(user_input.title(), user_input.upper())
        
        # Initialize Financial Data Agent
        agent = FinancialDataAgent()
        agent.set_user_input(ticker=ticker)
        
        # Fetch and display stock summary
        # Note: Replace 'YOUR_ALPHA_VANTAGE_API_KEY' with your actual Alpha Vantage API key
        stock_summary = get_stock_summary(ticker, agent.alpha_vantage_api_key)
        
        if stock_summary:
            # Stock Information Columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Stock Information")
                st.write(f"Current Price: ${stock_summary['current_price']}")
                st.write(f"Open: ${stock_summary['open']}")
                st.write(f"Previous Close: ${stock_summary['previous_close']}")
                st.write(f"Today's High: ${stock_summary['high']}")
                st.write(f"Today's Low: ${stock_summary['low']}")
            
            with col2:
                st.subheader("Company Details")
                st.write(f"Sector: {stock_summary['sector']}")
                st.write(f"Industry: {stock_summary['industry']}")
                st.write(f"52 Week High: ${stock_summary['52_week_high']}")
                st.write(f"52 Week Low: ${stock_summary['52_week_low']}")
            
            # Fetch News
            news_data = agent.fetch_finnhub_news()
            
            # Analyze News Sentiment
            news_sentiment_score = agent.analyze_news_sentiment(news_data)
            market_sentiment = agent.determine_market_sentiment(news_sentiment_score)
            
            # Fetch Stock Data for Trend Analysis
            stock = yf.Ticker(ticker)
            try:
                stock_data = stock.history(period="6mo")
                if stock_data.empty:
                    st.warning("No historical stock data available.")
                    stock_trend = "Unable to Determine"
                else:
                    stock_trend = determine_stock_trend(stock_data)
            except Exception as e:
                st.error(f"Error retrieving stock data: {e}")
                stock_trend = "Unable to Determine"
            
            # Investment Recommendation
            recommendation = investment_recommendation(
                stock_trend, 
                market_sentiment, 
                stock_data['Close'].std() if 'stock_data' in locals() and not stock_data.empty else 0
            )
            
            # Display News
            st.subheader("Latest News")
            for article in news_data[:5]:
                st.markdown(f"**{article['title']}**")
                st.write(f"Published: {article['published_at']}")
                st.write(f"Sentiment: {article['sentiment_score']:.2f}")
                st.write(f"[Read More]({article['url']})")
                st.write("---")
            
            # Display Sentiment and Trend
            st.subheader("Market Analysis")
            st.write(f"Market Sentiment: {market_sentiment}")
            st.write(f"Stock Trend: {stock_trend}")
            
            # Investment Recommendation
            st.subheader("Investment Recommendation")
            st.write(recommendation)

if __name__ == "__main__":
    main()
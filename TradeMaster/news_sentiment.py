from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import torch
import pickle
import requests
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class SentimentAnalyzer:
    def __init__(self, model_name="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = ["negative", "neutral", "positive"]

    def analyze(self, headline: str) -> dict:
        inputs = self.tokenizer(headline, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        pred_label = self.labels[pred_idx]
        confidence = probs[pred_idx]

        # Map to percentile scale
        if pred_label == "negative":
            percentile = (1 - confidence) * 0.3333
        elif pred_label == "neutral":
            if probs[0] > probs[2]:  # leaning negative
                percentile = 0.3334 + confidence / 2 * (0.5 - 0.3334)
            else:  # leaning positive
                percentile = 0.5 + confidence / 2 * (0.6667 - 0.5)
        else:  # positive
            percentile = 0.6667 + confidence * (1.0 - 0.6667)

        return {
            "headline": headline,
            "sentiment": pred_label,
            "confidence": round(confidence, 4),
            "percentile": round(percentile, 4),
        }
    
class NewsFetcher:
    def __init__(self, api_key_path="seekingalpha_api_key.pkl"):
        with open(api_key_path, "rb") as handle:
            self.api_key = pickle.load(handle)
        self.base_host = "seeking-alpha.p.rapidapi.com"
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": self.base_host,
        }

    def fetch_news_list(self, symbol: str, page: int = 1, size: int = 5) -> dict:
        url = f"https://{self.base_host}/news/v2/list-by-symbol"
        params = {"id": symbol, "page": str(page), "size": str(size)}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

class TickerNewsSentiment:
    def __init__(self, news_fetcher: NewsFetcher, sentiment_analyzer: SentimentAnalyzer):
        """
        news_fetcher: instance of NewsFetcher
        sentiment_analyzer: instance of SentimentAnalyzer
        """
        self.news_fetcher = news_fetcher
        self.sentiment_analyzer = sentiment_analyzer

    def get_news_with_sentiment(self, symbol: str, page: int = 1, size: int = 5) -> pd.DataFrame:
        """
        Fetch news for a ticker and return a DataFrame with publish time, title, and sentiment score.
        """
        raw_news = self.news_fetcher.fetch_news_list(symbol, page=page, size=size)

        records = []
        for item in raw_news.get("data", []):
            headline = item.get("attributes", {}).get("title", "")
            publish_time = item.get("attributes", {}).get("publishOn", "")

            sentiment_result = self.sentiment_analyzer.analyze(headline)

            records.append({
                "publish_time": publish_time,
                "title": headline,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "percentile": sentiment_result["percentile"]
            })
        rec_df = pd.DataFrame(records)
        rec_df['publish_time'] = pd.to_datetime(rec_df['publish_time'].str[:19])
        
        return rec_df

class StockDataFetcher:
    def __init__(self):
        pass

    def get_stock_data(self, stock_name: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
        """
        Uses the Yahoo Finance API to extract historical stock price data
        for a given stock and date range.
        """
        data = yf.Ticker(stock_name).history(start=start, end=end, interval=interval)
        data.reset_index(inplace=True)
        data["Datetime"] = data["Datetime"].apply(lambda x: str(x)[:-6])
        data["Datetime"] = pd.to_datetime(data["Datetime"])
        data["timestamp"] = data["Datetime"].dt.to_pydatetime()
        data["target"] = data["Close"].pct_change() * 10000
        data = data.dropna().reset_index(drop=True)
        return data


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    news_fetcher = NewsFetcher()
    ticker_sentiment = TickerNewsSentiment(news_fetcher, analyzer)

    # Fetch enriched news table
    df = ticker_sentiment.get_news_with_sentiment("AAPL", page=1, size=3)
    print(df)


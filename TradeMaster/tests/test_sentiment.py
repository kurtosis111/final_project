import unittest
from news_sentiment import SentimentAnalyzer

class TestNewsSentiment(unittest.TestCase):
    def test_positive_sentiment(self):
        text = "The company reported excellent earnings and strong growth."
        analyzer = SentimentAnalyzer()
        score = analyzer.analyze(text)['percentile']
        self.assertGreater(score, 0.66, "Expected positive sentiment score.")
    
    def test_negative_sentiment(self):
        text = "The market creashed and investors lost confidence"
        analyzer = SentimentAnalyzer()
        score = analyzer.analyze(text)['percentile']
        self.assertLess(score, 0.34, "Expected negative sentiment score.")
    
    def test_neutral_sentiment(self):
        text = "The company announced a new product launch"
        analyzer = SentimentAnalyzer()
        score = analyzer.analyze(text)['percentile']
        self.assertGreater(score, 0.33, "Value should be >=0.33")
        self.assertLessEqual(score, 0.67, "Value should be <=0.67")


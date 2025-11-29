import unittest
from predictor import TabPFNStockPredictor
import pandas as pd

class TestPredictor(unittest.TestCase):
    def setUp(self):
        # Initialize predictor in test mode (mock if needed)
        self.predictor = TabPFNStockPredictor()

    def test_predict_returns_df(self):
        sample_data = "AAPL"
        result = self.predictor.fit_predict(sample_data)
        self.assertIsInstance(result, pd.DataFrame, "Prediction should be a Pandas DataFrame")
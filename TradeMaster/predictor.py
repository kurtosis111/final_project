import warnings
warnings.filterwarnings("ignore")

import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode
from tabpfn_time_series import FeatureTransformer, TimeSeriesDataFrame
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)

# import tabpfn_client
# token = tabpfn_client.get_access_token()
# tabpfn_client.set_access_token(token)

class TabPFNStockPredictor:
    def __init__(self, lags=None):
        # api_key = os.getenv("TABPFN_API_KEY")
        # if api_key:
        #     # If key is provided, set it for the client
        #     os.environ["TABPFN_API_KEY"] = api_key
        
        if lags is None:
            lags = [1, 2, 5, 10, 20, 60, 120, 250]
        self.lags = lags
        self.selected_features = [
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ]
        self.feature_transformer = FeatureTransformer(self.selected_features)
        self.predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
        self.eval_df = None

    def _prepare_data(self, ticker: str, period="5y", interval="1d"):
        df = yf.download(ticker, period=period, interval=interval)
        df = df[['Close']].rename(columns={'Close': 'price'})
        df['ret'] = np.log(df['price']).diff()
        df = df.dropna()

        for L in self.lags:
            df[f'ret_lag_{L}'] = df['ret'].shift(L)

        df['ret_next'] = df['ret'].shift(-1)

        df_single = df.copy()
        df_single.columns = df.columns.get_level_values(0)
        df_single['item_id'] = ticker
        df_single['timestamp'] = df_single.index

        return df_single

    def fit_predict(self, ticker: str):
        df_single = self._prepare_data(ticker)

        features = [c for c in df_single.columns if c.startswith('ret_lag_')]
        target = 'ret_next'

        X = df_single[['item_id', 'timestamp'] + features]
        y = df_single[target]

        # Chronological split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, shuffle=False
        )

        # Construct TabPFN TimeSeriesDataFrame
        train_tsd = X_train.copy()
        train_tsd['target'] = y_train.values
        train_tsd = TimeSeriesDataFrame(train_tsd.set_index(['item_id', 'timestamp']))

        test_tsd = X_test.copy()
        test_tsd['target'] = np.nan
        test_tsd = TimeSeriesDataFrame(test_tsd.set_index(['item_id', 'timestamp']))

        train_tsd, test_tsd = self.feature_transformer.transform(train_tsd, test_tsd)

        pred_df = self.predictor.predict(train_tsd, test_tsd)
        pred_df = pred_df.reset_index().rename(columns={'mean': 'pred'})

        # Evaluation DataFrame
        eval_df = pred_df[['timestamp', 'target']].rename(columns={'target': 'pred'})
        eval_df.insert(1, 'target', y_test.values)
        eval_df['pos'] = round(eval_df['pred']*1000)
        eval_df['cumu_ret_act'] = ((eval_df['pos'].mean() * eval_df['target'] + 1).cumprod() - 1)
        eval_df['cumu_ret_pred'] = ((eval_df['pos'] * eval_df['target'] + 1).cumprod() - 1)

        self.eval_df = eval_df
        return eval_df

    def most_recent_prediction(self):
        # if self.eval_df is None:
        #     raise ValueError("Run fit_predict first.")
        # return self.eval_df.iloc[-1]['timestamp'], round(self.eval_df.iloc[-1]['pred']*1000)
        if self.eval_df is None:
            raise ValueError("Run fit_predict first.")
        row = self.eval_df.iloc[-1]
        return {
            "timestamp": row["timestamp"],
            "pred": round(row["pred"]*1000)
        }

    def plot_cumulative_returns(self):
        if self.eval_df is None:
            raise ValueError("Run fit_predict first.")
        plt.figure(figsize=(10, 4))
        plt.plot(self.eval_df['timestamp'], self.eval_df['cumu_ret_act'], label='Cumulative Actual Return')
        plt.plot(self.eval_df['timestamp'], self.eval_df['cumu_ret_pred'], label='Cumulative Predicted Return')
        plt.legend()
        plt.title("Cumulative Actual vs Predicted Returns")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Return")
        plt.show()


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    predictor1 = TabPFNStockPredictor()
    eval_df = predictor1.fit_predict("TSLA")
    print("Most recent prediction:", predictor1.most_recent_prediction())
    predictor1.plot_cumulative_returns()


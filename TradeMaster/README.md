# TradeMaster 📈

TradeMaster is a modular, containerized dashboard for **financial market prediction and sentiment analysis**.  
It integrates **time series forecasting (TabPFN)** with **financial news sentiment analysis (DistilRoBERTa)** to provide actionable insights for portfolio management and trading research.

---

## 🚀 Features
- **Financial News Sentiment Analysis**  
  Uses [`mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis) to classify headlines into `positive`, `neutral`, or `negative`.

- **Time Series Prediction**  
  Powered by **TabPFN** for zero-shot forecasting of stock returns and market signals.

- **Interactive Dashboard**  
  Built with **Dash + Plotly**, providing multi-graph layouts, percentile tables, and intuitive color-coded outputs.

- **Containerized Deployment**  
  Dockerized for reproducibility and easy sharing. Includes pinned requirements for conflict-free builds.

- **Modular Workflow**  
  Clear separation of sentiment, prediction, and visualization modules for maintainability.

---

## 📂 Project Structure
TradeMaster/
├── app.py                     # Dash dashboard entrypoint
├── predictor.py               # TabPFN-based stock prediction pipeline
├── news_sentiment.py          # Financial news sentiment analysis + data fetchers
│
├── requirements.txt           # Pinned Python dependencies
├── Dockerfile                 # Image build instructions
├── docker-compose.yml         # Service orchestration (ports, volumes, credentials)
│
├── tests/
│   ├── test_sentiment.py      # Unit tests for sentiment analyzer
│   ├── test_predictor.py      # Unit tests for TabPFN predictor
│   └── test_app.py            # Integration tests for Dash app
│
└── README.md                  # Project overview and usage instructions


---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/TradeMaster.git
   cd TradeMaster
2. ** Build Docker Image
    docker-compose build
3. ** Run the app
    docker-compose up

## 🧪 Testing
  python3 -m unittests discover -s tests

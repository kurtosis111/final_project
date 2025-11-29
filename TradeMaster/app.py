import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go

from predictor import TabPFNStockPredictor
from news_sentiment import NewsFetcher, SentimentAnalyzer, TickerNewsSentiment

# Initialize components
predictor = TabPFNStockPredictor()
news_fetcher = NewsFetcher()
sentiment_analyzer = SentimentAnalyzer()
ticker_sentiment = TickerNewsSentiment(news_fetcher, sentiment_analyzer)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Prediction Dashboard"),
    html.Div([
        dcc.Input(id="ticker-input", type="text", placeholder="Enter tickers (comma-separated)", style={"width":"50%"}),
        html.Button("Run Prediction", id="submit-button", n_clicks=0)
    ]),
    html.Div(id="output-container"),
    html.Div(id="graphs-container"),   # multiple graphs will be injected here
    html.Div(id="news-container")
])


@app.callback(
    [Output("graphs-container", "children"),
     Output("output-container", "children"),
     Output("news-container", "children")],
    [Input("submit-button", "n_clicks")],
    [State("ticker-input", "value")]
)
def update_output(n_clicks, tickers):
    if not tickers:
        return [], "Please enter tickers.", ""

    tickers = [t.strip().upper() for t in tickers.split(",")]
    graphs = []
    latest_preds = []
    news_blocks = []

    for ticker in tickers:
        # Run prediction
        eval_df = predictor.fit_predict(ticker)
        latest_pred = predictor.most_recent_prediction()

        # Build figure for this ticker
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["cumu_ret_act"],
                                 mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=eval_df["timestamp"], y=eval_df["cumu_ret_pred"],
                                 mode="lines", name="Predicted"))
        fig.update_layout(
            title=f"{ticker}: Cumulative Actual vs Predicted Returns",
            xaxis_title="Time",
            yaxis_title="Cumulative Return",
            legend_title="Type"
        )

        graphs.append(html.Div([
            dcc.Graph(figure=fig)
        ]))

        latest_preds.append(html.Div([
            html.H3(f"{ticker} Latest Prediction"),
            html.P(f"Timestamp: {latest_pred['timestamp']}"),
            html.P(f"Predicted Return: {latest_pred['pred']:.4f}")
        ]))

        # Fetch news + sentiment
        news_df = ticker_sentiment.get_news_with_sentiment(ticker, page=1, size=3)
        news_table = html.Table([
            html.Thead(html.Tr([html.Th("Publish Time"), html.Th("Title"), html.Th("Sentiment"), html.Th("Confidence"), html.Th("Percentile")]))
        ] + [
            html.Tr([
                html.Td(row["publish_time"]),
                html.Td(row["title"]),
                html.Td(row["sentiment"]),
                html.Td(row["confidence"]),
                html.Th(row["percentile"])
            ]) for _, row in news_df.iterrows()
        ])
        news_blocks.append(html.Div([html.H3(f"{ticker} Latest News"), news_table]))

    return graphs, latest_preds, news_blocks


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

# backend/live_streamer.py
import torch
from data_stream.fetch_market_data import fetch_ohlcv
from backend.huggingface_models import load_timeseries_transformer, predict_timeseries, load_cryptobert, predict_sentiment, ensemble_prediction

class LiveStreamer:
    def __init__(self):
        self.ts_model = load_timeseries_transformer()
        self.tokenizer, self.sentiment_model = load_cryptobert()

    def get_market_data(self, symbol="ETHUSDT", timeframe="1m", limit=60):
        df = fetch_ohlcv(symbol, timeframe, limit)
        ts_input = torch.tensor(df[['open','high','low','close','volume']].values, dtype=torch.float).unsqueeze(0)
        return df, ts_input

    def predict(self, symbol="ETHUSDT"):
        df, ts_input = self.get_market_data(symbol)
        ts_pred = torch.sigmoid(predict_timeseries(self.ts_model, ts_input))[0].item()

        # Example: get latest 3 news headlines
        sample_news = ["ETH surges as market rallies", "Binance updates crypto fees", "Bitcoin dominance falls"]
        sentiment_score = sum(predict_sentiment(self.tokenizer, self.sentiment_model, sample_news)) / len(sample_news)

        prediction = ensemble_prediction(ts_pred, sentiment_score)
        return prediction

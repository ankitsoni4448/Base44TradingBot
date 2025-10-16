# backend/api.py
from fastapi import FastAPI
from backend.live_streamer import LiveStreamer

app = FastAPI()
streamer = LiveStreamer()

@app.get("/predict/{symbol}")
def predict(symbol: str):
    result = streamer.predict(symbol)
    return {"symbol": symbol, "prediction": result}

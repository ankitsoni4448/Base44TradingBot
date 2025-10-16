# backend/predictor.py
import math
import numpy as np
import pandas as pd
from typing import Optional
from backend.indicators import compute_indicators
from backend.data_loader import load_historical

class Predictor:
    """
    Lightweight predictor that outputs UP / DOWN + confidence.
    - If a trained sklearn model exists it will use it (optional).
    - Otherwise it uses indicator rules (EMA crossover + MACD sign).
    """

    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        # Placeholder for a trained model — optional in future
        self.model = None

    def score_from_indicators(self, df: pd.DataFrame) -> float:
        """
        Compute a small heuristic score:
         + ema crossover: ema9 - ema21
         + macd_hist
         + rsi distance from 50
        Score roughly in range [-3, +3], positive => UP
        """
        if df is None or df.empty:
            return 0.0
        dfi = compute_indicators(df)
        # use last row
        last = dfi.iloc[-1]
        ema_diff = float(last.get("ema_9", 0.0) - last.get("ema_21", 0.0))
        macd_hist = float(last.get("macd_hist", 0.0))
        rsi = float(last.get("rsi_14", 50.0))
        # normalize with price scale
        price = float(last.get("close", 1.0)) or 1.0
        ema_score = np.tanh(ema_diff / (price + 1e-8)) * 2.0
        macd_score = np.tanh(macd_hist) * 1.5
        rsi_score = ((rsi - 50.0) / 50.0) * 1.0
        score = ema_score + macd_score + rsi_score
        return float(score)

    def predict_from_df(self, df: pd.DataFrame, timeframe: Optional[str] = "1m"):
        """
        Returns dictionary: { signal: "UP"/"DOWN", score: float, confidence: float }
        """
        try:
            if df is None or df.empty:
                return {"signal": "HOLD", "score": 0.0, "confidence": 0.5}

            score = self.score_from_indicators(df)
            # score -> probability via sigmoid mapping
            prob_up = 1.0 / (1.0 + math.exp(-3.0 * score))  # steeper sigmoid
            # map to confidence (0.5..0.99)
            confidence = 0.5 + (abs(prob_up - 0.5) * 0.99)
            signal = "UP" if prob_up >= 0.5 else "DOWN"

            return {
                "signal": signal,
                "score": float(round(score, 6)),
                "confidence": float(round(confidence, 3)),
                "timeframe": timeframe
            }
        except Exception as e:
            return {"signal": "HOLD", "score": 0.0, "confidence": 0.5, "error": str(e)}

    def predict(self, df: pd.DataFrame = None, timeframe: Optional[str] = "1m"):
        # If a trained model exists in the future, call it here.
        return self.predict_from_df(df, timeframe)

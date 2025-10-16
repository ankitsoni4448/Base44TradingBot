# backend/huggingface_models.py
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -------------------------
# Time Series Transformer
# -------------------------
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig

def load_timeseries_transformer(model_name="huggingface/time-series-transformer"):
    config = TimeSeriesTransformerConfig.from_pretrained(model_name)
    model = TimeSeriesTransformerForPrediction.from_pretrained(model_name, config=config)
    return model

def predict_timeseries(model, input_tensor):
    """
    input_tensor: torch.Tensor of shape [batch_size, seq_len, features]
    returns: predicted value
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.logits  # adjust depending on model

# -------------------------
# CryptoBERT Sentiment Analysis
# -------------------------
def load_cryptobert(model_name="kk08/CryptoBERT"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_sentiment(tokenizer, model, texts):
    """
    texts: list of strings (news headlines / tweets)
    returns: list of sentiment scores
    """
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings)
    probs = torch.softmax(outputs.logits, dim=1)
    return probs[:, 1].tolist()  # 1 = positive sentiment

# -------------------------
# Ensemble Function
# -------------------------
def ensemble_prediction(ts_pred, sentiment_score, threshold=0.5):
    """
    Combine time-series prediction with sentiment score.
    ts_pred: float (0-1 probability up)
    sentiment_score: float (0-1 probability positive)
    returns: "UP" or "DOWN"
    """
    combined_score = 0.7 * ts_pred + 0.3 * sentiment_score
    return "UP" if combined_score >= threshold else "DOWN"

# Base44/frontend_service.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="Base44 CryptoIDX (UP/DOWN) Dashboard", layout="wide")
st.title("Base44 CryptoIDX - UP / DOWN Dashboard")

BACKEND_URL = st.sidebar.text_input("Backend URL", "http://127.0.0.1:8000")
symbol = st.sidebar.selectbox("Symbol", ["ETHUSDT", "BTCUSDT"])
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h"])
refresh = st.sidebar.slider("Refresh (s)", 2, 30, 5)

placeholder = st.empty()
history = []

def fetch_prediction():
    try:
        r = requests.get(f"{BACKEND_URL}/predict", params={"symbol": symbol, "tf": timeframe}, timeout=5)
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def send_feedback(pred, actual, conf):
    payload = {"symbol": symbol, "timeframe": timeframe, "predicted": pred, "actual": actual, "confidence": conf}
    try:
        r = requests.post(f"{BACKEND_URL}/feedback", json=payload, timeout=5)
        return r.status_code == 200
    except:
        return False

st.sidebar.markdown("## Manual feedback (click after trade)")
col_left, col_right = st.columns(2)
with col_left:
    if st.button("Report ACTUAL UP"):
        if history:
            last = history[-1]
            send_feedback(last["signal"], "UP", last["confidence"])
            st.success("Feedback saved: ACTUAL UP")
with col_right:
    if st.button("Report ACTUAL DOWN"):
        if history:
            last = history[-1]
            send_feedback(last["signal"], "DOWN", last["confidence"])
            st.success("Feedback saved: ACTUAL DOWN")

st.markdown("### Live prediction stream")
while True:
    data = fetch_prediction()
    now = datetime.utcnow().strftime("%H:%M:%S")
    if "error" in data:
        placeholder.error(f"Error: {data['error']}")
        time.sleep(refresh)
        continue

    signal = data.get("signal", {})
    trend = signal.get("signal", "HOLD")
    conf = signal.get("confidence", 0.0)
    price = data.get("price")

    history.append({"time": now, "price": price, "signal": trend, "confidence": conf})

    df_hist = pd.DataFrame(history).tail(200)

    with placeholder.container():
        st.subheader(f"{symbol} — {timeframe} — {now}")
        col1, col2, col3 = st.columns([2,2,2])
        col1.metric("Price", f"{price:.6f}" if price else "N/A")
        col2.metric("Signal", trend)
        col3.metric("Confidence", f"{conf:.2f}")

        st.line_chart(df_hist.set_index("time")["price"])

        st.markdown("#### Last raw")
        st.json(signal)

    time.sleep(refresh)

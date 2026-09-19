import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

IST = ZoneInfo("Asia/Kolkata")
GROWW_BASE = "https://api.groww.in"
TOKEN = os.getenv("GROWW_ACCESS_TOKEN", "").strip()

app = FastAPI(title="Trading AI Tool", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def groww_headers():
    if not TOKEN:
        raise HTTPException(503, "GROWW_ACCESS_TOKEN is not configured on the server.")
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "X-API-VERSION": "1.0",
    }

def fetch_sensex_candles(days=5):
    end = datetime.now(IST).replace(tzinfo=None)
    start = end - timedelta(days=days)
    params = {
        "exchange": "BSE",
        "segment": "CASH",
        "groww_symbol": "BSE-SENSEX",
        "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
        "candle_interval": "15minute",
    }
    r = requests.get(
        f"{GROWW_BASE}/v1/historical/candles",
        params=params,
        headers=groww_headers(),
        timeout=15,
    )
    if r.status_code >= 400:
        raise HTTPException(r.status_code, f"Groww historical-data error: {r.text[:300]}")
    body = r.json()
    candles = body.get("payload", {}).get("candles", [])
    if not candles:
        raise HTTPException(502, "Groww returned no SENSEX candles.")
    rows = []
    for c in candles:
        if len(c) >= 5:
            rows.append({
                "time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]) if len(c) > 5 and c[5] is not None else 0,
            })
    return pd.DataFrame(rows)

def add_indicators(df):
    d = df.copy()
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    d["rsi"] = 100 - (100 / (1 + rs))
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["vol_ma20"] = d["volume"].rolling(20).mean()
    d["hh20"] = d["high"].rolling(20).max().shift(1)
    d["ll20"] = d["low"].rolling(20).min().shift(1)
    return d

def make_signal(df):
    d = add_indicators(df).dropna().copy()
    if len(d) < 30:
        raise HTTPException(502, "Not enough completed candles for the signal engine.")
    x = d.iloc[-1]
    score_long = 0
    score_short = 0

    if x.close > x.ema9: score_long += 1
    if x.ema9 > x.ema21: score_long += 1
    if 52 <= x.rsi <= 72: score_long += 1
    if x.close > x.hh20: score_long += 2
    if x.volume > x.vol_ma20: score_long += 1

    if x.close < x.ema9: score_short += 1
    if x.ema9 < x.ema21: score_short += 1
    if 28 <= x.rsi <= 48: score_short += 1
    if x.close < x.ll20: score_short += 2
    if x.volume > x.vol_ma20: score_short += 1

    side = "WAIT"
    score = max(score_long, score_short)
    if score_long >= 5 and score_long > score_short:
        side = "LONG"
    elif score_short >= 5 and score_short > score_long:
        side = "SHORT"

    atr = float(x.atr)
    entry = float(x.close)
    if side == "LONG":
        sl = entry - 1.2 * atr
        t1, t2, t3 = entry + 1.0*atr, entry + 2.0*atr, entry + 3.0*atr
    elif side == "SHORT":
        sl = entry + 1.2 * atr
        t1, t2, t3 = entry - 1.0*atr, entry - 2.0*atr, entry - 3.0*atr
    else:
        sl = t1 = t2 = t3 = None

    return {
        "asset": "SENSEX",
        "timeframe": "15m",
        "signal": side,
        "score": int(score),
        "long_score": int(score_long),
        "short_score": int(score_short),
        "candle_time": str(x["time"]),
        "price": round(entry, 2),
        "rsi": round(float(x.rsi), 2),
        "ema9": round(float(x.ema9), 2),
        "ema21": round(float(x.ema21), 2),
        "atr": round(atr, 2),
        "stop_loss": round(sl, 2) if sl is not None else None,
        "target1": round(t1, 2) if t1 is not None else None,
        "target2": round(t2, 2) if t2 is not None else None,
        "target3": round(t3, 2) if t3 is not None else None,
        "mode": "PAPER",
        "warning": "Signal engine is rule-based; it cannot guarantee exact entries/exits or profits."
    }

@app.get("/health")
def health():
    return {"ok": True, "groww_token_configured": bool(TOKEN), "mode": "PAPER"}

@app.get("/api/signal")
def signal():
    return make_signal(fetch_sensex_candles(5))

@app.get("/api/candles")
def candles(limit: int = Query(80, ge=20, le=200)):
    d = fetch_sensex_candles(5).tail(limit)
    return {"asset": "SENSEX", "timeframe": "15m", "candles": d.to_dict(orient="records")}

@app.get("/")
def home():
    return FileResponse("web/index.html")

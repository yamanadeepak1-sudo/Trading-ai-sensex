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

SUPPORTED_INTERVALS = {
    "1minute": 7, "2minute": 15, "3minute": 15, "5minute": 15,
    "10minute": 30, "15minute": 30, "30minute": 60,
    "1hour": 150, "4hour": 365, "1day": 1080,
    "1week": 3650, "1month": 3650,
}

def fetch_sensex_candles(interval="15minute"):
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Unsupported timeframe.")
    days = SUPPORTED_INTERVALS[interval]
    end = datetime.now(IST).replace(tzinfo=None)
    start = end - timedelta(days=days)
    params = {
        "exchange": "BSE",
        "segment": "CASH",
        "groww_symbol": "BSE-SENSEX",
        "start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": end.strftime("%Y-%m-%d %H:%M:%S"),
        "candle_interval": interval,
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

def make_signal(df, timeframe="15minute"):
    d = add_indicators(df).dropna().copy()
    if len(d) < 30:
        raise HTTPException(502, f"Not enough completed candles for {timeframe}.")
    x = d.iloc[-1]
    long_score = 0
    short_score = 0
    if x.close > x.ema9: long_score += 1
    if x.ema9 > x.ema21: long_score += 1
    if 52 <= x.rsi <= 72: long_score += 1
    if x.close > x.hh20: long_score += 2
    if x.volume > x.vol_ma20: long_score += 1
    if x.close < x.ema9: short_score += 1
    if x.ema9 < x.ema21: short_score += 1
    if 28 <= x.rsi <= 48: short_score += 1
    if x.close < x.ll20: short_score += 2
    if x.volume > x.vol_ma20: short_score += 1
    side = "WAIT"
    score = max(long_score, short_score)
    if long_score >= 5 and long_score > short_score:
        side = "LONG"
    elif short_score >= 5 and short_score > long_score:
        side = "SHORT"
    return {
        "asset": "SENSEX", "timeframe": timeframe, "signal": side,
        "score": int(score), "long_score": int(long_score), "short_score": int(short_score),
        "candle_time": str(x["time"]), "price": round(float(x.close), 2),
        "rsi": round(float(x.rsi), 2), "ema9": round(float(x.ema9), 2),
        "ema21": round(float(x.ema21), 2), "atr": round(float(x.atr), 2)
    }

@app.get("/api/multi-signal")
def multi_signal():
    timeframes = ["5minute", "10minute", "15minute", "30minute", "1hour"]
    results = []
    for tf in timeframes:
        results.append(make_signal(fetch_sensex_candles(tf), tf))
    long_votes = sum(1 for x in results if x["signal"] == "LONG")
    short_votes = sum(1 for x in results if x["signal"] == "SHORT")
    if long_votes >= 3 and long_votes > short_votes:
        overall = "LONG"
    elif short_votes >= 3 and short_votes > long_votes:
        overall = "SHORT"
    else:
        overall = "WAIT"
    entry_data = next((x for x in results if x["timeframe"] == "5minute"), results[0])
    entry = entry_data["price"]
    atr = entry_data["atr"]
    if overall == "LONG":
        sl = entry - 1.2 * atr
        t1, t2, t3 = entry + atr, entry + 2*atr, entry + 3*atr
    elif overall == "SHORT":
        sl = entry + 1.2 * atr
        t1, t2, t3 = entry - atr, entry - 2*atr, entry - 3*atr
    else:
        sl = t1 = t2 = t3 = None
    return {
        "asset": "SENSEX", "mode": "PAPER",
        "signal": overall, "votes": {"LONG": long_votes, "SHORT": short_votes, "WAIT": 5-long_votes-short_votes},
        "entry_timeframe": "5minute", "price": entry, "stop_loss": round(sl,2) if sl else None,
        "target1": round(t1,2) if t1 else None, "target2": round(t2,2) if t2 else None,
        "target3": round(t3,2) if t3 else None, "timeframes": results,
        "warning": "Multi-timeframe confirmation is rule-based and cannot guarantee exact entries, exits, or profits."
    }

@app.get("/health")
def health():
    return {"ok": True, "groww_token_configured": bool(TOKEN), "mode": "PAPER"}

@app.get("/api/signal")
def signal(interval: str = Query("15minute")):
    return make_signal(fetch_sensex_candles(interval), interval)

@app.get("/api/candles")
def candles(limit: int = Query(80, ge=20, le=200), interval: str = Query("15minute")):
    d = fetch_sensex_candles(interval).tail(limit)
    return {"asset": "SENSEX", "timeframe": interval, "candles": d.to_dict(orient="records")}

@app.get("/")
def home():
    return FileResponse("web/index.html")

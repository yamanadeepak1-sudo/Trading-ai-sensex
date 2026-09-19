import os
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

IST = ZoneInfo("Asia/Kolkata")
YAHOO_SYMBOL = "^BSESN"

app = FastAPI(title="Trading AI Tool", version="3.0.0 - Free Data")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_INTERVALS = {
    "1minute", "2minute", "3minute", "5minute", "10minute", "15minute",
    "30minute", "1hour", "4hour", "1day", "1week", "1month"
}

def resample_candles(df, rule):
    d = df.copy()
    d = d.set_index("time")
    out = d.resample(rule, origin="start_day", offset="15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])
    out["time"] = out.index.astype(str)
    return out.reset_index(drop=True)[
        ["time", "open", "high", "low", "close", "volume"]
    ]

def fetch_sensex_candles(interval="15minute"):
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(400, "Unsupported timeframe.")

    try:
        base_interval = "1m" if interval in {"1minute", "2minute", "3minute"} else "5m"
        period = "7d" if base_interval == "1m" else "60d"

        d = yf.download(
            YAHOO_SYMBOL,
            period=period,
            interval=base_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if d is None or d.empty:
            raise HTTPException(502, "Free market-data source returned no SENSEX candles.")

        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)

        d = d.reset_index()
        time_col = "Datetime" if "Datetime" in d.columns else "Date"
        d = d.rename(columns={
            time_col: "time", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        d = d[["time", "open", "high", "low", "close", "volume"]].dropna(
            subset=["open", "high", "low", "close"]
        )

        times = pd.to_datetime(d["time"])
        if times.dt.tz is None:
            times = times.dt.tz_localize(IST)
        else:
            times = times.dt.tz_convert(IST)
        d["time"] = times

        now = pd.Timestamp.now(tz=IST)
        cutoff = now.floor("5min") - pd.Timedelta(minutes=5)
        d = d[d["time"] <= cutoff]

        if d.empty:
            raise HTTPException(502, "No completed SENSEX candles available yet.")

        if interval == "1minute":
            return d.tail(500).assign(time=d.tail(500)["time"].astype(str))

        if interval == "2minute":
            return resample_candles(d, "2min").tail(500)

        if interval == "3minute":
            return resample_candles(d, "3min").tail(500)

        rules = {
            "5minute": "5min", "10minute": "10min", "15minute": "15min",
            "30minute": "30min", "1hour": "1h", "4hour": "4h",
            "1day": "1D", "1week": "1W", "1month": "1ME"
        }
        return resample_candles(d, rules[interval]).tail(500)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Free SENSEX data error: {str(e)[:250]}")

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
        "score": int(score), "long_score": int(long_score),
        "short_score": int(short_score), "candle_time": str(x["time"]),
        "price": round(float(x.close), 2), "rsi": round(float(x.rsi), 2),
        "ema9": round(float(x.ema9), 2), "ema21": round(float(x.ema21), 2),
        "atr": round(float(x.atr), 2)
    }

@app.get("/api/multi-signal")
def multi_signal():
    timeframes = ["5minute", "10minute", "15minute", "30minute", "1hour"]
    results = [make_signal(fetch_sensex_candles(tf), tf) for tf in timeframes]
    long_votes = sum(1 for x in results if x["signal"] == "LONG")
    short_votes = sum(1 for x in results if x["signal"] == "SHORT")
    if long_votes >= 3 and long_votes > short_votes:
        overall = "LONG"
    elif short_votes >= 3 and short_votes > long_votes:
        overall = "SHORT"
    else:
        overall = "WAIT"
    entry_data = next(x for x in results if x["timeframe"] == "5minute")
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
        "asset": "SENSEX", "mode": "PAPER", "signal": overall,
        "votes": {"LONG": long_votes, "SHORT": short_votes,
                  "WAIT": 5-long_votes-short_votes},
        "entry_timeframe": "5minute", "price": entry,
        "stop_loss": round(sl, 2) if sl is not None else None,
        "target1": round(t1, 2) if t1 is not None else None,
        "target2": round(t2, 2) if t2 is not None else None,
        "target3": round(t3, 2) if t3 is not None else None,
        "timeframes": results,
        "warning": "Free market data may be delayed and is not a broker execution feed. Signals are rule-based paper research only."
    }

@app.get("/health")
def health():
    return {"ok": True, "data_source": "Yahoo Finance", "symbol": YAHOO_SYMBOL, "mode": "PAPER"}

@app.get("/api/signal")
def signal(interval: str = Query("15minute")):
    return make_signal(fetch_sensex_candles(interval), interval)

@app.get("/api/candles")
def candles(limit: int = Query(80, ge=20, le=200), interval: str = Query("15minute")):
    d = fetch_sensex_candles(interval).tail(limit)
    return {"asset": "SENSEX", "timeframe": interval,
            "candles": d.to_dict(orient="records")}

@app.get("/")
def home():
    return FileResponse("web/index.html")

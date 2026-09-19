# Trading AI Tool v2 — SENSEX 15m

## What this version does
- Mobile-friendly Chrome dashboard.
- FastAPI backend.
- Pulls 15-minute BSE SENSEX historical candles from Groww.
- Calculates EMA(9), EMA(21), RSI(14), ATR(14), volume average and 20-candle breakout.
- Produces LONG / SHORT / WAIT plus ATR-based stop and three targets.
- PAPER mode only. Order placement is deliberately disabled.

## Groww requirement
The current Groww API requires an active Trading API subscription and an access token/API credentials. Put the token into the GROWW_ACCESS_TOKEN environment variable. Never put the token in web/index.html.

## Run locally
Python 3.9+:
    pip install -r requirements.txt
    set GROWW_ACCESS_TOKEN=YOUR_TOKEN   # Windows
    # or: export GROWW_ACCESS_TOKEN=YOUR_TOKEN
    uvicorn app:app --reload

Open:
    http://127.0.0.1:8000

## Render deployment
1. Push this folder to GitHub.
2. In Render, create a Web Service from the repository.
3. Build command: pip install -r requirements.txt
4. Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
5. Select Free.
6. Add environment variable GROWW_ACCESS_TOKEN with your token.
7. Open the generated onrender.com URL in Chrome on your phone.

## Important
The free Render service can sleep after inactivity. It may take time to wake.

## Gold
This build intentionally keeps Gold behind a separate provider adapter. Gold/MCX will be added after choosing a currently supported market-data source.

## Next engineering steps
1. Add a dedicated Gold data adapter.
2. Add backtesting endpoint and trade journal.
3. Add authentication for the dashboard.
4. Add optional alerts.
5. Only after extensive paper testing, consider a separately gated order-execution module.

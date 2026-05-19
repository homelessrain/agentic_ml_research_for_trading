---
name: fetch-bars
description: Fetch OHLCV bar data for one or more symbols using AlpacaBarFetcher or YahooBarFetcher and cache the result to disk. Use when any skill or user request needs raw bar data as input.
---

# fetch-bars

Fetch OHLCV bar data for one or more symbols. Both fetchers handle disk caching transparently — repeated calls for the same parameters are served from `data/bars/` without hitting the API.

## Inputs

Parse these from the user's request (all required unless noted):

| Parameter | Description | Default |
|---|---|---|
| `symbol` | Ticker string or comma-separated list (e.g. `AAPL`, `BTC/USD`, `AAPL,MSFT`) | — |
| `start` | Start date `YYYY-MM-DD` | — |
| `end` | End date `YYYY-MM-DD` (inclusive) | — |
| `timeframe` | `minute` / `hour` / `day` / `week` / `month` | `day` |
| `only_market_hours` | Filter to exchange hours (equities: 9:30–16:00 ET; crypto: always full day) | `True` |
| `source` | `alpaca` or `yahoo` | `alpaca` |

If any required parameter is missing, ask the user before proceeding.

## Choosing a source

| | `AlpacaBarFetcher` | `YahooBarFetcher` |
|---|---|---|
| **Intraday (minute/hour)** | Yes, including crypto | Last 30 days only |
| **Crypto** | Yes (`BTC/USD` etc.) | No |
| **Price adjustment** | Unadjusted | Adjusted (auto_adjust=True) |
| **Extra columns** | `trade_count`, `vwap` | `trade_count`/`vwap` are NaN |
| **Auth required** | Yes (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY` in `.env`) | No |

Default to `alpaca`. Prefer `yahoo` only when the user explicitly asks for it or needs adjusted prices.

## Steps

1. **Fetch bars** using the appropriate fetcher:

   ```python
   import sys
   sys.path.insert(0, '.')
   from utils.bar_fetcher import AlpacaBarFetcher, YahooBarFetcher

   # source = 'alpaca' (default) or 'yahoo'
   fetcher = AlpacaBarFetcher() if source == 'alpaca' else YahooBarFetcher()
   df = fetcher.fetch_bars(
       symbol=symbol,          # str or list[str]
       start_datestr=start,
       end_datestr=end,
       timeframe=timeframe,
       only_market_hours=only_market_hours,
   )
   ```

2. **Report back** with:
   - Source used (`alpaca` or `yahoo`)
   - Shape of the DataFrame (rows × columns)
   - Date range actually covered (first and last timestamp in the index)
   - Whether data was freshly fetched or served from cache (check if the parquet file existed before the call)

## Notes

- `AlpacaBarFetcher` auto-routes stock vs crypto to the correct Alpaca client and applies exchange-calendar market hours.
- `YahooBarFetcher` returns adjusted prices and naturally limits intraday data to market hours; `only_market_hours` is accepted but has no additional effect.
- Cache files are named `<source>_<start>_<end>_<timeframe>_<mh|full>.parquet` under `data/bars/<symbol>/`.
- When called by another skill, load cached data directly:
  ```python
  cache_path = fetcher._cache_path(symbol, start, end, timeframe, only_market_hours)
  df = pd.read_parquet(cache_path)
  ```

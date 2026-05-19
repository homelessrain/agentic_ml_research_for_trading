---
name: fetch-bars
description: Fetch OHLCV bar data for one or more symbols using AlpacaBarFetcher and cache the result to disk. Use when any skill or user request needs raw bar data as input.
---

# fetch-bars

Fetch OHLCV bar data for one or more symbols. `AlpacaBarFetcher` handles disk caching transparently — repeated calls for the same parameters are served from `data/bars/` without hitting the API.

## Inputs

Parse these from the user's request (all required unless noted):

| Parameter | Description | Default |
|---|---|---|
| `symbol` | Ticker string or comma-separated list (e.g. `AAPL`, `BTC/USD`, `AAPL,MSFT`) | — |
| `start` | Start date `YYYY-MM-DD` | — |
| `end` | End date `YYYY-MM-DD` (inclusive) | — |
| `timeframe` | `minute` / `hour` / `day` / `week` / `month` | `day` |
| `only_market_hours` | Filter to exchange hours (equities: 9:30–16:00 ET; crypto: always full day) | `True` |

If any required parameter is missing, ask the user before proceeding.

## Steps

1. **Fetch bars** using `AlpacaBarFetcher`:
   ```python
   import sys
   sys.path.insert(0, '.')
   from utils.bar_fetcher import AlpacaBarFetcher

   fetcher = AlpacaBarFetcher()
   df = fetcher.fetch_bars(
       symbol=symbol,          # str or list[str]
       start_datestr=start,
       end_datestr=end,
       timeframe=timeframe,
       only_market_hours=only_market_hours,
   )
   ```

2. **Report back** with:
   - Shape of the DataFrame (rows × columns)
   - Date range actually covered (first and last timestamp in the index)
   - Whether data was freshly fetched or served from cache (check if the parquet file existed before the call)

## Notes

- `AlpacaBarFetcher` automatically selects the right Alpaca client (stock vs crypto) and applies the correct market hours — no manual routing needed.
- When called by another skill, the calling skill can load the cached data directly:
  ```python
  cache_path = fetcher._cache_path(symbol, start, end, timeframe, only_market_hours)
  df = pd.read_parquet(cache_path)
  ```

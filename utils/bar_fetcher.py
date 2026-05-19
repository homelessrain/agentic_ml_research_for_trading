from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import os
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import pandas as pd
import yfinance as yf
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import GetCalendarRequest

load_dotenv()

_ET = ZoneInfo('America/New_York')
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_DIR = os.path.join(_PROJECT_ROOT, 'data', 'bars')


class BaseBarFetcher(ABC):
    """
    This class is responsible for fetching bars from different data sources.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fetch_bars(self) -> pd.DataFrame:
        """
        Fetch bars from the data source.
        """
        pass


class AlpacaBarFetcher:
    """
    This class is responsible for fetching bars from different data sources.
    """

    _SOURCE = 'alpaca'
    _TIMEFRAME_MAP = {
        'minute': TimeFrame.Minute,
        'hour':   TimeFrame.Hour,
        'day':    TimeFrame.Day,
        'week':   TimeFrame.Week,
        'month':  TimeFrame.Month,
    }

    def __init__(self):
        super().__init__()
        api_key = os.environ.get('ALPACA_API_KEY')
        secret_key = os.environ.get('ALPACA_SECRET_KEY')

        assert api_key is not None and secret_key is not None, "API key and secret key are required"

        self._stock_client  = StockHistoricalDataClient(api_key, secret_key)
        self._crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        self._trading_client = TradingClient(api_key, secret_key)
        self._asset_class_cache: dict[str, AssetClass] = {}

    def _get_asset_class(self, symbol: str) -> AssetClass:
        if symbol not in self._asset_class_cache:
            self._asset_class_cache[symbol] = self._trading_client.get_asset(symbol).asset_class
        return self._asset_class_cache[symbol]

    def get_market_hours(self, symbol: str, start_datestr: str, end_datestr: str) -> pd.DataFrame:
        """
        Return a DataFrame of trading days with UTC-aware open and close timestamps.

        For equities, hours come from the Alpaca exchange calendar (handles early closures).
        For crypto, every calendar day is open 00:00–23:59:59 UTC.

        Columns: date, open (UTC), close (UTC)
        """
        start = datetime.strptime(start_datestr, '%Y-%m-%d').date()
        end   = datetime.strptime(end_datestr,   '%Y-%m-%d').date()

        if self._get_asset_class(symbol) == AssetClass.CRYPTO:
            days = pd.date_range(start=start_datestr, end=end_datestr, freq='D')
            rows = [
                {
                    'date':  d.date(),
                    'open':  datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=timezone.utc),
                    'close': datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc),
                }
                for d in days
            ]
        else:
            calendar = self._trading_client.get_calendar(
                GetCalendarRequest(start=start, end=end)
            )
            rows = [
                {
                    'date':  entry.date,
                    'open':  entry.open.replace(tzinfo=_ET).astimezone(timezone.utc),
                    'close': entry.close.replace(tzinfo=_ET).astimezone(timezone.utc),
                }
                for entry in calendar
            ]

        return pd.DataFrame(rows, columns=['date', 'open', 'close'])


    def _cache_path(self, symbol, start_datestr: str, end_datestr: str, timeframe: str, only_market_hours: bool) -> str:
        sym_key = symbol.replace('/', '-') if isinstance(symbol, str) else '_'.join(s.replace('/', '-') for s in sorted(symbol))
        mh = 'mh' if only_market_hours else 'full'
        filename = f"{self._SOURCE}_{start_datestr}_{end_datestr}_{timeframe}_{mh}.parquet"
        return os.path.join(_CACHE_DIR, sym_key, filename)

    def fetch_bars(self, symbol, start_datestr, end_datestr, timeframe='minute', only_market_hours=True) -> pd.DataFrame:
        """
        Fetch bars from Alpaca, with transparent disk caching under data/bars/.

        Cache path: data/bars/<symbol>/<start>_<end>_<timeframe>_<mh|full>.parquet

        Args:
            symbol: ticker string or list of ticker strings
            start_datestr: start date as 'YYYY-MM-DD'
            end_datestr: end date as 'YYYY-MM-DD' (inclusive)
            timeframe: 'minute' | 'hour' | 'day' | 'week' | 'month'
            only_market_hours: filter to exchange hours (equities) or full day (crypto)

        Returns:
            DataFrame indexed by (symbol, timestamp) with columns
            open, high, low, close, volume, trade_count, vwap
        """
        cache = self._cache_path(symbol, start_datestr, end_datestr, timeframe, only_market_hours)
        if os.path.exists(cache):
            return pd.read_parquet(cache)

        tf = self._TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from: {list(self._TIMEFRAME_MAP)}")

        start = datetime.strptime(start_datestr, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end = datetime.strptime(end_datestr, '%Y-%m-%d').replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)

        if self._get_asset_class(symbol) == AssetClass.CRYPTO:
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
            bar_set = self._crypto_client.get_crypto_bars(request)
        else:
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
            bar_set = self._stock_client.get_stock_bars(request)

        df = bar_set.df

        if only_market_hours and timeframe in ('minute', 'hour'):
            market_hours = self.get_market_hours(symbol, start_datestr, end_datestr)
            timestamps = df.index.get_level_values('timestamp')
            mask = pd.Series(False, index=df.index)
            for _, row in market_hours.iterrows():
                mask |= (timestamps >= row['open']) & (timestamps <= row['close'])
            df = df[mask]

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache)
        return df


class YahooBarFetcher(BaseBarFetcher):
    """
    This class is responsible for fetching bars from Yahoo Finance.
    """

    _SOURCE = 'yahoo'
    _INTERVAL_MAP = {
        'minute': '1m',
        'hour':   '1h',
        'day':    '1d',
        'week':   '1wk',
        'month':  '1mo',
    }

    def __init__(self):
        super().__init__()

    def _cache_path(self, symbol, start_datestr: str, end_datestr: str, timeframe: str, only_market_hours: bool) -> str:
        sym_key = symbol.replace('/', '-') if isinstance(symbol, str) else '_'.join(s.replace('/', '-') for s in sorted(symbol))
        mh = 'mh' if only_market_hours else 'full'
        filename = f"{self._SOURCE}_{start_datestr}_{end_datestr}_{timeframe}_{mh}.parquet"
        return os.path.join(_CACHE_DIR, sym_key, filename)

    def fetch_bars(self, symbol, start_datestr, end_datestr, timeframe='day', only_market_hours=True) -> pd.DataFrame:
        """
        Fetch bars from Yahoo Finance, with transparent disk caching under data/bars/.

        Cache path: data/bars/<symbol>/yahoo_<start>_<end>_<timeframe>_<mh|full>.parquet

        Args:
            symbol: ticker string or list of ticker strings
            start_datestr: start date as 'YYYY-MM-DD'
            end_datestr: end date as 'YYYY-MM-DD' (inclusive)
            timeframe: 'minute' | 'hour' | 'day' | 'week' | 'month'
            only_market_hours: accepted for API compatibility; yfinance naturally
                               returns market-hours-only data for intraday bars

        Returns:
            DataFrame indexed by (symbol, timestamp) with columns
            open, high, low, close, volume, trade_count, vwap
        """
        cache = self._cache_path(symbol, start_datestr, end_datestr, timeframe, only_market_hours)
        if os.path.exists(cache):
            return pd.read_parquet(cache)

        interval = self._INTERVAL_MAP.get(timeframe)
        if interval is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Choose from: {list(self._INTERVAL_MAP)}")

        # yfinance end is exclusive — add one day to include end_datestr
        end_exclusive = (datetime.strptime(end_datestr, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

        symbols = [symbol] if isinstance(symbol, str) else list(symbol)

        if len(symbols) == 1:
            raw = yf.Ticker(symbols[0]).history(
                start=start_datestr, end=end_exclusive, interval=interval, auto_adjust=True
            )
            if raw.empty:
                df = pd.DataFrame(
                    columns=['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap'],
                    index=pd.MultiIndex.from_tuples([], names=['symbol', 'timestamp']),
                )
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                df.to_parquet(cache)
                return df
            raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']]
            raw.columns = ['open', 'high', 'low', 'close', 'volume']
            raw.index = raw.index.tz_convert('UTC').rename('timestamp')
            raw.insert(0, 'symbol', symbols[0])
            df = raw.reset_index().set_index(['symbol', 'timestamp'])
        else:
            raw = yf.download(symbols, start=start_datestr, end=end_exclusive,
                              interval=interval, auto_adjust=True, progress=False)
            df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].stack(level='Ticker').rename_axis(
                index=['timestamp', 'symbol']
            )
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.reset_index()
            df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')
            df = df.set_index(['symbol', 'timestamp']).sort_index()

        df['trade_count'] = float('nan')
        df['vwap'] = float('nan')

        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_parquet(cache)
        return df

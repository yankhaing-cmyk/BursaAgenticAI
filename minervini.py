"""
Minervini-style Stage 2 screener (SEPA Trend Template).
Runs weekly (Saturday morning) on the same tickers.csv as the main screener
and sends a watchlist to Telegram.

Trend Template (ALL price conditions must pass):
 1. Price above 150-day and 200-day SMA
 2. 150-day SMA above 200-day SMA
 3. 200-day SMA rising for at least ~1 month
 4. 50-day SMA above 150-day and 200-day SMA
 5. Price above 50-day SMA
 6. Price at least 30% above its 52-week low
 7. Price within 25% of its 52-week high
 8. Relative Strength percentile >= 70 vs the whole screened universe
    (IBD-style weighting: 40% x 3-month + 20% x 6/9/12-month returns)

VCP-lite bonus flags (approximations of chart-read patterns, not requirements):
  C = volatility contraction (10d ATR well below 40d ATR)
  D = volume dry-up (10d avg volume below 50d avg)
  P = near pivot (within 5% of 20-day closing high)
"""

import os
import sys
import time
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# --------- Market config (Bursa values; for HKEX repo change these three) ----
TICKER_SUFFIX = ".KL"      # HKEX: ".HK"
CURRENCY = "RM"            # HKEX: "HK$"
MIN_AVG_VALUE = 1_000_000  # HKEX: 20_000_000
# ------------------------------------------------------------------------------

TICKER_FILE = "tickers.csv"
LOOKBACK_PERIOD = "420d"   # ~290 trading days (enough for 200SMA trend + 52w)
MIN_BARS = 240
MIN_RS_PCTL = 70           # Minervini: RS rating of 70+, ideally 80-90+
TOP_N = 20
BATCH_SIZE = 100

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TG_CHAR_LIMIT = 3800


def sma(s, w):
    return s.rolling(w).mean()


def rs_raw(close: pd.Series):
    """IBD-style weighted return: 40% x 3m + 20% x (6m + 9m + 12m)."""
    def ret(n):
        if len(close) <= n:
            return None
        past = float(close.iloc[-n - 1])
        return (float(close.iloc[-1]) / past - 1) if past > 0 else None
    r3, r6, r9, r12 = ret(63), ret(126), ret(189), ret(252)
    if None in (r3, r6, r9, r12):
        # fall back to available horizons for slightly younger listings
        if r3 is None or r6 is None:
            return None
        r9 = r9 if r9 is not None else r6
        r12 = r12 if r12 is not None else r9
    return 0.4 * r3 + 0.2 * r6 + 0.2 * r9 + 0.2 * r12


def analyze(ticker: str, name: str, df: pd.DataFrame, today: dt.date):
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < MIN_BARS:
        return None
    if (today - df.index[-1].date()).days > 4:      # stale/suspended
        return None

    close, volume = df["Close"], df["Volume"]
    high, low = df["High"], df["Low"]
    price = float(close.iloc[-1])

    avg_value = float((close * volume).rolling(20).mean().iloc[-1])
    if avg_value < MIN_AVG_VALUE:
        return None

    s50 = sma(close, 50)
    s150 = sma(close, 150)
    s200 = sma(close, 200)
    lo52 = float(close.tail(252).min())
    hi52 = float(close.tail(252).max())

    v50, v150, v200 = float(s50.iloc[-1]), float(s150.iloc[-1]), float(s200.iloc[-1])
    template = (
        price > v150 and price > v200 and          # 1
        v150 > v200 and                            # 2
        v200 > float(s200.iloc[-22]) and           # 3 (rising ~1 month)
        v50 > v150 and v50 > v200 and              # 4
        price > v50 and                            # 5
        price >= 1.30 * lo52 and                   # 6
        price >= 0.75 * hi52                       # 7
    )
    rs = rs_raw(close)
    if not template or rs is None:
        return None

    # VCP-lite flags
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr10 = float(tr.rolling(10).mean().iloc[-1])
    atr40 = float(tr.rolling(40).mean().iloc[-1])
    flags = ""
    if atr40 > 0 and atr10 / atr40 <= 0.70:
        flags += "C"
    if float(volume.tail(10).mean()) <= 0.8 * float(volume.tail(50).mean()):
        flags += "D"
    if price >= 0.95 * float(close.tail(20).max()):
        flags += "P"

    return {
        "ticker": ticker.replace(TICKER_SUFFIX, ""), "name": name,
        "price": price, "rs": rs, "off_high": (price / hi52 - 1) * 100,
        "off_low": (price / lo52 - 1) * 100, "flags": flags,
        "avg_value": avg_value,
    }


def run():
    tickers = pd.read_csv(TICKER_FILE)
    symbols = tickers["ticker"].tolist()
    names = dict(zip(tickers["ticker"], tickers["name"]))
    today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).date()

    print(f"Minervini screen over {len(symbols)} tickers...")
    passed, all_rs = [], []

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        print(f"  batch {start // BATCH_SIZE + 1}")
        try:
            data = yf.download(batch, period=LOOKBACK_PERIOD, interval="1d",
                               group_by="ticker", auto_adjust=True,
                               threads=True, progress=False)
        except Exception as e:
            print(f"  batch failed: {e}")
            continue
        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                close = df["Close"].dropna()
                if len(close) >= 130:
                    r = rs_raw(close)
                    if r is not None:
                        all_rs.append(r)      # universe for percentile ranking
                res = analyze(sym, names.get(sym, sym), df, today)
                if res:
                    passed.append(res)
            except Exception:
                pass

    # RS percentile vs the whole screened universe
    all_rs = np.array(sorted(all_rs))
    results = []
    for r in passed:
        pctl = float((all_rs < r["rs"]).mean() * 100) if len(all_rs) else 0
        if pctl >= MIN_RS_PCTL:
            r["rs_pctl"] = pctl
            results.append(r)

    results.sort(key=lambda x: -x["rs_pctl"])
    print(f"{len(passed)} passed template; {len(results)} also passed RS>= {MIN_RS_PCTL}.")
    return results[:TOP_N], len(all_rs)


def build_message(results, universe_n):
    today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).strftime("%d %b %Y")
    lines = [f"\U0001F3C6 <b>Minervini Stage 2 Watchlist</b>",
             f"\U0001F5D3 {today} | RS ranked vs {universe_n} stocks\n"]
    if not results:
        lines.append("No stocks pass the full Trend Template right now. "
                     "In Minervini's playbook, a thin or empty list is a "
                     "signal to stay patient - the market has few Stage 2 "
                     "leaders at the moment.")
    else:
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. <b>{r['name']}</b> ({r['ticker']})\n"
                f"   {CURRENCY}{r['price']:.3f} | RS {r['rs_pctl']:.0f} | "
                f"{r['off_high']:+.0f}% vs 52wH | +{r['off_low']:.0f}% off low"
                + (f" | [{r['flags']}]" if r['flags'] else "")
            )
        lines.append("\n<i>All passed: price > 50/150/200SMA stack, 200SMA "
                     "rising, >=30% off 52w low, within 25% of 52w high, "
                     "RS pctl >= 70.</i>")
        lines.append("<i>VCP-lite flags: C=volatility contraction "
                     "D=volume dry-up P=near pivot. These approximate chart "
                     "patterns - always inspect the chart before acting.</i>")
    lines.append("<i>Not financial advice.</i>")
    return lines


def send_telegram(lines):
    chunks, cur = [], ""
    for line in lines:
        cand = (cur + "\n" + line) if cur else line
        if len(cand) > TG_CHAR_LIMIT:
            chunks.append(cur); cur = line
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; printing instead:\n")
        [print(c) for c in chunks]
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for i, chunk in enumerate(chunks, 1):
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                                 "parse_mode": "HTML"}, timeout=30
                      ).raise_for_status()
        print(f"Telegram message {i}/{len(chunks)} sent.")
        time.sleep(1)


if __name__ == "__main__":
    try:
        results, n = run()
        send_telegram(build_message(results, n))
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)

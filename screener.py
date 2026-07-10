"""
Bursa Malaysia Screener v3
Three screens on the day's close, sent to Telegram at 6pm MYT:
  1) Established momentum  2) Early uptrend  3) Potential reversals

v3 improvements over v2:
- Stale-data guard: skips suspended/non-trading counters
- Trading consistency: must have traded 18 of last 20 days
- Momentum: sustained volume (5d vs 20d),
  52-week-high proximity criterion, RSI cap relaxed to 80
- Early: MACD trigger requires rising histogram (anti-whipsaw),
  "not extended" is now mandatory, tight-base support added
- Reversal: up-day or strong-close is now MANDATORY (no falling knives),
  oversold tightened to RSI<35, limit-move guard (skip +/-25% days)
- Market regime header: KLCI vs its 50-day average
- Ranking tie-break by traded value (quality) instead of raw ROC (spikiness)
"""

import os
import re
import sys
import time
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# ----------------------------- Config ---------------------------------------

TICKER_FILE = "tickers.csv"
LOOKBACK_DAYS = 300
TOP_N = 50               # established momentum
EARLY_TOP_N = 25         # early uptrend
REVERSAL_TOP_N = 25      # potential reversals
ACC_TOP_N = 20           # accumulation watch (GCB-type base awakening)

MIN_PRICE = 0.20         # RM
MIN_AVG_VALUE = 1_000_000  # RM, 20-day average daily traded value
MAX_STALE_DAYS = 4       # skip if last bar older than this (suspended counters)
MIN_TRADED_DAYS = 18     # of the last 20 sessions

HOLIDAYS_2026 = {
    "2026-01-01", "2026-02-17", "2026-02-18", "2026-03-21",
    "2026-03-27", "2026-05-01", "2026-05-27", "2026-05-31",
    "2026-06-01", "2026-06-16", "2026-08-25", "2026-08-31",
    "2026-09-16", "2026-11-09", "2026-12-25",
}

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TG_CHAR_LIMIT = 3800
HISTORY_FILE = "history.csv"
TICKER_SUFFIX = ".KL"

# ----------------------------- Indicators -----------------------------------

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def sma(s, window):
    return s.rolling(window).mean()

def rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s):
    line = ema(s, 12) - ema(s, 26)
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig

# ----------------------------- Shared prep -----------------------------------

def prep(df: pd.DataFrame, today: dt.date):
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 70:
        return None

    # Stale-data guard: suspended counters keep old prices in Yahoo
    last_bar = df.index[-1].date()
    if (today - last_bar).days > MAX_STALE_DAYS:
        return None

    close, volume = df["Close"], df["Volume"]
    high, low = df["High"], df["Low"]
    price = float(close.iloc[-1])

    # Liquidity + consistency filters
    avg_value_20 = float((close * volume).rolling(20).mean().iloc[-1])
    if price < MIN_PRICE or avg_value_20 < MIN_AVG_VALUE:
        return None
    if int((volume.tail(20) > 0).sum()) < MIN_TRADED_DAYS:
        return None

    ind = {
        "close": close, "volume": volume, "high": high, "low": low,
        "price": price, "avg_value": avg_value_20,
        "ema20": ema(close, 20),
        "sma50": sma(close, 50),
        "rsi14": rsi(close, 14),
        "vol_ma20": sma(volume, 20),
        "roc20": (close / close.shift(20) - 1) * 100,
        "hi52": float(close.tail(252).max()),
    }
    ind["macd_line"], ind["macd_sig"] = macd(close)
    ind["hist"] = ind["macd_line"] - ind["macd_sig"]
    v20 = float(ind["vol_ma20"].iloc[-1])
    ind["v_day"] = float(volume.iloc[-1]) / v20 if v20 > 0 else 0.0
    ind["v_sust"] = float(volume.tail(5).mean()) / v20 if v20 > 0 else 0.0
    return ind

# --------------------- Screen 1: established momentum ------------------------

def analyze_momentum(ticker, name, ind, P, klci_roc=0.0):
    price = ind["price"]
    # Mandatory: full trend stack incl. long-term filter
    if not (price > float(ind["ema20"].iloc[-1]) > float(ind["sma50"].iloc[-1])):
        return None

    r = float(ind["rsi14"].iloc[-1])
    roc = float(ind["roc20"].iloc[-1])

    # Playbook-level mandatory extras
    if P.get("mom_need_pos_roc") and roc <= 0:
        return None
    if P.get("mom_near_high") and price < P["mom_near_high"] * ind["hi52"]:
        return None
    if P.get("mom_rel_strength") is not None \
            and (roc - klci_roc) < P["mom_rel_strength"]:
        return None

    criteria = {
        "rsi":     P["rsi_lo"] <= r <= P["rsi_hi"],
        "macd":    float(ind["macd_line"].iloc[-1]) > float(ind["macd_sig"].iloc[-1]),
        "volume":  ind["v_sust"] >= P["vol_sust"],          # sustained 5d interest
        "roc":     roc >= 5.0,
        "hi52":    price >= 0.85 * ind["hi52"],   # within 15% of 52w high
    }
    score = sum(criteria.values())
    min_score = 5 if P.get("mom_all_criteria") else P["mom_min_score"]
    if score < min_score:
        return None

    return {
        "ticker": ticker.replace(".KL", ""), "name": name,
        "price": price, "rsi": r, "vol_ratio": ind["v_sust"],
        "roc20": roc, "score": score, "avg_value": ind["avg_value"],
        "flags": "T" + "".join(k[0].upper() for k, v in criteria.items() if v),
    }

# --------------------- Screen 2: early uptrend -------------------------------

def analyze_early(ticker, name, ind, P):
    if not P.get("early_enabled", True):
        return None
    close, price = ind["close"], ind["price"]
    ema20 = ind["ema20"]
    hist = ind["hist"]

    # Mandatory: not extended (early means not already stretched)
    if price > float(ema20.iloc[-1]) * 1.06:
        return None

    # Trigger A: fresh MACD cross WITH rising histogram (anti-whipsaw)
    macd_now = float(ind["macd_line"].iloc[-1]) > float(ind["macd_sig"].iloc[-1])
    macd_then = float(ind["macd_line"].iloc[-4]) <= float(ind["macd_sig"].iloc[-4])
    hist_rising = (float(hist.iloc[-1]) > float(hist.iloc[-2])
                   > float(hist.iloc[-3]))
    fresh_macd = macd_now and macd_then and hist_rising

    # Trigger B: price freshly reclaimed EMA20
    above_now = price > float(ema20.iloc[-1])
    was_below = bool((close.iloc[-6:-1] <= ema20.iloc[-6:-1]).any())
    fresh_price = above_now and was_below

    if not (fresh_macd or fresh_price):
        return None

    r_now, r_then = float(ind["rsi14"].iloc[-1]), float(ind["rsi14"].iloc[-6])
    rng20 = close.tail(20)
    tight_base = (float(rng20.max()) / float(rng20.min()) - 1) <= 0.15

    supports = {
        "rsi_recover": 45 <= r_now <= 62 and r_now > r_then,
        "vol_pickup":  ind["v_day"] >= 1.2,
        "base_ok":     float(ind["sma50"].iloc[-1])
                       >= float(ind["sma50"].iloc[-11]) * 0.99,
        "tight_base":  tight_base,
    }
    if P.get("early_tight_mandatory") and not tight_base:
        return None
    n = sum(supports.values())
    if n < P.get("early_supports", 2):
        return None

    trigger = "M+P" if (fresh_macd and fresh_price) else ("M" if fresh_macd else "P")
    return {
        "ticker": ticker.replace(".KL", ""), "name": name,
        "price": price, "rsi": r_now, "vol_ratio": ind["v_day"],
        "roc20": float(ind["roc20"].iloc[-1]),
        "score": n + (2 if trigger == "M+P" else 1),
        "avg_value": ind["avg_value"],
        "flags": trigger + "".join(
            {"rsi_recover": "R", "vol_pickup": "V",
             "base_ok": "B", "tight_base": "T"}[k]
            for k, v in supports.items() if v),
    }

# --------------------- Screen 3: potential reversal --------------------------

def analyze_reversal(ticker, name, ind, P):
    price = ind["price"]
    close, high, low = ind["close"], ind["high"], ind["low"]
    roc = float(ind["roc20"].iloc[-1])

    # Mandatory: genuine downtrend
    if not (price < float(ind["ema20"].iloc[-1]) < float(ind["sma50"].iloc[-1])
            and roc < 0):
        return None

    # Mandatory: strong volume spike (follow-through mode scans older bars)
    if P.get("rev_followthrough"):
        vol, v20 = ind["volume"], float(ind["vol_ma20"].iloc[-1])
        spike_idx = None
        for k in (-3, -2):                       # spike must be 2+ bars old
            if v20 > 0 and float(vol.iloc[k]) / v20 >= P["rev_spike"]:
                spike_idx = k
                break
        if spike_idx is None:
            return None
        spike_low = float(ind["low"].iloc[spike_idx])
        if not all(float(close.iloc[j]) > spike_low
                   for j in range(spike_idx + 1, 0)):
            return None                          # failed to hold spike low
        eval_idx = spike_idx                     # judge direction ON spike day
    elif ind["v_day"] < P["rev_spike"]:
        return None
    else:
        eval_idx = -1

    day_change = float(close.iloc[eval_idx] / close.iloc[eval_idx - 1] - 1) * 100
    # Limit-move guard: skip extreme days (limit-down mechanics, corp events)
    if abs(day_change) > 25:
        return None

    day_high, day_low = float(high.iloc[eval_idx]), float(low.iloc[eval_idx])
    day_close = float(close.iloc[eval_idx])
    day_range = day_high - day_low
    close_pos = (day_close - day_low) / day_range if day_range > 0 else 0.5

    # Mandatory direction: the spike must show BUYING, not just selling
    up_day = day_change > 0
    strong_close = close_pos >= 0.5
    if P.get("rev_both_mandatory"):
        if not (up_day and strong_close):
            return None
    elif not (up_day or strong_close):
        return None

    r_now = float(ind["rsi14"].iloc[-1])
    supports = {
        "up_day": up_day,
        "strong_close": strong_close,
        "oversold": r_now < P.get("rev_rsi", 35),
        "washed_out": roc <= -10.0,
    }
    n = sum(supports.values())
    if n < 2:
        return None

    return {
        "ticker": ticker.replace(".KL", ""), "name": name,
        "price": price, "rsi": r_now, "vol_ratio": ind["v_day"],
        "roc20": roc, "score": n, "avg_value": ind["avg_value"],
        "flags": "".join(
            {"up_day": "U", "strong_close": "C",
             "oversold": "O", "washed_out": "W"}[k]
            for k, v in supports.items() if v),
    }

# ----------------------------- TradingView cross-check -----------------------

def _norm_name(s: str) -> str:
    """Normalize company/short names for matching across data sources."""
    s = str(s).upper().replace("BERHAD", "BHD")
    s = re.sub(r"[^A-Z0-9]", "", s)
    if s.endswith("BHD"):
        s = s[:-3]
    return s

def tv_snapshot():
    """Fetch a one-shot snapshot of all MYX stocks from TradingView.
    Returns a dict keyed by normalized name -> info, or None on any failure."""
    try:
        from tradingview_screener import Query
        _, df = (
            Query()
            .set_markets("malaysia")
            .select("name", "description", "close", "Value.Traded",
                    "RSI", "MACD.macd", "MACD.signal",
                    "EMA20", "SMA50", "price_52_week_high")
            .limit(1500)
            .get_scanner_data()
        )
    except Exception as e:
        print(f"TradingView fetch failed ({e}) - continuing with yfinance only.")
        return None

    tv = {}
    for _, row in df.iterrows():
        try:
            price = float(row["close"])
            ema20 = float(row["EMA20"])
            sma50 = float(row["SMA50"])
            r = float(row["RSI"])
            checks = [
                price > ema20 > sma50,
                55 <= r <= 80,
                float(row["MACD.macd"]) > float(row["MACD.signal"]),
                price >= 0.85 * float(row["price_52_week_high"]),
            ]
            info = {
                "symbol": str(row["name"]),
                "price": price,
                "value": float(row.get("Value.Traded") or 0),
                "momentum": checks[0] and sum(checks) >= 3,
            }
            for key in (_norm_name(row["name"]), _norm_name(row["description"])):
                if key and key not in tv:
                    tv[key] = info
        except Exception:
            continue
    print(f"TradingView snapshot: {len(df)} rows fetched.")
    return tv

def tv_annotate(results: list, tv: dict) -> set:
    """Tag yfinance results with TV confirmation / price-discrepancy flags.
    Returns the set of TV symbols that were matched."""
    matched = set()
    if not tv:
        return matched
    for r in results:
        info = tv.get(_norm_name(r["name"]))
        if not info:
            continue
        matched.add(info["symbol"])
        tags = ""
        if info["momentum"]:
            tags += " \u2714TV"
        if r["price"] > 0 and abs(info["price"] - r["price"]) / r["price"] > 0.03:
            tags += " \u26A0data"
        r["flags"] += tags
    return matched

def tv_extras(tv: dict, matched: set, limit: int = 5) -> list:
    """TV-rated momentum stocks that our yfinance screen did not surface."""
    if not tv:
        return []
    seen, extras = set(), []
    for info in tv.values():
        sym = info["symbol"]
        if sym in seen or sym in matched or not info["momentum"]:
            continue
        seen.add(sym)
        extras.append(info)
    extras.sort(key=lambda x: -x["value"])
    return extras[:limit]

# ----------------------------- Market regime ---------------------------------

def market_regime():
    try:
        klci = yf.download("^KLSE", period="120d", interval="1d",
                           auto_adjust=True, progress=False)
        c = klci["Close"].dropna()
        last = float(c.iloc[-1])
        ma50 = float(c.rolling(50).mean().iloc[-1])
        if last >= ma50:
            return f"🟢 KLCI {last:,.0f} above 50-day avg — risk-on"
        return (f"🔴 KLCI {last:,.0f} below 50-day avg — caution: "
                f"momentum less reliable, expect crowded reversal list")
    except Exception as e:
        print(f"Regime check failed: {e}")
        return "⚪ KLCI regime unavailable"

# ----------------------------- Runner ----------------------------------------

BATCH_SIZE = 100


def snap_row(sym, name, ind):
    """Every ingredient the browser app needs to re-screen with custom
    thresholds. Booleans are NOT baked in - only raw measurements."""
    f = float
    close, low, high = ind["close"], ind["low"], ind["high"]
    rng20 = close.tail(20)
    d = f(close.iloc[-1] / close.iloc[-2] - 1) * 100
    dh, dl = f(high.iloc[-1]), f(low.iloc[-1])
    rngd = dh - dl
    return {
        "t": sym.replace(TICKER_SUFFIX, ""), "n": name,
        "p": round(ind["price"], 4),
        "e20": round(f(ind["ema20"].iloc[-1]), 4),
        "s50": round(f(ind["sma50"].iloc[-1]), 4),
        "s50p": round(f(ind["sma50"].iloc[-11]), 4),
        "rsi": round(f(ind["rsi14"].iloc[-1]), 2),
        "rsip": round(f(ind["rsi14"].iloc[-6]), 2),
        "md": round(f(ind["macd_line"].iloc[-1]) - f(ind["macd_sig"].iloc[-1]), 6),
        "mdp": round(f(ind["macd_line"].iloc[-4]) - f(ind["macd_sig"].iloc[-4]), 6),
        "h1": round(f(ind["hist"].iloc[-1]), 6),
        "h2": round(f(ind["hist"].iloc[-2]), 6),
        "h3": round(f(ind["hist"].iloc[-3]), 6),
        "vd": round(ind["v_day"], 3),
        "vs": round(ind["v_sust"], 3),
        "roc": round(f(ind["roc20"].iloc[-1]), 2),
        "hi52": round(ind["hi52"], 4),
        "av": int(ind["avg_value"]),
        "wb": bool((close.iloc[-6:-1] <= ind["ema20"].iloc[-6:-1]).any()),
        "tb": round(f(rng20.max()) / f(rng20.min()) - 1, 4),
        "dchg": round(d, 2),
        "cpos": round((ind["price"] - dl) / rngd, 3) if rngd > 0 else 0.5,
    }


def analyze_accumulation(ticker, name, ind):
    """Phase-2 'base awakening': volume wakes up while price still sleeps.
    Regime-independent watch list - these are months-early candidates,
    not entries. Modeled on the GCB-type stage transition."""
    close, low, volume = ind["close"], ind["low"], ind["volume"]
    price = ind["price"]
    if len(close) < 150:                     # need base + volume history
        return None

    # 1) A real base: last ~6 months range <= 30%, and price well off
    #    the high of the available window (beaten down / forgotten)
    base = close.tail(126)
    lo_b, hi_b = float(base.min()), float(base.max())
    if lo_b <= 0 or (hi_b / lo_b - 1) > 0.30:
        return None
    hi_all = float(close.max())
    if price > 0.80 * hi_all:                # within 20% of high = not forgotten
        return None

    # 2) Accumulation divergence: 20d volume >= 1.5x 120d volume,
    #    while 60d price change stays quiet (within +/-8%)
    v20 = float(volume.rolling(20).mean().iloc[-1])
    v120 = float(volume.rolling(120).mean().iloc[-1])
    if v120 <= 0:
        return None
    vol_ratio = v20 / v120
    if vol_ratio < 1.5:
        return None
    roc60 = float(close.iloc[-1] / close.iloc[-61] - 1) * 100
    if abs(roc60) > 8.0:
        return None

    # 3) Demand bias: up-day volume dominates down-day volume (21d),
    #    and swing lows are ascending (20d min > prior 20d min)
    chg = close.diff().tail(21)
    v21 = volume.tail(21)
    up_vol = float(v21[chg > 0].sum())
    dn_vol = float(v21[chg < 0].sum())
    updown = (up_vol / dn_vol) if dn_vol > 0 else 2.0
    lows_rising = float(low.tail(20).min()) > float(low.iloc[-40:-20].min())

    supports = {"U": updown >= 1.3, "L": lows_rising}
    n = sum(supports.values())
    if n < 1:                                 # need at least one demand tell
        return None

    flags = "BQ" + "".join(k for k, v in supports.items() if v)
    return {
        "ticker": ticker.replace(".KL", ""), "name": name,
        "price": price, "rsi": float(ind["rsi14"].iloc[-1]),
        "vol_ratio": vol_ratio,               # 20d vs 120d volume
        "roc20": float(ind["roc20"].iloc[-1]),
        "flags": flags,
        "score": round(vol_ratio * (1 + 0.2 * n), 3),
        "avg_value": ind["avg_value"],
    }


def run_screen(today: dt.date, P=None, klci_roc=0.0):
    if P is None:
        import regime as _rg
        P = _rg.PLAYBOOKS["TRENDING"]
    tickers = pd.read_csv(TICKER_FILE)
    symbols = tickers["ticker"].tolist()
    names = dict(zip(tickers["ticker"], tickers["name"]))

    print(f"Screening {len(symbols)} tickers in batches of {BATCH_SIZE}...")
    momentum, early, reversal, accum = [], [], [], []
    analyzed, skipped, above_ema20 = 0, 0, 0
    snapshot = []

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]
        print(f"  batch {start // BATCH_SIZE + 1}: {len(batch)} tickers")
        try:
            data = yf.download(
                batch, period=f"{LOOKBACK_DAYS}d", interval="1d",
                group_by="ticker", auto_adjust=True,
                threads=True, progress=False,
            )
        except Exception as e:
            print(f"  batch failed: {e}")
            continue

        for sym in batch:
            try:
                df = data[sym] if len(batch) > 1 else data
                ind = prep(df, today)
                if ind is None:
                    skipped += 1
                    continue
                analyzed += 1
                if ind["price"] > float(ind["ema20"].iloc[-1]):
                    above_ema20 += 1
                name = names.get(sym, sym)
                snapshot.append(snap_row(sym, name, ind))
                res = analyze_momentum(sym, name, ind, P, klci_roc)
                if res:
                    momentum.append(res)
                    continue
                res = analyze_early(sym, name, ind, P)
                if res:
                    early.append(res)
                    continue
                res = analyze_reversal(sym, name, ind, P)
                if res:
                    reversal.append(res)
                    continue
                res = analyze_accumulation(sym, name, ind)
                if res:
                    accum.append(res)
            except Exception:
                skipped += 1

    for lst in (momentum, early, reversal):
        lst.sort(key=lambda x: (-x["score"], -x["avg_value"]))

    breadth = (above_ema20 / analyzed * 100) if analyzed else None
    stats = {"total": len(symbols), "analyzed": analyzed, "skipped": skipped,
             "breadth": breadth}
    print(f"Quality gates: {analyzed} analyzed, {skipped} skipped.")
    print(f"{len(momentum)} momentum, {len(early)} early, "
          f"{len(reversal)} reversal.")
    stats["snapshot"] = snapshot
    accum.sort(key=lambda x: (-x["score"], -x["avg_value"]))
    cm, ce, cr = P.get("caps", (TOP_N, EARLY_TOP_N, REVERSAL_TOP_N))
    return momentum[:cm], early[:ce], reversal[:cr], accum[:ACC_TOP_N], stats

# ----------------------------- Pick history / streaks ------------------------

def load_history():
    try:
        return pd.read_csv(HISTORY_FILE, dtype=str)
    except Exception:
        return pd.DataFrame(
            columns=["date", "list", "ticker", "name", "price", "score", "flags"])

def streaks_for(hist, list_name, today_str):
    """Consecutive prior run-days each ticker appeared in this list."""
    sub = hist[(hist["list"] == list_name) & (hist["date"] < today_str)]
    if sub.empty:
        return {}
    dates = sorted(sub["date"].unique(), reverse=True)
    by_date = {d: set(sub[sub["date"] == d]["ticker"]) for d in dates}
    out = {}
    for t in by_date[dates[0]]:
        n = 0
        for d in dates:
            if t in by_date[d]:
                n += 1
            else:
                break
        out[t.replace(TICKER_SUFFIX, "")] = n   # match display tickers
    return out

def append_history(hist, picks_by_list, today_str):
    rows = []
    for lname, picks in picks_by_list.items():
        for r in picks:
            rows.append({
                "date": today_str, "list": lname,
                "ticker": r["ticker"] + TICKER_SUFFIX, "name": r["name"],
                "price": f"{r['price']:.4f}", "score": str(r["score"]),
                "flags": r["flags"],
            })
    hist = hist[hist["date"] != today_str]          # dedupe same-day re-runs
    pd.concat([hist, pd.DataFrame(rows)], ignore_index=True).to_csv(
        HISTORY_FILE, index=False)
    print(f"history.csv: +{len(rows)} rows for {today_str}.")

# ----------------------------- Telegram -------------------------------------

def fmt(r, i, streaks=None):
    s = None if streaks is None else streaks.get(r["ticker"])
    badge = " \U0001F195" if s is None else f" \u00d7{s + 1}"
    return (
        f"{i}. <b>{r['name']}</b> ({r['ticker']}){badge}\n"
        f"   RM{r['price']:.3f} | RSI {r['rsi']:.0f} | "
        f"Vol {r['vol_ratio']:.1f}x | 20d {r['roc20']:+.1f}% | "
        f"[{r['flags']}]"
    )

def build_lines(momentum, early, reversal, accum, ref_date, regime, stats,
                tv_extras_list=None, tv_ok=False, streaks=None):
    lines = [
        f"\U0001F4CA <b>Bursa Screen v3</b>\n\U0001F5D3 Close of {ref_date}\n{regime}\n"
        f"\U0001F50E {stats['total']} listed | {stats['analyzed']} passed quality "
        f"gates | {stats['skipped']} skipped (illiquid/stale)\n"
    ]

    lines.append("📈 <b>— Established momentum —</b>")
    lines += [fmt(r, i, (streaks or {}).get("momentum")) for i, r in enumerate(momentum, 1)] or ["None today."]

    lines.append("\n🌱 <b>— Early uptrend candidates —</b>")
    lines += [fmt(r, i, (streaks or {}).get("early")) for i, r in enumerate(early, 1)] or ["None today."]

    lines.append("\n🔄 <b>— Potential reversals (high risk) —</b>")
    lines += [fmt(r, i, (streaks or {}).get("reversal")) for i, r in enumerate(reversal, 1)] or ["None today."]

    lines.append("\n\U0001F9F2 <b>— Accumulation KLSE Stock List —</b>")
    lines.append("<i>Base awakening: volume rising while price still flat. "
                 "Watch stage - months-early, NOT entries.</i>")
    lines += [fmt(r, i, (streaks or {}).get("accumulation")) for i, r in enumerate(accum, 1)] or ["None today."]

    if tv_ok:
        lines.append("\n\U0001F4E1 <b>\u2014 TradingView cross-check \u2014</b>")
        if tv_extras_list:
            lines.append("TV also rates these as momentum (not in lists above):")
            for e in tv_extras_list:
                lines.append(f"\u2022 {e['symbol']} RM{e['price']:.3f}")
        else:
            lines.append("No additional TV-only momentum names.")
    else:
        lines.append("\n\u26AA TradingView cross-check unavailable this run.")

    lines.append(
        "\n<i>Momentum flags: T=trend R=RSI M=MACD V=sustained-vol "
        "R=ROC H=near 52w high</i>"
    )
    lines.append(
        "<i>Early: M=MACD cross P=price cross R=RSI-recover V=volume "
        "B=base-rising T=tight-base | Reversal: U=up-day C=strong-close "
        "O=oversold W=washed-out</i>"
    )
    lines.append("<i>\U0001F195=first appearance \u00d7N=Nth consecutive day</i>")
    lines.append("<i>Not financial advice. Data: Yahoo Finance (EOD).</i>")
    return lines

def chunk_lines(lines):
    chunks, current = [], ""
    for line in lines:
        candidate = (current + "\n" + line) if current else line
        if len(candidate) > TG_CHAR_LIMIT:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks

def send_telegram(lines):
    chunks = chunk_lines(lines)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; printing message instead:\n")
        for c in chunks:
            print(c)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for i, chunk in enumerate(chunks, 1):
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk,
                  "parse_mode": "HTML"},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"Telegram message {i}/{len(chunks)} sent.")
        time.sleep(1)

# ----------------------------- Main ------------------------------------------

def main():
    today_myt = dt.datetime.utcnow() + dt.timedelta(hours=8)
    date_str = today_myt.strftime("%Y-%m-%d")

    if today_myt.weekday() >= 5:
        print(f"{date_str} is a weekend. Exiting.")
        return
    if date_str in HOLIDAYS_2026:
        print(f"{date_str} is a Malaysian public holiday. Exiting.")
        return

    import regime as rg
    playbook, regime_lines, rstate, klci_roc = rg.evaluate(date_str)
    regime = "\n".join(regime_lines)
    momentum, early, reversal, accum, stats = run_screen(today_myt.date(),
                                                         playbook, klci_roc)
    rg.finalize(rstate, stats.get("breadth"),
                [r["ticker"] for r in momentum])
    tv = tv_snapshot()
    matched = tv_annotate(momentum, tv)
    extras = tv_extras(tv, matched)

    hist = load_history()
    streaks = {ln: streaks_for(hist, ln, date_str)
               for ln in ("momentum", "early", "reversal", "accumulation")}
    send_telegram(build_lines(momentum, early, reversal, accum,
                              today_myt.strftime("%d %b %Y"), regime, stats,
                              extras, tv is not None, streaks))
    append_history(hist, {"momentum": momentum, "early": early,
                          "reversal": reversal,
                          "accumulation": accum}, date_str)

    # Hand off to the trade-card agent (runs as the next workflow step)
    import json
    with open("results.json", "w") as f:
        json.dump({"date": date_str, "regime": regime,
                   "playbook": rstate["active"],
                   "rr_floor": playbook["rr_floor"],
                   "momentum": momentum, "early": early,
                   "reversal": reversal, "accumulation": accum}, f)
    print("results.json written for trade-card agent.")

    with open("snapshot.json", "w") as f:
        json.dump({"date": date_str, "playbook": rstate["active"],
                   "klci_roc20": round(klci_roc, 2),
                   "breadth": stats.get("breadth"),
                   "stocks": stats.get("snapshot", [])}, f,
                  separators=(",", ":"))
    print(f"snapshot.json written ({len(stats.get('snapshot', []))} stocks).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)

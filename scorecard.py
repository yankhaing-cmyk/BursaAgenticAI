"""
Weekly scorecard: measures how the screener's past picks actually performed.
Reads history.csv, computes 5-day and 20-day forward returns per list
(and TV-confirmed vs not, for momentum), compares against the index,
and sends a summary to Telegram every Sunday.
"""

import os
import sys
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import requests

HISTORY_FILE = "history.csv"
INDEX_SYMBOL = "^KLSE"      # change to "^HSI" in the HKEX repo
MIN_SIGNALS = 3             # need at least this many evaluated picks to report

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def forward_return(closes: pd.Series, entry_date: str, horizon: int):
    """Return % change from entry close to `horizon` trading days later."""
    closes = closes.dropna()
    pos = closes.index.searchsorted(pd.Timestamp(entry_date))
    if pos >= len(closes) or pos + horizon >= len(closes):
        return None
    entry, exit_ = float(closes.iloc[pos]), float(closes.iloc[pos + horizon])
    if entry <= 0:
        return None
    return (exit_ / entry - 1) * 100


def evaluate(hist: pd.DataFrame):
    tickers = sorted(hist["ticker"].unique())
    start = (pd.Timestamp(hist["date"].min()) - pd.Timedelta(days=7)).date()
    print(f"Downloading {len(tickers)} tickers + index since {start}...")

    data = yf.download(tickers, start=str(start), interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)
    idx = yf.download(INDEX_SYMBOL, start=str(start), interval="1d",
                      auto_adjust=True, progress=False)["Close"].dropna()
    if isinstance(idx, pd.DataFrame):
        idx = idx.iloc[:, 0]

    rows = []
    for _, r in hist.iterrows():
        try:
            closes = (data[r["ticker"]]["Close"]
                      if len(tickers) > 1 else data["Close"])
        except Exception:
            continue
        r5 = forward_return(closes, r["date"], 5)
        r20 = forward_return(closes, r["date"], 20)
        i20 = forward_return(idx, r["date"], 20)
        rows.append({
            "list": r["list"], "flags": str(r["flags"]),
            "r5": r5, "r20": r20,
            "x20": (r20 - i20) if (r20 is not None and i20 is not None) else None,
        })
    return pd.DataFrame(rows)


def block(df: pd.DataFrame, label: str) -> str:
    d5 = df.dropna(subset=["r5"])
    d20 = df.dropna(subset=["r20"])
    if len(d5) < MIN_SIGNALS and len(d20) < MIN_SIGNALS:
        return f"<b>{label}</b>: not enough matured picks yet."
    parts = [f"<b>{label}</b> ({len(d5)} @5d, {len(d20)} @20d)"]
    if len(d5) >= MIN_SIGNALS:
        parts.append(f"  5d: avg {d5['r5'].mean():+.1f}% | "
                     f"win {(d5['r5'] > 0).mean()*100:.0f}%")
    if len(d20) >= MIN_SIGNALS:
        x = d20.dropna(subset=["x20"])
        excess = f" | vs index {x['x20'].mean():+.1f}%" if len(x) else ""
        parts.append(f"  20d: avg {d20['r20'].mean():+.1f}% | "
                     f"median {d20['r20'].median():+.1f}% | "
                     f"win {(d20['r20'] > 0).mean()*100:.0f}%{excess}")
    return "\n".join(parts)


def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets not set; printing instead:\n\n" + text)
        return
    resp = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
        timeout=30)
    resp.raise_for_status()
    print("Scorecard sent.")


def main():
    try:
        hist = pd.read_csv(HISTORY_FILE, dtype=str)
    except Exception:
        print("No history.csv yet - nothing to score.")
        return
    if hist.empty:
        print("history.csv is empty - nothing to score.")
        return

    res = evaluate(hist)
    if res.empty:
        print("No picks could be evaluated yet.")
        return

    today = (dt.datetime.utcnow() + dt.timedelta(hours=8)).strftime("%d %b %Y")
    lines = [f"\U0001F3AF <b>Weekly Scorecard</b> \u2014 {today}",
             f"All picks since {hist['date'].min()}\n"]

    for lname, label in [("momentum", "\U0001F4C8 Momentum"),
                         ("early", "\U0001F331 Early uptrend"),
                         ("reversal", "\U0001F504 Reversals")]:
        sub = res[res["list"] == lname]
        if len(sub):
            lines.append(block(sub, label))

    mom = res[res["list"] == "momentum"]
    tv_yes = mom[mom["flags"].str.contains("TV", na=False)]
    tv_no = mom[~mom["flags"].str.contains("TV", na=False)]
    if len(tv_yes.dropna(subset=["r20"])) >= MIN_SIGNALS \
            and len(tv_no.dropna(subset=["r20"])) >= MIN_SIGNALS:
        lines.append(
            f"\n<b>\u2714TV effect (momentum, 20d avg)</b>: "
            f"confirmed {tv_yes['r20'].mean():+.1f}% vs "
            f"unconfirmed {tv_no['r20'].mean():+.1f}%")

    lines.append("\n<i>Forward returns from pick-day close. "
                 "Not financial advice.</i>")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)

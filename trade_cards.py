"""
Trade-card agent.
Runs after the daily screener. For the top picks it:
  1. DETERMINISTIC: computes a menu of structural stop/target candidates
     (swing low, EMA20 buffer, ATR stop; 52w high, measured move, ATR targets)
  2. AGENTIC: asks Claude to select the levels that fit each stock's actual
     structure and flag qualitative risks
  3. DETERMINISTIC: computes R:R from the chosen levels, applies the floor,
     and computes the viability price for rejected setups
Sends a second Telegram message with the trade cards.

Analysis frames, not recommendations. The agent never says buy/sell and
never sizes positions.
"""

import os
import sys
import json
import time
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import requests

# ----------------------------- Config ---------------------------------------
TICKER_SUFFIX = ".KL"
CURRENCY = "RM"
RR_FLOOR = 2.0             # minimum reward:risk (Layer 3 playbooks can vary this)
MAX_CARDS = 12             # agent analyzes top N candidates (cost control)
MODEL = "claude-haiku-4-5"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TG_CHAR_LIMIT = 3800

# ----------------------------- Level engine (deterministic) ------------------

def compute_levels(df: pd.DataFrame):
    """Return the structural level menu for one stock, or None."""
    df = df.dropna(subset=["Close"])
    if len(df) < 60:
        return None
    close, high, low = df["Close"], df["High"], df["Low"]
    price = float(close.iloc[-1])

    tr = pd.concat([(high - low), (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    swing_low = float(low.tail(12).min())
    hi52 = float(close.tail(252).max())
    lo_base = float(low.tail(40).min())            # base bottom for measured move

    stops = {
        "swing_low": round(swing_low * 0.995, 4),
        "ema20": round(ema20 * 0.98, 4),
        "atr2x": round(price - 2 * atr, 4),
    }
    measured = price + (price - lo_base)           # base depth projected up
    targets = {
        "hi52": round(hi52, 4),
        "measured_move": round(measured, 4),
        "atr3x": round(price + 3 * atr, 4),
    }
    # discard nonsensical candidates
    stops = {k: v for k, v in stops.items() if 0 < v < price}
    targets = {k: v for k, v in targets.items() if v > price * 1.01}
    if not stops or not targets:
        return None
    return {"price": price, "atr_pct": round(atr / price * 100, 2),
            "stops": stops, "targets": targets, "hi52": hi52}

# ----------------------------- Agent call ------------------------------------

AGENT_SYSTEM = """You are the risk analyst for a stock screening system on \
Bursa Malaysia. For each candidate you receive the current price, ATR%, a menu \
of structurally-derived stop candidates and target candidates, plus signal \
context (list, flags, streak, days to earnings if known).

For EACH candidate, choose the ONE stop and ONE target that best fit the \
stock's situation, and note any risk concerns (earnings proximity, stop \
tighter than 1.5x ATR, target blocked by nearby 52w high, extended move).

Reply with ONLY a JSON array, one object per candidate, no other text:
[{"ticker": "...", "stop_key": "...", "target_key": "...", "note": "<max 15 words>"}]

Rules: stop_key must be one of the provided stop keys; target_key one of the \
provided target keys. Prefer swing_low stops for fresh breakouts, ema20 for \
extended trends. Prefer conservative targets when a 52w high sits close above. \
The note is a caution or confirmation, not a buy/sell recommendation."""


def ask_agent(payload: list):
    """Send candidates to Claude; return {ticker: {stop_key, target_key, note}}."""
    if not ANTHROPIC_API_KEY:
        print("ANTHROPIC_API_KEY not set - using deterministic defaults.")
        return None
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 1500,
                "system": AGENT_SYSTEM,
                "messages": [{"role": "user",
                              "content": json.dumps(payload)}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        text = "".join(b.get("text", "") for b in resp.json()["content"])
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        picks = json.loads(text)
        return {p["ticker"]: p for p in picks if "ticker" in p}
    except Exception as e:
        print(f"Agent call failed ({e}) - using deterministic defaults.")
        return None

# ----------------------------- Card assembly (deterministic) -----------------

def default_choice(levels):
    """Fallback when the agent is unavailable: conservative structural picks."""
    stop_key = "swing_low" if "swing_low" in levels["stops"] else \
        max(levels["stops"], key=lambda k: levels["stops"][k])   # nearest stop
    target_key = min(levels["targets"], key=lambda k: levels["targets"][k])
    return stop_key, target_key

def build_cards(candidates, agent_picks):
    cards = []
    for c in candidates:
        lv = c["levels"]
        pick = (agent_picks or {}).get(c["ticker"])
        if pick and pick.get("stop_key") in lv["stops"] \
                and pick.get("target_key") in lv["targets"]:
            sk, tk = pick["stop_key"], pick["target_key"]
            note = str(pick.get("note", ""))[:120]
            agent_ok = True
        else:
            sk, tk = default_choice(lv)
            note = "agent unavailable - conservative defaults"
            agent_ok = False

        price, stop, target = lv["price"], lv["stops"][sk], lv["targets"][tk]
        risk, reward = price - stop, target - price
        rr = reward / risk if risk > 0 else 0.0
        # entry price at which this setup would meet the floor (same levels)
        viable_entry = (target + RR_FLOOR * stop) / (1 + RR_FLOOR)
        cards.append({
            **c, "stop": stop, "stop_key": sk, "target": target,
            "target_key": tk, "rr": rr, "viable_entry": viable_entry,
            "note": note, "agent_ok": agent_ok,
            "stop_pct": (stop / price - 1) * 100,
            "target_pct": (target / price - 1) * 100,
        })
    cards.sort(key=lambda x: -x["rr"])
    return cards

# ----------------------------- Telegram --------------------------------------

LEVEL_NAMES = {"swing_low": "swing low", "ema20": "EMA20", "atr2x": "2xATR",
               "hi52": "52w high", "measured_move": "measured move",
               "atr3x": "3xATR"}

def format_cards(cards, date_str):
    lines = [f"\U0001F3B4 <b>Trade Cards</b> \u2014 {date_str}",
             f"R:R floor {RR_FLOOR}:1 | analysis frames, not recommendations\n"]
    passing = [c for c in cards if c["rr"] >= RR_FLOOR]
    failing = [c for c in cards if c["rr"] < RR_FLOOR]

    if passing:
        lines.append("\u2705 <b>Meet the floor</b>")
        for c in passing:
            warn = "" if c["agent_ok"] else " \u26AA"
            lines.append(
                f"<b>{c['name']}</b> ({c['ticker']}) [{c['list']}] "
                f"\u2014 R:R <b>{c['rr']:.1f}:1</b>{warn}\n"
                f"  Entry {CURRENCY}{c['price']:.3f} | "
                f"Stop {CURRENCY}{c['stop']:.3f} "
                f"({LEVEL_NAMES.get(c['stop_key'], c['stop_key'])}, {c['stop_pct']:+.1f}%) | "
                f"Target {CURRENCY}{c['target']:.3f} "
                f"({LEVEL_NAMES.get(c['target_key'], c['target_key'])}, {c['target_pct']:+.1f}%)\n"
                f"  \U0001F4DD {c['note']}"
            )
    if failing:
        lines.append("\n\u274C <b>Below the floor (watch levels)</b>")
        for c in failing:
            lines.append(
                f"<b>{c['name']}</b> ({c['ticker']}) \u2014 R:R {c['rr']:.1f}:1 | "
                f"viable near {CURRENCY}{c['viable_entry']:.3f}\n"
                f"  \U0001F4DD {c['note']}"
            )
    if not cards:
        lines.append("No candidates had computable level structures today.")
    lines.append("\n<i>Stops/targets are structural levels computed at signal "
                 "time, selected by AI review. ATR-based. Not financial "
                 "advice.</i>")
    return lines

def chunk_lines(lines):
    chunks, cur = [], ""
    for line in lines:
        cand = (cur + "\n" + line) if cur else line
        if len(cand) > TG_CHAR_LIMIT:
            chunks.append(cur); cur = line
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks

def send_telegram(lines):
    chunks = chunk_lines(lines)
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

# ----------------------------- Main -------------------------------------------

def main():
    try:
        with open("results.json") as f:
            res = json.load(f)
    except Exception:
        print("No results.json - screener may have exited early (weekend/holiday).")
        return

    # Top candidates: momentum first, then early (reversals excluded from
    # trade cards by design - watchlist material, not entries)
    raw = [dict(r, list="momentum") for r in res.get("momentum", [])[:8]] + \
          [dict(r, list="early") for r in res.get("early", [])[:4]]
    raw = raw[:MAX_CARDS]
    if not raw:
        print("No candidates today - no trade cards.")
        return

    symbols = [r["ticker"] + TICKER_SUFFIX for r in raw]
    print(f"Computing levels for {len(symbols)} candidates...")
    data = yf.download(symbols, period="300d", interval="1d",
                       group_by="ticker", auto_adjust=True,
                       threads=True, progress=False)

    candidates, payload = [], []
    for r in raw:
        sym = r["ticker"] + TICKER_SUFFIX
        try:
            df = data[sym] if len(symbols) > 1 else data
            lv = compute_levels(df)
        except Exception:
            lv = None
        if lv is None:
            continue
        cand = {"ticker": r["ticker"], "name": r["name"],
                "list": r["list"], "flags": r.get("flags", ""),
                "levels": lv, "price": lv["price"]}
        candidates.append(cand)
        payload.append({"ticker": r["ticker"], "list": r["list"],
                        "flags": r.get("flags", ""),
                        "price": lv["price"], "atr_pct": lv["atr_pct"],
                        "pct_below_52w_high": round((lv["price"] / lv["hi52"] - 1) * 100, 1),
                        "stops": lv["stops"], "targets": lv["targets"]})

    if not candidates:
        print("No computable level structures - no cards.")
        return

    agent_picks = ask_agent(payload)
    cards = build_cards(candidates, agent_picks)
    date_lbl = (dt.datetime.utcnow() + dt.timedelta(hours=8)).strftime("%d %b %Y")
    send_telegram(format_cards(cards, date_lbl))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)

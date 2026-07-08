"""
Layer 3: market regime engine + playbooks for the Bursa screener.

Deterministic classifier over a small dashboard:
  - KLCI vs 50-day and 200-day MA
  - KLCI 20-day realized volatility vs its 1-year median
  - Breadth: % of analyzed stocks above EMA20 (from previous run's state)
  - Momentum-list churn (from previous run's state)

State machine (anti-thrash):
  - TRENDING / BEAR need 3 consecutive days of votes to activate
  - CHOPPY activates after 1 day (safe middle, also the default)
  - Minimum dwell 5 trading days once switched
  - Circuit breaker: KLCI single-day drop > 4% -> BEAR immediately
  - No TRENDING <-> BEAR jumps without passing through CHOPPY

State persists in regime_state.json (committed by the workflow); every
day's classification is appended to regime_log.csv for the scorecard.
"""

import os
import json
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf
import requests

STATE_FILE = "regime_state.json"
LOG_FILE = "regime_log.csv"

PLAYBOOKS = {
    "TRENDING": {
        "emoji": "\U0001F7E2",
        "mom_min_score": 3, "rsi_lo": 55, "rsi_hi": 80, "vol_sust": 1.3,
        "mom_rel_strength": None, "mom_need_pos_roc": False,
        "mom_near_high": None, "mom_all_criteria": False,
        "early_enabled": True, "early_supports": 2, "early_tight_mandatory": False,
        "rev_spike": 2.0, "rev_rsi": 35, "rev_both_mandatory": False,
        "rev_followthrough": False,
        "caps": (50, 25, 25), "rr_floor": 2.0,
    },
    "CHOPPY": {
        "emoji": "\U0001F7E1",
        "mom_min_score": 4, "rsi_lo": 58, "rsi_hi": 70, "vol_sust": 1.5,
        "mom_rel_strength": 3.0,      # 20d return must beat KLCI by >= 3%
        "mom_need_pos_roc": False,
        "mom_near_high": None, "mom_all_criteria": False,
        "early_enabled": True, "early_supports": 3, "early_tight_mandatory": True,
        "rev_spike": 2.5, "rev_rsi": 35, "rev_both_mandatory": False,
        "rev_followthrough": False,
        "caps": (20, 10, 10), "rr_floor": 2.5,
    },
    "BEAR": {
        "emoji": "\U0001F534",
        "mom_min_score": 5, "rsi_lo": 55, "rsi_hi": 80, "vol_sust": 1.3,
        "mom_rel_strength": None,
        "mom_need_pos_roc": True,     # absolute positive 20d return required
        "mom_near_high": 0.90,        # within 10% of 52w high
        "mom_all_criteria": True,
        "early_enabled": False,       # dormant: worst signal class in a bear
        "early_supports": 3, "early_tight_mandatory": True,
        "rev_spike": 3.0, "rev_rsi": 30, "rev_both_mandatory": True,
        "rev_followthrough": True,    # must hold spike-day low for 2 sessions
        "caps": (10, 0, 15), "rr_floor": 3.0,
    },
}

CONFIRM_DAYS = {"TRENDING": 3, "CHOPPY": 1, "BEAR": 3}
MIN_DWELL = 5
CIRCUIT_BREAKER_DROP = -4.0   # % single-day KLCI drop -> BEAR immediately
ADJACENT = {"TRENDING": {"CHOPPY"}, "CHOPPY": {"TRENDING", "BEAR"},
            "BEAR": {"CHOPPY"}}


# ----------------------------- Dashboard --------------------------------------

def klci_dashboard():
    """Fetch KLCI and compute the index-side signals."""
    df = yf.download("^KLSE", period="380d", interval="1d",
                     auto_adjust=True, progress=False)
    c = df["Close"].dropna()
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    last = float(c.iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    day_chg = float(c.iloc[-1] / c.iloc[-2] - 1) * 100
    rets = c.pct_change().dropna()
    vol20 = float(rets.tail(20).std() * np.sqrt(252) * 100)
    vol_series = rets.rolling(20).std().dropna() * np.sqrt(252) * 100
    vol_median = float(vol_series.median()) if len(vol_series) else vol20
    roc20 = float(c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) > 21 else 0.0
    return {"klci": last, "above_50": last >= ma50, "above_200": last >= ma200,
            "day_chg": day_chg, "vol20": vol20, "vol_elevated": vol20 > 1.3 * vol_median,
            "klci_roc20": roc20}


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"active": "CHOPPY", "days_in_regime": 0,
                "votes": [], "breadth": None, "churn": None,
                "prev_momentum": []}


def raw_vote(dash, state):
    """One day's raw classification from the dashboard."""
    breadth = state.get("breadth")        # T-1 breadth (0-100 or None)
    churn = state.get("churn")            # T-1 momentum-list churn (0-1 or None)

    bear = (not dash["above_50"]) and (not dash["above_200"]) and \
           (breadth is not None and breadth < 35)
    trending = dash["above_50"] and \
               (breadth is None or breadth > 50) and \
               (churn is None or churn < 0.5) and \
               not dash["vol_elevated"]
    if bear:
        return "BEAR"
    if trending:
        return "TRENDING"
    return "CHOPPY"


def step_state(dash, state):
    """Apply confirmation / dwell / circuit-breaker rules. Returns new state
    plus a switch explanation (or None)."""
    vote = raw_vote(dash, state)
    votes = (state.get("votes") or []) + [vote]
    votes = votes[-5:]
    active = state.get("active", "CHOPPY")
    days = int(state.get("days_in_regime", 0)) + 1
    switch_note = None

    # Circuit breaker overrides everything
    if dash["day_chg"] <= CIRCUIT_BREAKER_DROP:
        if active != "BEAR":
            switch_note = (f"CIRCUIT BREAKER: KLCI {dash['day_chg']:+.1f}% today "
                           f"-> BEAR immediately.")
            active, days = "BEAR", 1
        return _pack(active, days, votes, state), switch_note

    if vote != active and days > MIN_DWELL:
        need = CONFIRM_DAYS[vote]
        recent = votes[-need:]
        if len(recent) == need and all(v == vote for v in recent):
            target = vote if vote in ADJACENT[active] else "CHOPPY"
            if target != active:
                switch_note = (f"Regime switch {active} -> {target} "
                               f"(vote '{vote}' confirmed {need} day(s); "
                               f"KLCI {'above' if dash['above_50'] else 'below'} "
                               f"50d, breadth "
                               f"{state.get('breadth') if state.get('breadth') is not None else 'n/a'}%).")
                active, days = target, 1
    return _pack(active, days, votes, state), switch_note


def _pack(active, days, votes, old):
    return {"active": active, "days_in_regime": days, "votes": votes,
            "breadth": old.get("breadth"), "churn": old.get("churn"),
            "prev_momentum": old.get("prev_momentum", [])}


# ----------------------------- Agent briefing ----------------------------------

def agent_briefing(dash, state, switch_note):
    """Optional 2-3 sentence Haiku commentary. Skips gracefully without key."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return None
    try:
        payload = {"dashboard": dash, "active_playbook": state["active"],
                   "days_in_regime": state["days_in_regime"],
                   "recent_votes": state["votes"],
                   "breadth_pct": state.get("breadth"),
                   "momentum_churn": state.get("churn"),
                   "switch": switch_note}
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5", "max_tokens": 300,
                  "system": ("You are the market analyst for a Bursa Malaysia "
                             "screening system. Given the regime dashboard, "
                             "write a 2-3 sentence briefing: confirm or "
                             "question the mechanical regime call, and note "
                             "anything an operator should watch. Plain "
                             "language, no advice, no preamble."),
                  "messages": [{"role": "user",
                                "content": json.dumps(payload)}]},
            timeout=60)
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json()["content"]).strip()
    except Exception as e:
        print(f"Regime briefing skipped ({e})")
        return None


# ----------------------------- Public API --------------------------------------

def evaluate(date_str):
    """Run the full regime step. Returns (params, header_lines, state, klci_roc20)."""
    dash = klci_dashboard()
    state = load_state()
    state, switch_note = step_state(dash, state)
    pb = PLAYBOOKS[state["active"]]

    a50 = "\u2191" if dash["above_50"] else "\u2193"
    a200 = "\u2191" if dash["above_200"] else "\u2193"
    lines = [f"{pb['emoji']} Playbook: <b>{state['active']}</b> "
             f"(day {state['days_in_regime']}) | KLCI {dash['klci']:,.0f} "
             f"{a50}50d {a200}200d | vol {dash['vol20']:.0f}%"]
    b = state.get("breadth")
    if b is not None:
        lines[0] += f" | breadth {b:.0f}%"
    if switch_note:
        lines.append(f"\u26A0 {switch_note}")
    if state["active"] == "BEAR":
        lines.append("\U0001F534 Capital preservation mode - lists are "
                     "watchlists for the next cycle, not buy signals.")

    briefing = agent_briefing(dash, state, switch_note)
    if briefing:
        lines.append(f"\U0001F9E0 {briefing}")

    # log the day's call
    try:
        row = pd.DataFrame([{"date": date_str, "active": state["active"],
                             "days": state["days_in_regime"],
                             "klci": round(dash["klci"], 1),
                             "above50": dash["above_50"],
                             "above200": dash["above_200"],
                             "vol20": round(dash["vol20"], 1),
                             "breadth": state.get("breadth"),
                             "churn": state.get("churn"),
                             "switched": bool(switch_note)}])
        try:
            log = pd.read_csv(LOG_FILE)
            log = log[log["date"] != date_str]
            log = pd.concat([log, row], ignore_index=True)
        except Exception:
            log = row
        log.to_csv(LOG_FILE, index=False)
    except Exception as e:
        print(f"regime log write failed: {e}")

    return pb, lines, state, dash["klci_roc20"]


def finalize(state, breadth_pct, momentum_tickers):
    """Called by the screener AFTER screening: store today's breadth/churn
    for tomorrow's classification, then persist state."""
    prev = set(state.get("prev_momentum") or [])
    cur = set(momentum_tickers)
    churn = None
    if prev:
        churn = 1.0 - (len(prev & cur) / max(len(prev), 1))
    state["breadth"] = round(breadth_pct, 1) if breadth_pct is not None else None
    state["churn"] = round(churn, 2) if churn is not None else None
    state["prev_momentum"] = sorted(cur)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
    print(f"Regime state saved: {state['active']} day {state['days_in_regime']}, "
          f"breadth {state['breadth']}, churn {state['churn']}.")

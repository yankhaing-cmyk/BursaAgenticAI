"""
Auto-generate tickers.csv containing ALL Bursa Malaysia ordinary shares.
Source: KLSE Screener (free). Runs inside GitHub Actions where network works.
Keeps only 4-digit stock codes (Main + ACE market ordinary shares);
excludes warrants, preference shares, ETFs, bonds, and LEAP market.
"""

import re
import sys
import csv

import requests

URL = "https://www.klsescreener.com/v2/screener/quote_results"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.klsescreener.com/v2/",
}

def fetch_html() -> str:
    resp = requests.post(URL, data={"getquote": "1"}, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.text

def parse_stocks(html: str) -> dict:
    """Extract {code: name} for ordinary shares (exactly 4 digits)."""
    stocks = {}

    # Pattern A: link to stock page with a title attribute carrying the name
    for m in re.finditer(
        r'title="([^"]+)"[^>]*>\s*<a[^>]+/v2/stocks/view/(\w+)', html
    ):
        name, code = m.group(1).strip(), m.group(2).strip()
        stocks.setdefault(code, name)

    # Pattern B: any stock link; use link text as fallback name
    for m in re.finditer(r'/v2/stocks/view/(\w+)"[^>]*>([^<]+)<', html):
        code, text = m.group(1).strip(), m.group(2).strip()
        if code not in stocks and text:
            stocks[code] = text

    # Keep only ordinary shares: exactly 4 characters, all digits
    return {c: n for c, n in stocks.items() if len(c) == 4 and c.isdigit()}

def main():
    html = fetch_html()
    stocks = parse_stocks(html)

    if len(stocks) < 500:
        print(f"Only found {len(stocks)} stocks - source may have changed. "
              f"NOT overwriting tickers.csv.")
        sys.exit(1)

    with open("tickers.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "name"])
        for code in sorted(stocks):
            name = " ".join(stocks[code].split())  # clean whitespace
            writer.writerow([f"{code}.KL", name])

    print(f"Wrote tickers.csv with {len(stocks)} stocks.")

if __name__ == "__main__":
    main()

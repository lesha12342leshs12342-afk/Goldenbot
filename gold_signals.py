#!/usr/bin/env python3
"""
Анализатор золота (XAU): технический анализ на нескольких таймфреймах
+ свечные паттерны + новости + макро (доллар, доходность облигаций).

Установка:  pip install yfinance pandas numpy
Запуск:     python gold_signals.py            (один анализ)
            python gold_signals.py --loop 15  (повтор каждые 15 минут)

Это образовательный инструмент, а не финансовая рекомендация.
"""
import sys, time, urllib.request, xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yfinance as yf

SYMBOL = "GC=F"  # фьючерс на золото; можно "XAUUSD=X"
# имя: (интервал yfinance, период, ресемплинг, вес)
TFS = {"15m": ("15m", "30d", None, 1), "1h": ("1h", "60d", None, 2),
       "4h": ("1h", "180d", "4h", 3), "1d": ("1d", "2y", None, 3)}

NEWS_URL = ("https://news.google.com/rss/search?q=gold+price+OR+Fed+OR+inflation"
            "+OR+%22safe+haven%22&hl=en-US&gl=US&ceid=US:en")
BULL = ["rally", "surge", "record high", "safe haven", "rate cut", "dovish", "weaker dollar",
        "weak dollar", "geopolitical", "tensions", "gains", "rises", "jumps", "climbs",
        "soars", "haven demand", "central bank buying", "inflation fears", "war"]
BEAR = ["falls", "drops", "slump", "rate hike", "hawkish", "stronger dollar", "strong dollar",
        "yields rise", "selloff", "sell-off", "slides", "declines", "lower", "pressure",
        "profit-taking", "retreats", "tumbles", "ceasefire", "risk-on"]


# ---------- данные ----------
def load(symbol, interval, period, resample=None):
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    if resample:
        df = df.resample(resample).agg({"Open": "first", "High": "max", "Low": "min",
                                        "Close": "last", "Volume": "sum"}).dropna()
    return df


# ---------- индикаторы ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def candle_patterns(df):
    """Возвращает (балл, описание) по последним свечам."""
    o, h, l, c = (df[k].iloc[-1] for k in ("Open", "High", "Low", "Close"))
    po, pc = df.Open.iloc[-2], df.Close.iloc[-2]
    body, rng = abs(c - o), max(h - l, 1e-9)
    lower, upper = min(o, c) - l, h - max(o, c)
    if pc < po and c > o and c >= po and o <= pc:
        return 1, "бычье поглощение"
    if pc > po and c < o and c <= po and o >= pc:
        return -1, "медвежье поглощение"
    if lower > 2 * body and upper < body and body / rng < 0.35:
        return 1, "молот / бычий пин-бар"
    if upper > 2 * body and lower < body and body / rng < 0.35:
        return -1, "падающая звезда / медвежий пин-бар"
    if body / rng < 0.1:
        return 0, "доджи (неопределённость)"
    return 0, ""


def analyze(df):
    """Технический балл в диапазоне [-1, 1] и список причин."""
    c = df.Close
    score, why = 0.0, []
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    if c.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]: score += 2; why.append("тренд вверх (цена>EMA50>EMA200)")
    elif c.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]: score -= 2; why.append("тренд вниз (цена<EMA50<EMA200)")
    if e20.iloc[-1] > e50.iloc[-1]: score += 1
    else: score -= 1

    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]: score += 1; why.append("MACD растёт")
    elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]: score -= 1; why.append("MACD падает")

    r = rsi(c).iloc[-1]
    if r < 30: score += 1; why.append(f"RSI {r:.0f} перепродан")
    elif r > 70: score -= 1; why.append(f"RSI {r:.0f} перекуплен")

    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    if c.iloc[-1] < (ma - 2 * sd).iloc[-1]: score += 1; why.append("цена ниже нижней полосы Боллинджера")
    elif c.iloc[-1] > (ma + 2 * sd).iloc[-1]: score -= 1; why.append("цена выше верхней полосы Боллинджера")

    lo, hi = df.Low.rolling(14).min(), df.High.rolling(14).max()
    k = 100 * (c - lo) / (hi - lo)
    d = k.rolling(3).mean()
    if k.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1]: score += 1; why.append("стохастик: разворот вверх")
    elif k.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1]: score -= 1; why.append("стохастик: разворот вниз")

    cs, name = candle_patterns(df)
    score += cs
    if name: why.append(f"свеча: {name}")

    return float(np.clip(score / 8, -1, 1)), why, float(atr(df).iloc[-1])


# ---------- новости ----------
def news_sentiment(limit=25):
    try:
        req = urllib.request.Request(NEWS_URL, headers={"User-Agent": "Mozilla/5.0"})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        return 0.0, [f"новости недоступны: {e}"]
    titles = [i.findtext("title", "") for i in root.iter("item")][:limit]
    total, shown = 0.0, []
    for n, t in enumerate(titles):
        tl = t.lower()
        s = sum(w in tl for w in BULL) - sum(w in tl for w in BEAR)
        weight = 1 - n / (2 * max(len(titles), 1))  # свежие новости весомее
        total += np.clip(s, -2, 2) * weight
        if s and len(shown) < 5:
            shown.append(f"{'▲' if s > 0 else '▼'} {t[:90]}")
    return float(np.clip(total / max(len(titles), 1) * 2, -1, 1)), shown


# ---------- макро ----------
def macro_score():
    """Рост доллара и доходностей обычно давит на золото, и наоборот."""
    s, why = 0.0, []
    for sym, name in (("DX-Y.NYB", "индекс доллара"), ("^TNX", "доходность 10Y")):
        try:
            c = load(sym, "1d", "1mo").Close
            ch = c.iloc[-1] / c.iloc[-5] - 1
            if ch > 0.005: s -= 0.5; why.append(f"{name} растёт → давление на золото")
            elif ch < -0.005: s += 0.5; why.append(f"{name} падает → поддержка золоту")
        except Exception:
            pass
    return s, why


# ---------- итог ----------
def run():
    print(f"\n{'=' * 60}\nАнализ {SYMBOL}  ·  {time.strftime('%Y-%m-%d %H:%M')}\n{'=' * 60}")
    tech_sum, wsum, price, atr_1h = 0.0, 0, None, None
    for name, (iv, per, rs, w) in TFS.items():
        try:
            df = load(SYMBOL, iv, per, rs)
            sc, why, a = analyze(df)
        except Exception as e:
            print(f"[{name}] ошибка данных: {e}"); continue
        tech_sum += sc * w; wsum += w
        price = float(df.Close.iloc[-1])
        if name == "1h": atr_1h = a
        print(f"[{name:>3}] балл {sc:+.2f} | " + "; ".join(why[:4]))
    if not wsum:
        print("Нет данных."); return
    tech = tech_sum / wsum
    news, headlines = news_sentiment()
    macro, mwhy = macro_score()
    print("\nНовости:", f"{news:+.2f}"); [print("  ", h) for h in headlines]
    print("Макро:", f"{macro:+.2f}", "; ".join(mwhy))

    total = 0.6 * tech + 0.25 * news + 0.15 * macro
    if total >= 0.5: sig = "СИЛЬНАЯ ПОКУПКА"
    elif total >= 0.25: sig = "ПОКУПКА"
    elif total <= -0.5: sig = "СИЛЬНАЯ ПРОДАЖА"
    elif total <= -0.25: sig = "ПРОДАЖА"
    else: sig = "ЖДАТЬ (нет чёткого сигнала)"

    print(f"\nТехника {tech:+.2f} | Новости {news:+.2f} | Макро {macro:+.2f}")
    print(f">>> СИГНАЛ: {sig}  (итог {total:+.2f}), цена {price:.2f}")
    if atr_1h and abs(total) >= 0.25:
        d = 1 if total > 0 else -1
        print(f"    Стоп: {price - d * 1.5 * atr_1h:.2f}  |  Тейк: {price + d * 3 * atr_1h:.2f}"
              "  (1.5 и 3 ATR на 1h, риск/прибыль 1:2)")
    print("\nНе является финансовой рекомендацией. Риск потери капитала.")


if __name__ == "__main__":
    if "--loop" in sys.argv:
        mins = int(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            run(); time.sleep(mins * 60)
    else:
        run()

"""Gold Signals: анализ золота (мультитаймфрейм + свечи + новости + макро).
Образовательный инструмент, не финансовая рекомендация."""
import json, re, ssl, threading, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import numpy as np
import certifi
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.utils import escape_markup

CTX = ssl.create_default_context(cafile=certifi.where())
SYMBOL = "GC=F"
# имя: (интервал, период, группировка в секундах, вес)
TFS = {"15m": ("15m", "30d", None, 1), "1h": ("1h", "60d", None, 2),
       "4h": ("1h", "180d", 14400, 3), "1d": ("1d", "2y", None, 3)}
NEWS_URL = ("https://news.google.com/rss/search?q=gold+price+OR+Fed+OR+inflation"
            "+OR+%22safe+haven%22&hl=en-US&gl=US&ceid=US:en")
BULL = ["rally", "surge", "record high", "safe haven", "rate cut", "dovish", "weaker dollar",
        "weak dollar", "geopolitical", "tensions", "gains", "rises", "jumps", "climbs",
        "soars", "haven demand", "central bank buying", "inflation fears", "war"]
BEAR = ["falls", "drops", "slump", "rate hike", "hawkish", "stronger dollar", "strong dollar",
        "yields rise", "selloff", "sell-off", "slides", "declines", "lower", "pressure",
        "profit-taking", "retreats", "tumbles", "ceasefire", "risk-on"]


def http(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=20, context=CTX).read()


def load(sym, interval, rng, step=None):
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/"
           f"{urllib.parse.quote(sym)}?interval={interval}&range={rng}")
    r = json.loads(http(url))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    ts = np.array(r["timestamp"], dtype=float)
    a = np.array([q["open"], q["high"], q["low"], q["close"]], dtype=float)
    m = ~np.isnan(a).any(axis=0)
    ts, a = ts[m], a[:, m]
    if step:
        k = (ts // step).astype(int)
        s = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        e = np.r_[s[1:], len(k)]
        return (a[0][s], np.array([a[1][i:j].max() for i, j in zip(s, e)]),
                np.array([a[2][i:j].min() for i, j in zip(s, e)]), a[3][e - 1])
    return a[0], a[1], a[2], a[3]


# ---------- индикаторы (чистый numpy) ----------
def ewm(x, alpha):
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out

def ema(x, n): return ewm(x, 2 / (n + 1))

def rsi(c, n=14):
    d = np.diff(c)
    up, dn = ewm(np.clip(d, 0, None), 1 / n), ewm(-np.clip(d, None, 0), 1 / n)
    return 100 - 100 / (1 + up[-1] / max(dn[-1], 1e-12))

def atr(h, l, c, n=14):
    pc = c[:-1]
    tr = np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - pc), abs(l[1:] - pc)])
    return ewm(tr, 1 / n)[-1]

def candle(o, h, l, c):
    O, H, L, C, PO, PC = o[-1], h[-1], l[-1], c[-1], o[-2], c[-2]
    body, rng = abs(C - O), max(H - L, 1e-9)
    lo, up = min(O, C) - L, H - max(O, C)
    if PC < PO and C > O and C >= PO and O <= PC: return 1, "бычье поглощение"
    if PC > PO and C < O and C <= PO and O >= PC: return -1, "медвежье поглощение"
    if lo > 2 * body and up < body and body / rng < 0.35: return 1, "молот"
    if up > 2 * body and lo < body and body / rng < 0.35: return -1, "падающая звезда"
    if body / rng < 0.1: return 0, "доджи"
    return 0, ""

def analyze(o, h, l, c):
    s, why = 0.0, []
    e20, e50, e200 = ema(c, 20)[-1], ema(c, 50)[-1], ema(c, 200)[-1]
    if c[-1] > e50 > e200: s += 2; why.append("тренд вверх")
    elif c[-1] < e50 < e200: s -= 2; why.append("тренд вниз")
    s += 1 if e20 > e50 else -1
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    if hist[-1] > 0 and hist[-1] > hist[-2]: s += 1; why.append("MACD растёт")
    elif hist[-1] < 0 and hist[-1] < hist[-2]: s -= 1; why.append("MACD падает")
    r = rsi(c)
    if r < 30: s += 1; why.append(f"RSI {r:.0f} перепродан")
    elif r > 70: s -= 1; why.append(f"RSI {r:.0f} перекуплен")
    w = c[-20:]; ma, sd = w.mean(), w.std()
    if c[-1] < ma - 2 * sd: s += 1; why.append("ниже полосы Боллинджера")
    elif c[-1] > ma + 2 * sd: s -= 1; why.append("выше полосы Боллинджера")
    ks = [100 * (c[-i] - l[-i - 13:len(l) - i + 1].min()) /
          max(h[-i - 13:len(h) - i + 1].max() - l[-i - 13:len(l) - i + 1].min(), 1e-9)
          for i in (1, 2, 3)]
    k, d = ks[0], np.mean(ks)
    if k < 20 and k > d: s += 1; why.append("стохастик вверх")
    elif k > 80 and k < d: s -= 1; why.append("стохастик вниз")
    n = 14
    up_, dn_ = np.diff(h), -np.diff(l)
    pdm = np.where((up_ > dn_) & (up_ > 0), up_, 0.0); mdm = np.where((dn_ > up_) & (dn_ > 0), dn_, 0.0)
    tr = np.maximum.reduce([h[1:] - l[1:], abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])])
    at = np.maximum(ewm(tr, 1 / n), 1e-9)
    pdi, mdi = 100 * ewm(pdm, 1 / n) / at, 100 * ewm(mdm, 1 / n) / at
    adx = ewm(100 * abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-9), 1 / n)[-1]
    if adx > 25:
        s += 1 if pdi[-1] > mdi[-1] else -1; why.append(f"ADX {adx:.0f}: сильный тренд {'вверх' if pdi[-1] > mdi[-1] else 'вниз'}")
    tp = (h + l + c)[-20:] / 3
    cci = (tp[-1] - tp.mean()) / max(0.015 * np.abs(tp - tp.mean()).mean(), 1e-9)
    if cci < -100: s += 1; why.append(f"CCI {cci:.0f} перепродан")
    elif cci > 100: s -= 1; why.append(f"CCI {cci:.0f} перекуплен")
    cs, name = candle(o, h, l, c)
    s += cs
    if name: why.append(f"свеча: {name}")
    return float(np.clip(s / 10, -1, 1)), why, atr(h, l, c)


def news_sentiment(limit=25):
    try:
        titles = [i.findtext("title", "") for i in ET.fromstring(http(NEWS_URL)).iter("item")][:limit]
    except Exception as e:
        return 0.0, [f"новости недоступны: {e}"]
    tot, shown = 0.0, []
    for n, t in enumerate(titles):
        tl = t.lower()
        s = sum(w in tl for w in BULL) - sum(w in tl for w in BEAR)
        tot += np.clip(s, -2, 2) * (1 - n / (2 * max(len(titles), 1)))
        if s and len(shown) < 5:
            shown.append(("+ " if s > 0 else "- ") + t[:85])
    return float(np.clip(tot / max(len(titles), 1) * 2, -1, 1)), shown


HIGH = re.compile(r"\b(fed|powell|fomc|cpi|payrolls?|nfp|rate decision|war|strike|missile|sanctions?|tariffs?|attack|emergency|breaking)\b")
BANKS = ["Goldman", "JPMorgan", "UBS", "Citi", "Bank of America", "BofA", "HSBC", "Morgan Stanley",
         "Commerzbank", "ANZ", "Deutsche", "Barclays", "Standard Chartered", "Macquarie", "Wells Fargo"]
FC_URL = ("https://news.google.com/rss/search?q=gold+price+forecast+target+OR+outlook+Goldman+OR+JPMorgan"
          "+OR+UBS+OR+Citi+OR+HSBC+OR+%22Morgan+Stanley%22+OR+Commerzbank+OR+ANZ+when:7d&hl=en-US&gl=US&ceid=US:en")


def macro_score():
    s, why = 0.0, []
    for sym, name, sg, th in (("DX-Y.NYB", "индекс доллара", -1, .005), ("^TNX", "доходность 10Y", -1, .005),
                              ("^VIX", "VIX (страх)", 1, .05)):
        try:
            c = load(sym, "1d", "1mo")[3]
            ch = c[-1] / c[-5] - 1
            if abs(ch) > th:
                d = sg * (1 if ch > 0 else -1) * 0.4
                s += d
                why.append(f"{name} {'растёт' if ch > 0 else 'падает'}: {'плюс' if d > 0 else 'минус'} для золота")
        except Exception:
            pass
    return float(np.clip(s, -1, 1)), why


def rss(url, limit=40):
    out, now = [], datetime.now(timezone.utc)
    for i in ET.fromstring(http(url)).iter("item"):
        try: age = (now - parsedate_to_datetime(i.findtext("pubDate"))).total_seconds() / 60
        except Exception: age = 9999
        out.append((i.findtext("title", ""), age))
        if len(out) >= limit: break
    return out


def forecasts(titles, price):
    """Прогнозы банков/брокеров из заголовков: цели цены и тон (бычий/медвежий)."""
    sc, lines = [], []
    for t, _ in titles:
        tl = t.lower()
        bank = next((b for b in BANKS if b.lower() in tl), None)
        nums = [float(x.replace(",", "")) for x in re.findall(r"\$\s?(\d{1,2},?\d{3})(?!\d)", t)]
        nums = [n for n in nums if .6 * price < n < 1.8 * price]
        if nums:
            v = float(np.clip((nums[0] / price - 1) * 8, -1, 1))
        else:
            v = (any(w in tl for w in ("bullish", "upside", "raises", "higher", "surge"))
                 - any(w in tl for w in ("bearish", "downside", "cuts", "lower", "slump"))) * 0.5
        if (bank or nums) and v:
            sc.append(v)
            if bank and len(lines) < 6:
                lines.append(f"{bank}: " + (f"цель {nums[0]:.0f}, " if nums else "") + ("бычий" if v > 0 else "медвежий"))
    return (float(np.clip(np.mean(sc), -1, 1)) if sc else 0.0), list(dict.fromkeys(lines))


def levels(price, a):
    """Классические пивоты по вчерашнему дню: цена у поддержки (+) или сопротивления (-)."""
    h, l, c = cached("1d", TTL["1d"], lambda: load(SYMBOL, "1d", "2y", None))[1:]
    H, L, C = h[-2], l[-2], c[-2]
    P = (H + L + C) / 3
    lv = {"S2": P - (H - L), "S1": 2 * P - H, "P": P, "R1": 2 * P - L, "R2": P + H - L}
    a = a or price * .002
    sc = sum((.6 if k[0] == "S" else -.6 if k[0] == "R" else 0) for k, v in lv.items() if abs(price - v) < .6 * a)
    return float(np.clip(sc, -1, 1)), "  ".join(f"{k} {v:.0f}" for k, v in lv.items())


def sharp(p, fresh):
    """Риск резкого движения в ближайшие минуты: волатильность, squeeze, импульс, новости."""
    o, h, l, c = cached("1m", 15, lambda: load(SYMBOL, "1m", "1d"))
    h, l, c = h[-120:].copy(), l[-120:].copy(), c[-120:].copy()
    if len(c) < 70: return 0.0, 0, ["мало минутных данных (рынок закрыт?)"]
    if p: c[-1] = p; h[-1] = max(h[-1], p); l[-1] = min(l[-1], p)
    af, asl = atr(h, l, c, 5), max(atr(h, l, c, 60), 1e-9)
    risk, why, d = 0.0, [], 0
    if af / asl > 1.6: risk += .3; why.append(f"волатильность x{af / asl:.1f} от нормы")
    bw = [c[i - 20:i].std() for i in range(20, len(c) + 1)]
    if bw[-1] <= np.percentile(bw, 15): risk += .25; why.append("сжатие (squeeze): близок пробой")
    mv = (c[-1] - c[-6]) / asl
    if abs(mv) > 2: risk += .3; d = 1 if mv > 0 else -1; why.append(f"импульс {mv:+.1f} ATR за 5 мин")
    if fresh: risk += .3; why.append("важная новость: " + fresh[:60])
    return min(risk, 1.0), d, why


CACHE = {}
TTL = {"15m": 30, "1h": 60, "4h": 60, "1d": 300}


def cached(key, ttl, fn):
    """Тяжёлые данные (свечи, новости, макро) обновляем редко, а не каждую секунду."""
    t, v = CACHE.get(key, (0, None))
    if v is None or time.time() - t > ttl:
        try:
            v = fn()
        except Exception:
            if v is None:
                raise
        CACHE[key] = (time.time(), v)
    return v


def live_price():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + urllib.parse.quote(SYMBOL) + "?interval=1m&range=1d"
    return float(json.loads(http(url))["chart"]["result"][0]["meta"]["regularMarketPrice"])


LAST = {"p": None}


def compute(pos=0, entry=None):
    try: p = live_price()
    except Exception: p = None
    lines, tech, wsum, price, a1 = [], 0.0, 0, None, None
    for name, (iv, per, step, w) in TFS.items():
        try:
            o, h, l, c = cached(name, TTL[name], lambda iv=iv, per=per, step=step: load(SYMBOL, iv, per, step))
            o, h, l, c = o[-400:], h[-400:].copy(), l[-400:].copy(), c[-400:].copy()
            if p: c[-1] = p; h[-1] = max(h[-1], p); l[-1] = min(l[-1], p)
            sc, why, a = analyze(o, h, l, c)
        except Exception as e:
            lines.append(f"[{name}] ошибка: {e}"); continue
        tech += sc * w; wsum += w; price = float(c[-1])
        if name == "1h": a1 = a
        lines.append(f"[b][{name}][/b] {sc:+.2f}: " + escape_markup("; ".join(why[:5])))
    if not wsum:
        return "НЕТ ДАННЫХ", (1, .4, .4, 1), "\n".join(lines), False
    LAST["p"], tech = price, tech / wsum
    news, heads = cached("news", 300, news_sentiment)
    macro, mwhy = cached("macro", 300, macro_score)
    fc, fclines = forecasts(cached("fc", 600, lambda: rss(FC_URL)), price)
    lv, lvtxt = levels(price, a1)
    fresh = next((t for t, age in cached("newsraw", 120, lambda: rss(NEWS_URL)) if age <= 30 and HIGH.search(t.lower())), "")
    try: risk, rdir, rwhy = sharp(p, fresh)
    except Exception as e: risk, rdir, rwhy = 0.0, 0, [f"минутные данные недоступны"]
    tot = .5 * tech + .15 * news + .15 * fc + .1 * macro + .1 * lv
    sig, col = "ЖДАТЬ", (.8, .8, .8, 1)
    if tot >= .5: sig, col = "СИЛЬНАЯ ПОКУПКА", (0, .9, .4, 1)
    elif tot >= .25: sig, col = "ПОКУПКА", (.3, .85, .4, 1)
    elif tot <= -.5: sig, col = "СИЛЬНАЯ ПРОДАЖА", (1, .2, .2, 1)
    elif tot <= -.25: sig, col = "ПРОДАЖА", (1, .45, .4, 1)
    head, alert, adv = sig, False, ""
    if pos:
        nm = "ЛОНГ" if pos > 0 else "ШОРТ"
        stop = take = None
        if a1 and entry: stop, take = entry - pos * 1.5 * a1, entry + pos * 3 * a1
        why = None
        if stop and (price - stop) * pos <= 0: why = "сработал стоп"
        elif take and (price - take) * pos >= 0: why = "цель достигнута, фиксируйте прибыль"
        elif tot * pos <= -.25: why = "сигнал развернулся против позиции"
        elif risk >= .5 and rdir != pos:
            why = "ожидается резкое движение ПРОТИВ вас" if rdir == -pos else "ожидается резкий скачок, направление неясно"
        if why: head, col, alert, adv = f"ЗАКРЫТЬ {nm}!", (1, .25, .25, 1), True, why
        else:
            head, col = f"ДЕРЖАТЬ {nm}", (.5, .8, 1, 1)
            adv = "резкое движение в вашу сторону: подтяните стоп" if risk >= .5 else "условия для удержания"
        adv += f"\nВход {entry:.2f}, P&L {(price - entry) * pos:+.2f}" if entry else ""
    elif risk >= .5:
        adv = "ВНИМАНИЕ: высокий риск скачка, вход опасен"
    out = [adv, f"Цена {price:.2f} | Итог {tot:+.2f} | Рынок: {sig} ({time.strftime('%H:%M:%S')})",
           f"Техника {tech:+.2f} Новости {news:+.2f} Прогнозы {fc:+.2f} Макро {macro:+.2f} Уровни {lv:+.2f}", "",
           f"[b]Риск резкого движения: {risk:.0%}[/b]"] + [escape_markup("  - " + w) for w in rwhy] + [""] + lines
    out += ["", "Пивоты: " + lvtxt]
    if a1 and abs(tot) >= .25 and not pos:
        d = 1 if tot > 0 else -1
        out += [f"Стоп {price - d * 1.5 * a1:.2f}  Тейк {price + d * 3 * a1:.2f} (ATR 1h)"]
    out += ["", "[b]Прогнозы банков (из заголовков)[/b]"] + [escape_markup(x) for x in fclines or ["не найдено"]]
    out += ["", "[b]Новости[/b]"] + [escape_markup(x) for x in heads]
    out += ["", "[b]Макро[/b]"] + [escape_markup(m) for m in mwhy]
    out += ["", "Не финансовая рекомендация. Риск потери капитала."]
    return head, col, "\n".join(out), alert


class GoldApp(App):
    def build(self):
        self.title = "Gold Signals"
        self.bg = (.07, .07, .09, 1)
        Window.clearcolor = self.bg
        root = BoxLayout(orientation="vertical", padding=12, spacing=10)
        self.sig = Label(text="Загрузка...", font_size="26sp", bold=True, size_hint_y=None, height=90)
        sv = ScrollView()
        self.det = Label(text="", markup=True, size_hint_y=None, halign="left", valign="top", font_size="14sp")
        self.det.bind(width=lambda *_: setattr(self.det, "text_size", (self.det.width - 10, None)),
                      texture_size=lambda *_: setattr(self.det, "height", self.det.texture_size[1]))
        sv.add_widget(self.det)
        row = BoxLayout(size_hint_y=None, height=56, spacing=8)
        self.pbtn = Button(text="Позиция: НЕТ", on_release=self.cycle)
        self.btn = Button(text="Пауза", on_release=self.toggle)
        row.add_widget(self.pbtn); row.add_widget(self.btn)
        root.add_widget(self.sig); root.add_widget(sv); root.add_widget(row)
        self.running = self.alive = True
        self.pos, self.entry = 0, None
        threading.Thread(target=self._loop, daemon=True).start()
        return root

    def cycle(self, *_):  # НЕТ -> LONG -> SHORT -> НЕТ; цена входа = текущая
        self.pos = {0: 1, 1: -1, -1: 0}[self.pos]
        self.entry = LAST["p"] if self.pos else None
        self.pbtn.text = "Позиция: " + {0: "НЕТ", 1: "LONG", -1: "SHORT"}[self.pos]

    def toggle(self, *_):
        self.running = not self.running
        self.btn.text = "Пауза" if self.running else "Продолжить"

    def on_stop(self):
        self.alive = False

    def _loop(self):
        while self.alive:
            t0, delay = time.time(), 1.0
            if self.running:
                try:
                    res = compute(self.pos, self.entry)
                except Exception as e:
                    res, delay = ("ОШИБКА", (1, .4, .4, 1), escape_markup(str(e)), False), 5.0
                Clock.schedule_once(lambda dt, r=res: self._show(*r))
            time.sleep(max(0.05, delay - (time.time() - t0)))

    def _show(self, head, col, text, alert):
        self.sig.text, self.sig.color, self.det.text = head, col, text
        Window.clearcolor = (.35, .05, .05, 1) if alert else self.bg


if __name__ == "__main__":
    GoldApp().run()

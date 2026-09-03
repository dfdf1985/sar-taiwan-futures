"""拋物線 SAR 順勢策略回測引擎（100% 曝險・獲利再投入）。

設計要點
--------
1. 訊號：SAR 於 K 棒收盤產生，下一根 K 棒開盤成交（以收盤價計算損益，
   等同於「訊號 t-1 決定部位 t」）。
2. 部位：目標曝險 = 當期權益 × EXPOSURE_RATIO，獲利再投入。口數只在
   「新進場」時計算一次，單筆存續期間不加減碼。分母使用**未調整**的真實
   指數價格，因為名目曝險是實務概念。COMPOUND=False 可切回固定金額口徑。
3. 結算日：每月第三個星期三強制平倉且不新進場。夜盤（15:00 之後）歸屬
   次一交易日，故結算日的範圍涵蓋前一晚夜盤。

資料：本專案不隨附回測資料（授權限制），見 DATA.md 與 data/README.md。
"""
import os

import numpy as np
import pandas as pd

from src.sar import parabolic_sar
from src.settlement import third_wednesday
from src import config as C


_MISSING = """找不到資料檔：{p}

本專案不隨附回測資料——原始分鐘資料來自商業供應商（啟書、TOUCHANCE），
授權不允許再散布，由其取樣而得的 60 分 K 序列同樣不發布。請擇一：

  1. 取得授權資料後執行  python scripts/01_build_data.py <parquet 路徑>
  2. 依 DATA.md 第四節規格自建等效序列
  3. 只想確認程式可執行：python scripts/00_make_sample_data.py
     （產生合成資料，數字不具研究意義）

詳見 data/README.md。"""


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(_MISSING.format(p=path))
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_bars(path=None):
    return _read(path or C.DATA_FILE)


def load_ref(path=None):
    return _read(path or C.BH_REF_FILE)


def mark_settlement(index):
    """結算日旗標。夜盤（>=15:00）歸屬次一交易日。"""
    nxt = (index + pd.Timedelta(days=1)).date
    same = index.date
    trade_date = np.where(index.hour >= 15, nxt, same)
    months = sorted({(d.year, d.month) for d in trade_date})
    sw = {ym: third_wednesday(*ym) for ym in months}
    return np.array([d == sw[(d.year, d.month)] for d in trade_date])


def run(bars, ref, af=None, af_max=None, direction='long',
        day_only=False, one_lot=False, compound=None, exposure=None,
        target=None, mult=None, fee=None, capital=None):
    """執行回測，回傳含 position / net_pnl / fee / equity 的 DataFrame。

    direction : 'long' 純多單 | 'short' 純空單 | 'both' 多空皆做
    day_only  : True 時只用日盤 K 棒重新計算 SAR（完全不參與夜盤）
    one_lot   : True 時固定 1 口（用於檢視訊號本身的體質）
    compound  : True 時目標曝險 = 當期權益 × exposure（獲利再投入，與 B&H 同為
                複利口徑）；False 時固定用 target（加法口徑）。
    """
    af = C.AF if af is None else af
    af_max = af if af_max is None else af_max
    compound = C.COMPOUND if compound is None else compound
    exposure = C.EXPOSURE_RATIO if exposure is None else exposure
    target = C.TARGET_NOTIONAL if target is None else target
    mult = C.MULTIPLIER if mult is None else mult
    fee = C.FEE_PER_SIDE if fee is None else fee
    capital = C.CAPITAL if capital is None else capital

    if day_only:
        bars = bars[bars.index.hour.isin(C.DAY_HOURS)]

    px = ref['close_none'].reindex(bars.index).ffill().values   # 真實指數（算名目用）
    sar = parabolic_sar(bars, af, af, af_max)
    settle = mark_settlement(bars.index)
    close = sar['Close'].values
    n = len(bars)

    raw = np.where(sar['trend'].values == 1, 1, -1)
    if direction == 'long':
        raw = np.where(raw == 1, 1, 0)
    elif direction == 'short':
        raw = np.where(raw == -1, -1, 0)

    pos = np.zeros(n, dtype=int)
    cost = np.zeros(n)
    pnl = np.zeros(n)
    equity = np.full(n, capital, dtype=float)
    eq = capital
    held = 0
    half_rev = C.REVERSAL_FEE / 2
    for t in range(1, n):
        tgt = raw[t - 1]
        if settle[t]:
            held = 0
        elif tgt == 0:
            held = 0
        elif held == 0 or np.sign(held) != tgt:
            # 新進場（或直接反手）：此刻決定口數，單筆存續期間不再調整
            budget = eq * exposure if compound else target
            lots = 1 if one_lot else max(int(budget // (px[t - 1] * mult)), 0)
            held = tgt * lots
        pos[t] = held

        prev = pos[t - 1]
        if prev != 0 and held != 0 and np.sign(prev) != np.sign(held):
            c = (abs(prev) + abs(held)) * half_rev
        else:
            c = abs(held - prev) * fee
        cost[t] = c
        pnl[t] = held * (close[t] - close[t - 1]) * mult
        eq += pnl[t] - c
        equity[t] = eq

    return pd.DataFrame({'position': pos, 'net_pnl': pnl - cost,
                         'fee': cost, 'equity': equity}, index=bars.index)


def _trade_spans(pos):
    """回傳每筆交易的 (起始索引, 結束索引)。一筆 = 一段連續同方向持倉。"""
    spans, i, n = [], 0, len(pos)
    while i < n:
        if pos[i] != 0:
            start, side = i, np.sign(pos[i])
            while i < n and np.sign(pos[i]) == side:
                i += 1
            end = i if (i < n and pos[i] == 0) else i - 1
            spans.append((start, end))
        else:
            i += 1
    return spans


def split_trades(seg):
    """每筆交易的淨損益（元）；出場手續費計入該筆。"""
    pos = seg['position'].values
    pnl = seg['net_pnl'].values
    return np.array([pnl[s:e + 1].sum() for s, e in _trade_spans(pos)])


def split_trade_returns(seg, capital=None):
    """每筆交易的報酬率 = 該筆淨損益 ÷ 進場時權益。

    複利制下後期交易的絕對金額較大，直接用金額計算勝率／盈虧比／獲利因子
    會過度加權後期交易；改用報酬率即為 scale-free。
    """
    capital = C.CAPITAL if capital is None else capital
    pos = seg['position'].values
    pnl = seg['net_pnl'].values
    eq = seg['equity'].values if 'equity' in seg else None
    out = []
    for s, e in _trade_spans(pos):
        base = capital if eq is None else (eq[s - 1] if s > 0 else eq[s])
        out.append(pnl[s:e + 1].sum() / base if base > 0 else 0.0)
    return np.array(out)

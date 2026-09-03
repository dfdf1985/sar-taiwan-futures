"""績效指標。

口徑（2026/09 更新：全案改為 100% 曝險、獲利再投入的複利制）
------------------------------------------------------------
* 權益曲線直接取回測輸出的 equity 欄；計算某一子期間時，將該期間起點
  重新基準化為 CAPITAL，使 IS / OOS / FULL 三欄可互相比較。
* MDD 一律以 60 分 K 逐根權益計算（全專案統一口徑）。
* 年化報酬為幾何年化，年數以日曆天數 / 365.25 計。
* 逐筆交易指標（勝率／盈虧比／獲利因子）以「該筆損益 ÷ 進場時權益」的
  報酬率計算，避免複利下後期交易因金額較大而被過度加權。
* 權益跌破 0 時年化報酬與 Calmar 無實數解，回傳 None。
"""
import numpy as np
import pandas as pd

from src.backtest import split_trades, split_trade_returns
from src import config as C


def equity_curve(out, period=None, capital=None):
    """取出（並重新基準化）某期間的權益曲線。"""
    capital = C.CAPITAL if capital is None else capital
    s = out['equity'] if 'equity' in out else capital + out['net_pnl'].cumsum()
    if period is not None:
        s = s.loc[period[0]:period[1]]
    base = s.iloc[0]
    return s / base * capital if base > 0 else s - base + capital


def metrics(out, period, capital=None):
    capital = C.CAPITAL if capital is None else capital
    a, b = period
    s = out.loc[a:b]
    eq = equity_curve(out, period, capital)
    years = (eq.index[-1] - eq.index[0]).days / 365.25

    total = eq.iloc[-1] / capital - 1
    ann = (eq.iloc[-1] / capital) ** (1 / years) - 1 if eq.iloc[-1] > 0 else None

    dd = (eq - eq.cummax()) / eq.cummax()
    mdd = dd.min()

    bar_ret = eq.pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
    bars_per_year = len(eq) / years
    vol = bar_ret.std() * np.sqrt(bars_per_year)
    sharpe = bar_ret.mean() * bars_per_year / vol if vol > 0 else None
    down = bar_ret[bar_ret < 0]
    dvol = down.std() * np.sqrt(bars_per_year) if len(down) > 1 else None
    sortino = bar_ret.mean() * bars_per_year / dvol if dvol else None

    t = split_trade_returns(s, capital)      # 每筆報酬率（scale-free）
    cash = split_trades(s)                   # 每筆損益（元）
    win, loss = t[t > 0], t[t < 0]
    ruin = eq[eq <= 0]

    return {
        '累積報酬': total,
        '年化報酬': ann,
        '年化波動度': vol,
        'MDD': mdd,
        '年化夏普': sharpe,
        'Calmar': (ann / abs(mdd)) if (ann is not None and mdd < 0) else None,
        'Sortino': sortino,
        '勝率': len(win) / len(t) if len(t) else None,
        '盈虧比': (win.mean() / -loss.mean()) if len(win) and len(loss) else None,
        '期望值': cash.mean() if len(cash) else None,
        '獲利因子': (win.sum() / -loss.sum()) if len(loss) else None,
        '交易次數': len(t),
        '破產日': str(ruin.index[0])[:10] if len(ruin) else None,
    }


def to_trade_date(series):
    """把 60 分 K 序列聚合成交易日序列（夜盤歸屬次一交易日）。"""
    i = series.index
    td = pd.Index(np.where(i.hour >= 15,
                           (i + pd.Timedelta(days=1)).normalize(),
                           i.normalize()))
    return series.groupby(td).last()

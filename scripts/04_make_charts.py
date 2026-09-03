"""產生簡報使用的全部圖表，輸出至 charts/。"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from src.backtest import load_bars, load_ref, run
from src.report import to_trade_date
from src import config as C

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
BLUE, AMBER, RED, GREY = '#2E75B6', '#D98C00', '#A63446', '#9AA5B1'
OOSD = pd.Timestamp(C.OOS[0])
bars, ref = load_bars(), load_ref()
A, B = C.FULL
E = {d: run(bars, ref, direction=d).loc[A:B, 'equity']
     for d in ('both', 'long', 'short')}
dd = lambda x: (x / x.cummax() - 1) * 100

# ── fig07 多空/多單/空單 回撤（60 分 K 逐根，標樣本內外最深點）───────
d1, d2, d3 = dd(E['both']), dd(E['long']), dd(E['short'])
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)
for a in ax:
    a.axvline(OOSD, color='#888', ls='--', lw=1)
    a.axvspan(pd.Timestamp(A), OOSD, color=BLUE, alpha=.045)
ax[0].plot(d1.index, d1, color=BLUE, lw=.7, label='多空皆做')
ax[0].plot(d2.index, d2, color=AMBER, lw=.7, label='純多單')
for a, ser, col in [(ax[0], d1, BLUE), (ax[0], d2, AMBER), (ax[1], d3, RED)]:
    i_, o_ = ser.loc[:C.IS[1]], ser.loc[C.OOS[0]:]
    a.plot([i_.idxmin()], [i_.min()], 'o', ms=6, color=col)
    a.annotate('樣本內 MDD %.2f%%\n%s' % (i_.min(), str(i_.idxmin())[:7]),
               xy=(i_.idxmin(), i_.min()), xytext=(12, -24),
               textcoords='offset points', color=col, fontsize=8.5, fontweight='bold')
    a.plot([o_.idxmin()], [o_.min()], 's', ms=5, mfc='none', mec=col, mew=1.6)
    a.annotate('樣本外 MDD %.2f%%' % o_.min(), xy=(o_.idxmin(), o_.min()),
               xytext=(-30, -20), textcoords='offset points', color=col, fontsize=8.5)
ax[0].set_title('多空皆做  vs  純多單', loc='left', fontsize=11)
ax[0].legend(loc='lower right', fontsize=9, ncol=2)
ax[1].plot(d3.index, d3, color=RED, lw=.7)
ax[1].fill_between(d3.index, d3, 0, color=RED, alpha=.10)
ax[1].set_title('純空單', loc='left', fontsize=11)
lo = min(d1.min(), d2.min(), d3.min()) * 1.22
for a in ax:
    a.set_ylim(lo, 1.4); a.set_ylabel('回撤 (%)'); a.grid(alpha=.25)
plt.tight_layout(); plt.savefig('charts/fig07_drawdown.png', dpi=150, bbox_inches='tight')

# ── fig12 權益曲線 + 回撤（vs 同曝險 B&H）────────────────────────
idx = E['long'].index
p = ref['close_ratio'].reindex(idx).ffill()
BH = pd.Series(C.CAPITAL * np.cumprod(1 + p.pct_change().fillna(0).values * C.EXPOSURE_RATIO), index=idx)
fig, ax = plt.subplots(2, 1, figsize=(11.5, 6.6), sharex=True,
                       gridspec_kw={'height_ratios': [1.45, 1]})
ax[0].plot(idx, E['long'], color=RED, lw=1.3, label='純多單策略（採用版本）')
ax[0].plot(idx, BH, color=GREY, lw=1.3, label='Buy & Hold 1.0倍名目（同曝險）')
ax[0].plot(idx, E['both'], color=BLUE, lw=.8, alpha=.75, label='多空皆做（對照）')
ax[0].axvline(OOSD, color='#888', ls='--', lw=1)
ax[0].set_ylabel('權益 (TWD)'); ax[0].legend(fontsize=9, loc='upper left'); ax[0].grid(alpha=.25)
ax[0].set_ylabel('權益 (TWD，對數刻度)'); ax[0].set_yscale('log')
ax[0].set_title('權益曲線（本金 1,000 萬，100% 曝險複利，60 分 K 逐根，對數刻度）', loc='left', fontsize=11)
dS, dH = dd(E['long']), dd(BH)
ax[1].fill_between(idx, dH, 0, color=GREY, alpha=.35, label='Buy & Hold 1.0倍')
ax[1].plot(idx, dS, color=RED, lw=.9, label='純多單策略')
ax[1].axvline(OOSD, color='#888', ls='--', lw=1)
ax[1].set_ylabel('回撤 (%)'); ax[1].legend(fontsize=9, loc='lower left'); ax[1].grid(alpha=.25)
ax[1].set_title('60 分 K 逐根回撤對照', loc='left', fontsize=11)
plt.tight_layout(); plt.savefig('charts/fig12_equity_curve.png', dpi=150, bbox_inches='tight')

# ── fig13 年度績效 ────────────────────────────────────────────
dS_, dH_ = to_trade_date(E['long']), to_trade_date(BH)
yS = dS_.resample('YE').last(); cS = (yS / yS.shift(1).fillna(C.CAPITAL) - 1) * 100
yB = dH_.resample('YE').last(); cB = (yB / yB.shift(1).fillna(C.CAPITAL) - 1) * 100
x = np.arange(len(cS)); w = .38
fig, ax = plt.subplots(figsize=(11.5, 4.6))
for i, v in enumerate(cB.values):
    if v < 0: ax.axvspan(i - .5, i + .5, color=RED, alpha=.07)
ax.bar(x - w/2, cS.values, w, color=RED, label='純多單策略')
ax.bar(x + w/2, cB.values, w, color=GREY, label='Buy & Hold 1.0倍名目（同曝險）')
ax.axhline(0, color='#444', lw=1)
ax.set_xticks(x); ax.set_xticklabels(cS.index.year)
ax.set_ylabel('年度報酬率 (%)'); ax.legend(fontsize=9); ax.grid(axis='y', alpha=.25)
ax.set_title('年度報酬：純多單策略 vs Buy & Hold 1.0倍（淡紅底＝大盤下跌年）',
             loc='left', fontsize=11)
plt.tight_layout(); plt.savefig('charts/fig13_yearly.png', dpi=150, bbox_inches='tight')
print('圖表已輸出至 charts/')

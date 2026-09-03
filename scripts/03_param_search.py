"""參數搜尋：一維掃描 + 二維網格 + 全域掃描。**僅使用樣本內資料**。

選點準則（三關）：
  1. 樣本內 Calmar 最高
  2. 每筆期望值與獲利因子具安全邊際
  3. 對成本假設不過度敏感（成本 ×3 時的 Calmar 衰減）
  4. 樣本內大盤下跌年維持防禦性
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src.backtest import load_bars, load_ref, run
from src.report import metrics
from src import config as C

bars, ref = load_bars(), load_ref()
P1 = ('2011-01-01', '2015-06-30')      # 樣本內前半
P2 = ('2015-07-01', '2019-12-31')      # 樣本內後半

# ── 一維掃描 ────────────────────────────────────────────────
rows = []
for c in np.round(np.arange(0.0020, 0.01201, 0.0002), 6):
    bt = run(bars, ref, c, c, direction='long')
    rows.append((c, metrics(bt, C.IS)['Calmar'],
                    metrics(bt, P1)['Calmar'], metrics(bt, P2)['Calmar']))
d1 = pd.DataFrame(rows, columns=['af', 'IS', 'P1', 'P2'])
os.makedirs('results', exist_ok=True)
d1.to_csv('results/param_sweep_1d.csv', index=False)
best = d1.loc[d1.IS.idxmax()]
print(f'一維最佳：AF={best.af:.4f}  樣本內 Calmar={best.IS:.3f}')

# ── 二維網格（起始值與上限分開搜）─────────────────────────────
starts = [0.0024, 0.0028, 0.0032, 0.0036, 0.0042, 0.0046, 0.0050, 0.0056, 0.0062]
maxs   = [0.0032, 0.0036, 0.0042, 0.0046, 0.0050, 0.0056, 0.0064]
rows = [(s, m, metrics(run(bars, ref, s, m, direction='long'), C.IS)['Calmar'])
        for s in starts for m in maxs if m >= s]
pd.DataFrame(rows, columns=['af_start', 'af_max', 'IS_calmar']).to_csv(
    'results/param_grid_2d.csv', index=False)

# ── 全域掃描（涵蓋 Wilder 0.02/0.2）───────────────────────────
S = [C.AF, 0.008, 0.014, 0.02, 0.035, 0.06]
M = [C.AF, 0.01, 0.02, 0.05, 0.1, 0.2]
for d in ('both', 'long'):
    rows = [(s, m, metrics(run(bars, ref, s, m, direction=d), C.IS)['年化夏普'])
            for s in S for m in M if m >= s]
    pd.DataFrame(rows, columns=['af_start', 'af_max', 'IS_sharpe']).to_csv(
        f'results/param_grid_wide_{d}.csv', index=False)

# ── 成本敏感度（第二關）──────────────────────────────────────
print('\n成本敏感度（樣本內 Calmar）')
print('%-8s %8s %8s %8s %8s' % ('AF', '600', '1200', '1800', '衰減'))
for c in [C.AF, 0.0100]:
    v = [metrics(run(bars, ref, c, c, direction='long', fee=f), C.IS)['Calmar']
         for f in (300., 600., 900.)]
    print('%-8.4f %8.3f %8.3f %8.3f %7.0f%%' % (c, *v, (1 - v[2] / v[0]) * 100))

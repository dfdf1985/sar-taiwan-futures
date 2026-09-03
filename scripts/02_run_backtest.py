"""跑出簡報所有表格的數字，輸出 results/backtest_summary.json。"""
import json, sys, os
sys.path.insert(0, os.getcwd())
from src.backtest import load_bars, load_ref, run
from src.report import metrics
from src import config as C

bars, ref = load_bars(), load_ref()
PERIODS = {'IS': C.IS, 'OOS': C.OOS, 'FULL': C.FULL}
PARAMS = {'w': (0.02, 0.2), 'o': (C.AF, C.AF)}   # w = Wilder 原始設定, o = 最佳化

out = {}
for tag, (af, afm) in PARAMS.items():
    for d in ('both', 'long', 'short'):
        bt = run(bars, ref, af, afm, direction=d)
        for pk, per in PERIODS.items():
            out[f'{tag}_{d}_{pk}'] = metrics(bt, per)
        if d != 'short':                                  # 無夜盤對照
            nn = run(bars, ref, af, afm, direction=d, day_only=True)
            for pk in ('IS', 'FULL'):
                out[f'{tag}_{d}_nn_{pk}'] = metrics(nn, PERIODS[pk])
        lot = run(bars, ref, af, afm, direction=d, one_lot=True)   # 固定 1 口對照
        out[f'lot_{tag}_{d}'] = metrics(lot, C.FULL)

os.makedirs('results', exist_ok=True)
json.dump(out, open('results/backtest_summary.json', 'w'),
          ensure_ascii=False, indent=1, default=lambda x: None)

k = out['o_long_FULL']
print('最終策略（純多單・AF=%.4f）全樣本：' % C.AF)
print('  累積 %.2f%%  年化 %.2f%%  MDD %.2f%%  夏普 %.2f  Calmar %.2f  獲利因子 %.2f  交易 %d 筆'
      % (k['累積報酬']*100, k['年化報酬']*100, k['MDD']*100,
         k['年化夏普'], k['Calmar'], k['獲利因子'], k['交易次數']))

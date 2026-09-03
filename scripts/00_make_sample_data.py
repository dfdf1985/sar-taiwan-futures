"""產生**合成**的示範資料，讓不具原始資料授權的人也能把流程跑完。

⚠️ 這不是市場資料。它是一段帶有正向漂移的隨機漫步，只用來驗證程式可執行。
   跑出來的績效數字沒有任何研究意義，請勿與簡報中的結果對照。

用法:  python scripts/00_make_sample_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from src import config as C

RNG = np.random.default_rng(20260903)
START, END = C.FULL

# 交易時段：日盤 08–13 時，夜盤 15–23 時＋次日 00–04 時（與真實資料同構）
HOURS = list(range(8, 14)) + list(range(15, 24)) + list(range(0, 5))

idx = pd.date_range(START, END, freq='h')
idx = idx[[h in HOURS for h in idx.hour]]
idx = idx[idx.dayofweek < 5]

n = len(idx)
drift, vol = 0.00004, 0.0022
ret = RNG.normal(drift, vol, n)
close = 8000.0 * np.exp(np.cumsum(ret))

spread = np.abs(RNG.normal(0, vol * 0.8, n)) * close
open_ = np.concatenate([[close[0]], close[:-1]])
high = np.maximum(open_, close) + spread * RNG.random(n)
low = np.minimum(open_, close) - spread * RNG.random(n)

bars = pd.DataFrame({'Open': open_, 'High': high, 'Low': low, 'Close': close,
                     'Volume': RNG.integers(500, 5000, n)}, index=idx)
bars.index.name = 'datetime'

os.makedirs('data', exist_ok=True)
bars.to_csv(C.DATA_FILE)

# close_none：未調整價（此處等同 close）；close_ratio：比例調整價（B&H 基準）
ref = pd.DataFrame({'close_ratio': close, 'close_none': close}, index=idx)
ref.index.name = 'datetime'
ref.to_csv(C.BH_REF_FILE)

print(f'⚠️ 已產生「合成」示範資料（非真實市場資料）：{n:,} 根 60 分 K')
print(f'   {C.DATA_FILE}')
print(f'   {C.BH_REF_FILE}')
print('   跑出來的績效數字不具研究意義，請勿與簡報結果對照。')

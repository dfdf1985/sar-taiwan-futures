"""從原始 parquet 建立回測輸入檔。

原始資料未隨本專案發布（含商業資料商授權內容），請先依 DATA.md 取得
`MTX_continuous_1m_post.parquet`，再執行本腳本。

用法:  python scripts/01_build_data.py /path/to/MTX_continuous_1m_post.parquet
"""
import sys, hashlib
import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else 'MTX_continuous_1m_post.parquet'
df = pd.read_parquet(src, columns=['ts', 'open_diff', 'high_diff', 'low_diff',
                                   'close_diff', 'close_ratio', 'close_none', 'volume'])
df = df.set_index('ts').sort_index()

bars = df.resample('60min').agg(
    Open=('open_diff', 'first'), High=('high_diff', 'max'),
    Low=('low_diff', 'min'), Close=('close_diff', 'last'),
    Volume=('volume', 'sum'),
).dropna(subset=['Close'])
bars.index.name = 'datetime'
bars.to_csv('data/mtx_60min_adj.csv.gz')

ref = df.resample('60min').agg(close_ratio=('close_ratio', 'last'),
                               close_none=('close_none', 'last')).dropna()
ref.index.name = 'datetime'
ref.to_csv('data/mtx_bh_ref.csv.gz')

h = hashlib.sha256(open('data/mtx_60min_adj.csv.gz', 'rb').read()).hexdigest()[:16]
print(f'60 分 K {len(bars):,} 根  {bars.index[0]} ~ {bars.index[-1]}')
print(f'SHA-256(前16碼) = {h}   （DATA.md 記錄值：40306e1e7010c653）')

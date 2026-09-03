import pandas as pd
import numpy as np

def parabolic_sar(df, af_start=0.02, af_step=0.02, af_max=0.2):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)

    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)  # 1 = up, -1 = down
    ep = np.zeros(n)
    af = np.zeros(n)

    # init: use first two bars to determine initial trend
    trend[0] = 1 if close[1] > close[0] else -1
    if trend[0] == 1:
        sar[0] = low[0]
        ep[0] = high[0]
    else:
        sar[0] = high[0]
        ep[0] = low[0]
    af[0] = af_start

    for t in range(1, n):
        prev_sar = sar[t-1]
        prev_ep = ep[t-1]
        prev_af = af[t-1]
        prev_trend = trend[t-1]

        calc_sar = prev_sar + prev_af * (prev_ep - prev_sar)

        if prev_trend == 1:
            # clip to below prior two lows
            lo_clip = min(low[t-1], low[t-2] if t >= 2 else low[t-1])
            calc_sar = min(calc_sar, lo_clip)
        else:
            hi_clip = max(high[t-1], high[t-2] if t >= 2 else high[t-1])
            calc_sar = max(calc_sar, hi_clip)

        reversed_ = False
        if prev_trend == 1:
            if low[t] <= calc_sar:
                reversed_ = True
                trend[t] = -1
                sar[t] = prev_ep
                ep[t] = low[t]
                af[t] = af_start
            else:
                trend[t] = 1
                sar[t] = calc_sar
                if high[t] > prev_ep:
                    ep[t] = high[t]
                    af[t] = min(prev_af + af_step, af_max)
                else:
                    ep[t] = prev_ep
                    af[t] = prev_af
        else:
            if high[t] >= calc_sar:
                reversed_ = True
                trend[t] = 1
                sar[t] = prev_ep
                ep[t] = high[t]
                af[t] = af_start
            else:
                trend[t] = -1
                sar[t] = calc_sar
                if low[t] < prev_ep:
                    ep[t] = low[t]
                    af[t] = min(prev_af + af_step, af_max)
                else:
                    ep[t] = prev_ep
                    af[t] = prev_af

    out = df.copy()
    out['SAR'] = sar
    out['trend'] = trend
    out['ep'] = ep
    out['af'] = af
    return out

if __name__ == '__main__':
    df = pd.read_csv('txf_60min.csv', index_col=0, parse_dates=True)
    res = parabolic_sar(df)
    print(res.tail(20)[['Close','SAR','trend']])
    flips = (res['trend'] != res['trend'].shift(1)).sum()
    print('flips:', flips, 'of', len(res))

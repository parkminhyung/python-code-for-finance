import pandas as pd
import numpy as np
import math

##############################
# 1. 기본 헬퍼 함수들
##############################

def true_range(df):
    """True Range 계산 (Pine의 tr()와 동일)"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def rma(series, period):
    """Wilder의 이동평균 (RMA)"""
    return series.ewm(alpha=1/period, adjust=False).mean()

def sma(series, period):
    """단순 이동평균 (SMA)"""
    return series.rolling(window=period, min_periods=period).mean()

def ema(series, period):
    """지수 이동평균 (EMA)"""
    return series.ewm(span=period, adjust=False).mean()

def wma(series, period):
    """가중 이동평균 (WMA) – 가중치는 1부터 period까지 선형증가"""
    weights = np.arange(1, period+1)
    return series.rolling(period).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)

def lsma(series, period):
    """선형 회귀 이동평균 (LSMA)"""
    def linreg(x):
        n = len(x)
        if n == 0:
            return np.nan
        t = np.arange(n)
        slope, intercept = np.polyfit(t, x, 1)
        return intercept + slope*(n-1)
    return series.rolling(window=period, min_periods=period).apply(linreg, raw=True)

def tma(series, period):
    """이중 SMA: TMA = SMA(SMA(src, ceil(period/2)), floor(period/2)+1)"""
    half1 = int(np.ceil(period/2))
    half2 = int(np.floor(period/2)) + 1
    return sma(sma(series, half1), half2)

def dema(series, period):
    """Double EMA (DEMA)"""
    e = ema(series, period)
    return 2 * e - ema(e, period)

def tema(series, period):
    """Triple EMA (TEMA)"""
    e1 = ema(series, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3

def hma(series, period):
    """Hull Moving Average (HMA)"""
    half_period = int(period/2)
    sqrt_period = int(round(np.sqrt(period)))
    return wma(2 * wma(series, half_period) - wma(series, period), sqrt_period)

def vama(series, period, volatility_lookback):
    """Volatility Adjusted MA (VAMA)"""
    mid = ema(series, period)
    dev = series - mid
    vol_up = dev.rolling(window=volatility_lookback, min_periods=volatility_lookback).max()
    vol_down = dev.rolling(window=volatility_lookback, min_periods=volatility_lookback).min()
    return mid + (vol_up + vol_down) / 2

def jma(series, period, jurik_phase=3, jurik_power=1):
    """Jurik Moving Average (JMA) – 반복문으로 계산"""
    if jurik_phase < -100:
        phaseRatio = 0.5
    elif jurik_phase > 100:
        phaseRatio = 2.5
    else:
        phaseRatio = 1.5 + jurik_phase / 100.0

    beta_val = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta_val ** jurik_power

    jma_values = np.zeros(len(series))
    e0 = np.zeros(len(series))
    e1 = np.zeros(len(series))
    e2 = np.zeros(len(series))
    
    for i in range(len(series)):
        x = series.iloc[i]
        if i == 0:
            e0[i] = x
            e1[i] = 0
            e2[i] = 0
            jma_values[i] = x
        else:
            e0[i] = (1 - alpha) * x + alpha * e0[i-1]
            e1[i] = (x - e0[i]) * (1 - beta_val) + beta_val * e1[i-1]
            e2[i] = (e0[i] + phaseRatio * e1[i] - jma_values[i-1]) * ((1 - alpha) ** 2) + (alpha ** 2) * e2[i-1]
            jma_values[i] = jma_values[i-1] + e2[i]
    return pd.Series(jma_values, index=series.index)

def kijun_v2(series, period, kidiv=1):
    """Kijun v2 – Baseline from (최저, 최고)의 평균과 환산선의 평균"""
    period_div = int(period / kidiv)
    lowest_val = series.rolling(window=period, min_periods=period).min()
    highest_val = series.rolling(window=period, min_periods=period).max()
    kijun = (lowest_val + highest_val) / 2
    lowest_conv = series.rolling(window=period_div, min_periods=period_div).min()
    highest_conv = series.rolling(window=period_div, min_periods=period_div).max()
    conversion = (lowest_conv + highest_conv) / 2
    return (kijun + conversion) / 2

def mcginley(series, period):
    """McGinley Dynamic – 재귀적으로 계산"""
    mg = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            mg[i] = series.iloc[i]
        else:
            if mg[i-1] == 0:
                mg[i] = series.iloc[i]
            else:
                mg[i] = mg[i-1] + (series.iloc[i] - mg[i-1]) / (period * (series.iloc[i] / mg[i-1])**4)
    return pd.Series(mg, index=series.index)

# EDSMA 관련 함수들
def get2PoleSSF(series, ssfLength):
    PI = 2 * np.arcsin(1)
    arg = np.sqrt(2) * PI / ssfLength
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(arg)
    c2 = b1
    c3 = -a1 ** 2
    c1 = 1 - c2 - c3
    ssf = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            ssf[i] = c1 * series.iloc[i]
        elif i == 1:
            ssf[i] = c1 * series.iloc[i] + c2 * ssf[i-1]
        else:
            ssf[i] = c1 * series.iloc[i] + c2 * ssf[i-1] + c3 * ssf[i-2]
    return pd.Series(ssf, index=series.index)

def get3PoleSSF(series, ssfLength):
    PI = 2 * np.arcsin(1)
    arg = PI / ssfLength
    a1 = np.exp(-arg)
    b1 = 2 * a1 * np.cos(1.738 * arg)
    c1 = a1 ** 2
    coef2 = b1 + c1
    coef3 = -(c1 + b1 * c1)
    coef4 = c1 ** 2
    coef1 = 1 - coef2 - coef3 - coef4
    ssf = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            ssf[i] = coef1 * series.iloc[i]
        elif i == 1:
            ssf[i] = coef1 * series.iloc[i] + coef2 * ssf[i-1]
        elif i == 2:
            ssf[i] = coef1 * series.iloc[i] + coef2 * ssf[i-1] + coef3 * ssf[i-2]
        else:
            ssf[i] = coef1 * series.iloc[i] + coef2 * ssf[i-1] + coef3 * ssf[i-2] + coef4 * ssf[i-3]
    return pd.Series(ssf, index=series.index)

def edsma(series, period, ssfLength, ssfPoles=2):
    """EDSMA – Ehlers’ Super Smoother 기반 필터"""
    zeros = series - series.shift(2)
    avgZeros = (zeros + zeros.shift(1)) / 2
    avgZeros = avgZeros.fillna(0)
    if ssfPoles == 2:
        ssf = get2PoleSSF(avgZeros, ssfLength)
    else:
        ssf = get3PoleSSF(avgZeros, ssfLength)
    stdev_ssf = ssf.rolling(window=period, min_periods=period).std()
    scaledFilter = ssf / stdev_ssf.replace(0, np.nan)
    scaledFilter = scaledFilter.fillna(0)
    alpha_val = 5 * scaledFilter.abs() / period
    edsma_values = np.zeros(len(series))
    for i in range(len(series)):
        if i == 0:
            edsma_values[i] = series.iloc[i]
        else:
            edsma_values[i] = alpha_val.iloc[i] * series.iloc[i] + (1 - alpha_val.iloc[i]) * edsma_values[i-1]
    return pd.Series(edsma_values, index=series.index)

##############################
# 2. 이동평균 선택 함수
##############################

def ma(ma_type, series, period, **kwargs):
    t = ma_type.upper()
    if t == "TMA":
        return tma(series, period)
    elif t == "MF":
        return ema(series, period)  # 여기서는 간단히 EMA로 대체
    elif t == "LSMA":
        return lsma(series, period)
    elif t == "SMA":
        return sma(series, period)
    elif t == "EMA":
        return ema(series, period)
    elif t == "DEMA":
        return dema(series, period)
    elif t == "TEMA":
        return tema(series, period)
    elif t == "WMA":
        return wma(series, period)
    elif t == "VAMA":
        volatility_lookback = kwargs.get("volatility_lookback", 10)
        return vama(series, period, volatility_lookback)
    elif t == "HMA":
        return hma(series, period)
    elif t == "JMA":
        jurik_phase = kwargs.get("jurik_phase", 3)
        jurik_power = kwargs.get("jurik_power", 1)
        return jma(series, period, jurik_phase, jurik_power)
    elif t == "KIJUN V2":
        kidiv = kwargs.get("kidiv", 1)
        return kijun_v2(series, period, kidiv)
    elif t == "MCGINLEY":
        return mcginley(series, period)
    elif t == "EDSMA":
        ssfLength = kwargs.get("ssfLength", 20)
        ssfPoles = kwargs.get("ssfPoles", 2)
        return edsma(series, period, ssfLength, ssfPoles)
    else:
        return ema(series, period)

##############################
# 3. ATR 스무딩용 함수
##############################

def ma_function(series, period, smoothing="WMA"):
    s = smoothing.upper()
    if s == "RMA":
        return rma(series, period)
    elif s == "SMA":
        return sma(series, period)
    elif s == "EMA":
        return ema(series, period)
    else:
        return wma(series, period)

##############################
# 4. SSL Hybrid 계산 함수 (파라미터 조정 가능)
##############################

def ssl_hybrid(df, **kwargs):
    """
    df: 'open', 'high', 'low', 'close' 컬럼이 포함된 DataFrame
    **kwargs: 예를 들어 atrlen=14, smoothing="EMA", len1=100 등의 키워드 인자를 전달할 수 있습니다.
    
    반환: 계산된 여러 컬럼이 추가된 DataFrame
    """
    # 기본 파라미터 설정
    default_params = {
        "show_Baseline": True,
        "show_SSL1": False,
        "show_atr": True,
        "atrlen": 14,
        "mult": 1.0,
        "smoothing": "WMA",
        "maType": "HMA",       # SSL1/Baseline용
        "len1": 60,
        "SSL2Type": "JMA",     # SSL2용
        "len2": 5,
        "SSL3Type": "HMA",     # EXIT용
        "len3": 15,
        "src": "close",
        "kidiv": 1,
        "jurik_phase": 3,
        "jurik_power": 1,
        "volatility_lookback": 10,
        # MF 파라미터
        "beta": 0.8,
        "feedback": False,
        "z": 0.5,
        # EDSMA 파라미터
        "ssfLength": 20,
        "ssfPoles": 2,
        # Baseline Channel 파라미터
        "useTrueRange": True,
        "multy": 0.2,
        # SSL2 Continuation ATR
        "atr_crit": 0.9
    }
    # kwargs로 전달된 값으로 기본 파라미터 덮어쓰기
    default_params.update(kwargs)
    params = default_params

    # 4.1 ATR 및 ATR 밴드 계산
    tr_series = true_range(df)
    atr_slen = ma_function(tr_series, params["atrlen"], params["smoothing"])
    df['ATR_slen'] = atr_slen
    df['upper_band'] = atr_slen * params["mult"] + df['close']
    df['lower_band'] = df['close'] - atr_slen * params["mult"]
    
    # 4.2 SSL1 (Baseline) 계산 – maType / len1 사용
    emaHigh = ma(params["maType"], df['high'], params["len1"],
                 jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                 kidiv=params["kidiv"], ssfLength=params["ssfLength"],
                 ssfPoles=params["ssfPoles"], volatility_lookback=params["volatility_lookback"])
    emaLow = ma(params["maType"], df['low'], params["len1"],
                jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                kidiv=params["kidiv"], ssfLength=params["ssfLength"],
                ssfPoles=params["ssfPoles"], volatility_lookback=params["volatility_lookback"])
    
    # Hlv: close와 emaHigh/emaLow 비교 (재귀적 계산)
    Hlv = np.zeros(len(df))
    for i in range(len(df)):
        c = df['close'].iloc[i]
        if i == 0:
            Hlv[i] = 1 if c > emaHigh.iloc[i] else (-1 if c < emaLow.iloc[i] else 0)
        else:
            if c > emaHigh.iloc[i]:
                Hlv[i] = 1
            elif c < emaLow.iloc[i]:
                Hlv[i] = -1
            else:
                Hlv[i] = Hlv[i-1]
    sslDown = pd.Series(np.where(Hlv < 0, emaHigh, emaLow), index=df.index)
    df['sslDown'] = sslDown

    # 4.3 SSL2 계산 – SSL2Type / len2 사용
    maHigh = ma(params["SSL2Type"], df['high'], params["len2"],
                jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                ssfLength=params["ssfLength"], ssfPoles=params["ssfPoles"],
                volatility_lookback=params["volatility_lookback"])
    maLow = ma(params["SSL2Type"], df['low'], params["len2"],
               jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
               ssfLength=params["ssfLength"], ssfPoles=params["ssfPoles"],
               volatility_lookback=params["volatility_lookback"])
    Hlv2 = np.zeros(len(df))
    for i in range(len(df)):
        c = df['close'].iloc[i]
        if i == 0:
            Hlv2[i] = 1 if c > maHigh.iloc[i] else (-1 if c < maLow.iloc[i] else 0)
        else:
            if c > maHigh.iloc[i]:
                Hlv2[i] = 1
            elif c < maLow.iloc[i]:
                Hlv2[i] = -1
            else:
                Hlv2[i] = Hlv2[i-1]
    sslDown2 = pd.Series(np.where(Hlv2 < 0, maHigh, maLow), index=df.index)
    df['sslDown2'] = sslDown2

    # 4.4 EXIT (SSL3) 계산 – SSL3Type / len3 사용
    ExitHigh = ma(params["SSL3Type"], df['high'], params["len3"],
                  jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                  ssfLength=params["ssfLength"], ssfPoles=params["ssfPoles"],
                  volatility_lookback=params["volatility_lookback"])
    ExitLow = ma(params["SSL3Type"], df['low'], params["len3"],
                 jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                 ssfLength=params["ssfLength"], ssfPoles=params["ssfPoles"],
                 volatility_lookback=params["volatility_lookback"])
    Hlv3 = np.zeros(len(df))
    for i in range(len(df)):
        c = df['close'].iloc[i]
        if i == 0:
            Hlv3[i] = 1 if c > ExitHigh.iloc[i] else (-1 if c < ExitLow.iloc[i] else 0)
        else:
            if c > ExitHigh.iloc[i]:
                Hlv3[i] = 1
            elif c < ExitLow.iloc[i]:
                Hlv3[i] = -1
            else:
                Hlv3[i] = Hlv3[i-1]
    sslExit = pd.Series(np.where(Hlv3 < 0, ExitHigh, ExitLow), index=df.index)
    df['sslExit'] = sslExit

    # 4.5 Keltner Baseline Channel 계산
    BBMC = ma(params["maType"], df['close'], params["len1"],
              jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
              kidiv=params["kidiv"], ssfLength=params["ssfLength"],
              ssfPoles=params["ssfPoles"], volatility_lookback=params["volatility_lookback"])
    Keltma = ma(params["maType"], df['close'], params["len1"],
                jurik_phase=params["jurik_phase"], jurik_power=params["jurik_power"],
                kidiv=params["kidiv"], ssfLength=params["ssfLength"],
                ssfPoles=params["ssfPoles"], volatility_lookback=params["volatility_lookback"])
    if params["useTrueRange"]:
        range_series = true_range(df)
    else:
        range_series = df['high'] - df['low']
    rangema = ema(range_series, params["len1"])
    upperk = Keltma + rangema * params["multy"]
    lowerk = Keltma - rangema * params["multy"]
    df['BBMC'] = BBMC
    df['upperk'] = upperk
    df['lowerk'] = lowerk

    # 4.6 Baseline Violation Candle 계산
    diff = (df['close'] - df['open']).abs()
    atr_violation = diff > atr_slen
    inRange = (upperk > BBMC) & (lowerk < BBMC)
    candlesize_violation = atr_violation & inRange
    df['candlesize_violation'] = candlesize_violation

    # 4.7 EXIT 신호 (cross over sslExit)
    base_cross_Long = (df['close'] > sslExit) & (df['close'].shift(1) <= sslExit.shift(1))
    base_cross_Short = (sslExit > df['close']) & (sslExit.shift(1) <= df['close'].shift(1))
    codiff = pd.Series(np.nan, index=df.index)
    codiff[base_cross_Long] = 1
    codiff[base_cross_Short] = -1
    df['codiff'] = codiff

    # 4.8 SSL2 Continuation – ATR 조건
    atr_crit = params["atr_crit"]
    upper_half = atr_slen * atr_crit + df['close']
    lower_half = df['close'] - atr_slen * atr_crit
    buy_inatr = lower_half < sslDown2
    sell_inatr = upper_half > sslDown2
    sell_cont = (df['close'] < BBMC) & (df['close'] < sslDown2)
    buy_cont = (df['close'] > BBMC) & (df['close'] > sslDown2)
    df['sell_atr'] = sell_inatr & sell_cont
    df['buy_atr'] = buy_inatr & buy_cont

    return df

##############################
# 5. 예제: 사용법
##############################

start_date = "2024-01-01"
end_date = datetime.datetime.today()

data = yf.download("AAPL",start =start_date,end = end_date,progress = False)

data.columns = data.columns.str.lower()
del data['close']

data = data.rename(columns = {"adj close":"close"})

result = ssl_hybrid(data, atrlen=20, smoothing="EMA", len1=100)



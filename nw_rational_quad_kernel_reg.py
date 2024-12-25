
def nw_rational_quad_kernel_reg(src: pd.Series, h: int, r: float, x_0: int):
    """
    Nadaraya - Watson Rational Quadratic Kernel Regression(Non-Repainting)
    Parameters:
      src : pd.Series - Source data (e.g., adjusted close prices)
      h : int - Lookback window
      r : float - Relative weighting
      x_0 : int - Start regression at bar
      lag : Lag for crossover detection
      size : the lenght of src(close value)
    
   made by jdehorty 
   TradingView link:
   https://www.tradingview.com/script/AWNvbPRM-Nadaraya-Watson-Rational-Quadratic-Kernel-Non-Repainting/
  
    """
    current_weight = [0.0]*len(src)
    cumulative_weight = 0.0
    
    for i in range(x_0 + 2):
        y = src.shift(i, fill_value=0.0)
        w = (1 + (i ** 2 / (h ** 2 * 2 * r))) ** -r
        current_weight += y.values * w
        cumulative_weight += w
        
    yhat_value = current_weight / cumulative_weight
    yhat_value[:x_0 + 1] = 0.0

    return yhat_value


#Example 
import yfinance as yf
data = yf.download("AAPL","2023-01-01","2024-12-20")

data['yhat1'] = nw_rational_quad_kernel_reg(data['Adj Close'], h=8, r=8, x_0=25)[:,-1]
data.tail(5)

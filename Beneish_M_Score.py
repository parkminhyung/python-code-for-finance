

# Beneish M Score formula = −4.84 + 0.92×DSRI + 0.528×GMI + 0.404×AQI + 0.892×SGI + 0.115×DEPI − 0.172×SGAI + 4.679×TATA − 0.327×LVGI
# The M-score was developed by Professor Messod Beneish. 
# Unlike the Altman Z-score, which assesses bankruptcy risk, or the Piotroski F-score, which evaluates business trends, the M-score is specifically designed to detect the risk of earnings manipulation. 
# This information is based on the original research paper on the M-score.
## Beneish M-Score of equal or less than -1.78 suggests that the company is unlikely to be a manipulator. Score of greater than -1.78 signals that the company is likely to be a manipulator.

import yfinance as yf

def Beneish_M_score(ticker):

    underlying = yf.Ticker(ticker)
    bs = underlying.balance_sheet
    fin = underlying.financials
    cf = underlying.cashflow
    
    ##DSRI = [Receivables(t)/Revenue(t)] / [Receivables(t-1) / Revenue(t-1)]  

    if "Receivables" in bs.index:
        Total_RCV_t = bs.at["Receivables",bs.columns[0]]
        Total_RCV_t_1 = bs.at["Receivables",bs.columns[1]]
    else :
        Total_RCV_t = bs.at["Accounts Receivable",bs.columns[0]] + bs.at["Other Receivables",bs.columns[0]] + bs.at["Gross Accounts Receivable",bs.columns[0]] - bs.at["Allowance For Doubtful Accounts Receivable",bs.columns[0]]
        Total_RCV_t_1 = bs.at["Accounts Receivable",bs.columns[1]] + bs.at["Other Receivables",bs.columns[1]] + bs.at["Gross Accounts Receivable",bs.columns[1]] - bs.at["Allowance For Doubtful Accounts Receivable",bs.columns[1]]
        

    RR_t = (Total_RCV_t/fin.at["Total Revenue",fin.columns[0]])
    RR_t_1 = (Total_RCV_t_1/fin.at["Total Revenue",fin.columns[1]])
    DSRI = RR_t/RR_t_1 
    DSRI

    ##GMI = Revenue ratio(t-1) / Revenue Ratio(t)

    GMI = (fin.at["Gross Profit",fin.columns[1]]/fin.at["Total Revenue",fin.columns[1]])/(fin.at["Gross Profit",fin.columns[0]]/fin.at["Total Revenue",fin.columns[0]])
    GMI

    ## AQI = module(t) = Current Assets + NPE + Intanglble(t), (1-(module(t))/Total Asset)/()

    CNPEI_Asset_t = 1-(bs.at["Current Assets",bs.columns[0]] + bs.at["Net PPE",bs.columns[0]])/bs.at["Total Assets",bs.columns[0]]
    CNPEI_Asset_t_1 = 1-(bs.at["Current Assets",bs.columns[1]] + bs.at["Net PPE",bs.columns[1]])/bs.at["Total Assets",bs.columns[1]]

    AQI = (CNPEI_Asset_t)/(CNPEI_Asset_t_1)
    AQI

    ## SGI (Rev t )/(Rev t-1)
    SGI = fin.at["Total Revenue",fin.columns[0]]/fin.at["Total Revenue",fin.columns[1]]
    SGI

    ## DEPI (DDA / Dep+ CA)t-1 / (DDA / Dep+ CA)t 
    DDACA_t_1 = cf.at["Depreciation And Amortization",cf.columns[1]]/(cf.at["Depreciation And Amortization",cf.columns[1]]+ bs.at["Net PPE",bs.columns[1]])
    DDACA_t = cf.at["Depreciation And Amortization",cf.columns[0]]/(cf.at["Depreciation And Amortization",cf.columns[0]]+ bs.at["Net PPE",bs.columns[0]])
    DEPI = DDACA_t_1/DDACA_t
    DEPI

    ## SGAI = (SG & A / sales (t))/(SG & A / sales (t-1))
    SGAREV_t = fin.at["Selling General And Administration",fin.columns[0]]/fin.at["Total Revenue",fin.columns[0]]
    SGAREV_t_1 = fin.at["Selling General And Administration",fin.columns[1]]/fin.at["Total Revenue",fin.columns[1]]
    SGAI  = SGAREV_t/SGAREV_t_1
    SGAI


    ## TATA = Net Income - CFop / TA

    if "Operating Cash Flow" in cf.index:
        OCF = cf.at["Operating Cash Flow",cf.columns[0]]
    else :
        OCF = cf.at["Cash Flow From Continuing Operating Activities",cf.columns[0]]

    TATA = (cf.at["Net Income From Continuing Operations",cf.columns[0]] - OCF)/bs.at["Total Assets",bs.columns[0]]
    TATA

    ## LVGI (Long term debt + curretn lia / total asset)t / ...t-1
    LTDCTA_t = (bs.at["Long Term Debt",bs.columns[0]] + bs.at["Current Liabilities",bs.columns[0]])/bs.at["Total Assets",bs.columns[0]]
    LTDCTA_t_1 = (bs.at["Long Term Debt",bs.columns[1]] + bs.at["Current Liabilities",bs.columns[1]])/bs.at["Total Assets",bs.columns[1]]
    LVGI = LTDCTA_t/LTDCTA_t_1
    LVGI

    BM_score = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.89 *SGI + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    BM_score = round(BM_score,5)
    
    fisical_yr = bs.columns[0].strftime('%Y-%m-%d')
    if BM_score <= -1.78 :
        print(ticker + ": This company is unlikely to be a mainpulator","\n","Latest fiscal year: " + fisical_yr,"\n","Beneish M score:",BM_score,"\n")
    else :
        print(ticker + ": This company is likely to be a mainpulator","\n","Latest fiscal year: " + fisical_yr,"\n","Beneish M score:",BM_score,"\n")

    return BM_score, fisical_yr


## Example

ticker = "TSLA"
Beneish_M_score(ticker)

## result : 
# TSLA: This company is likely to be a mainpulator 
# Latest fiscal year: 2023-12-31 
# Beneish M score: -1.62041 



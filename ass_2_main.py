# -*- coding: utf-8 -*-
"""
EQRM II
Assignment 1 main

Created on Mon Nov 29 16:00:04 2021

@author: Donald Hagestein, Connor Stevens and Maurits van den Oever
#tryout
"""


# import pachages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc


# define needed functions below
# structure it so every question is its own function, that prints all of its output...
# first the loadin_data(path) function


###########################################################
### loadin_data
def loadin_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    # the test tickers
    df_aapl = df[df['TICKER']=='AAPL'][['date', 'RET']].set_index('date').rename(columns={'RET':'AAPL_ret'})
    df_msft = df[df['TICKER']=='MSFT'][['date', 'RET']].set_index('date').rename(columns={'RET':'MSFT_ret'})
    df_csco = df[df['TICKER']=='CSCO'][['date', 'RET']].set_index('date').rename(columns={'RET':'CSCO_ret'})
    
    # the real tickers
    df_mrk = df[df['TICKER']=='MRK'][['date', 'RET']].set_index('date').rename(columns={'RET':'MRK_ret'})
    df_amzn = df[df['TICKER']=='AMZN'][['date', 'RET']].set_index('date').rename(columns={'RET':'AMZN_ret'})
    df_pep = df[df['TICKER']=='PEP'][['date', 'RET']].set_index('date').rename(columns={'RET':'PEP_ret'})

    df_test = pd.merge(pd.merge(df_aapl,df_msft,on='date'),df_csco,on='date')
    df_real = pd.merge(pd.merge(df_mrk,df_amzn,on='date'),df_pep,on='date')

    return df_test, df_real

###########################################################
### output_Q1
def output_Q1(df):
    """
    function that handles all the output for Question 1

    Parameters
    ----------
    df : can be df_test or df_real, so it can be used to check code and generate output for Q1

    Returns
    -------
    None.

    """
    print('Question 1: ')
    print('')

    series = df.columns
    index = df.index
    
    # plotting the return series
    print('Plots: ')
    fig, ax = plt.subplots(1,len(series), figsize=(15,5))
    for i in range(len(series)):
        ax[i].plot(index, df[series[i]])
        ax[i].set_title(series[i].replace('_ret',''))
    plt.tight_layout()
    plt.show()
    
    
    # summary statistics calculation
    print('Summary statistics: ')
    print('')
    summstats_df = pd.DataFrame()
    for i in series:
        summstats_df[i] = [len(df[i]), np.mean(df[i]), np.median(df[i]), np.std(df[i]), 
                     sc.skew(df[i]), sc.kurtosis(df[i]), min(df[i]), max(df[i])]
    summstats_df.index = ['Number of obs', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max']
    pd.set_option("display.max_columns", None)
    print(summstats_df.T)
    print('')
    
    # print('LaTeX output: ') # for the report
    # print(summstats_df.T.to_latex())
    # print('')
    
    return


def output_Q2():
    """Produces output for question 2."""
    print("Question 2: ")
    def Sig2Hat(dOmega, dBeta, dAlpha, dXt, dMu, dDelta, dSig2Initial, dLambda):
        """Calculates variance for time t+1."""
        #If statements to handle indicator function.
        if dXt < 0:
            iIndicator = 1
        else:
            iIndicator = 0

        #Variance calculation.
        dSig2tPlus1 = dOmega + (dBeta + (dAlpha * (dXt - dMu)**2 + (dDelta * (dXt - dMu)**2) * iIndicator)/(dSig2Initial + 1/dLambda * (dXt - dMu)**2)) * dSig2Initial

        return dSig2tPlus1

    #Class for easily calculating news-impact curves.
    class RobustGarchLev(object):
    
        def __init__(self, dLambda, dDelta):
            self.dSig2Initial = 1
            self.dMu = 0
            self.dOmega = 0
            self.dAlpha = 0.05
            self.dBeta = 0.9
            self.dLambda = dLambda
            self.dDelta = dDelta
            self.vX = np.linspace(start=-6, stop= 6, num=12000)
            self.vSig2tPlus1 = [Sig2Hat(self.dOmega, self.dBeta, self.dAlpha, x, self.dMu, self.dDelta, self.dSig2Initial, self.dLambda) for x in self.vX]

    #Pre-determined lambda and delta values.
    vLambda = np.array([2, 5, 10, 50])
    vDelta = np.array([0, 1, 0.2, 0.4])
    vXaxis = np.linspace(start=-6, stop= 6, num=12000)
    lNewsImpactCurves = []

    #Loop through values and store curves.
    for dDelta in vDelta:
        for dLambda in vLambda:
            lNewsImpactCurves.append(RobustGarchLev(dLambda, dDelta).vSig2tPlus1)

    print("\nPlots: ")
    #Plot curves.
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True)
    ax1.plot(vXaxis, lNewsImpactCurves[0], label = "d = 0, l = 2")
    ax1.plot(vXaxis, lNewsImpactCurves[1], label = "d = 0, l = 5")
    ax1.plot(vXaxis, lNewsImpactCurves[2], label = "d = 0, l = 10")
    ax1.plot(vXaxis, lNewsImpactCurves[3], label = "d = 0, l = 50")
    ax1.set_ylim([0,20])
    ax1.tick_params('x', labelbottom=False)
    ax1.set_ylabel("Sigma2_t+1")
    ax1.legend()
    ax2.plot(vXaxis, lNewsImpactCurves[4], label = "d = 0.1, l = 2")
    ax2.plot(vXaxis, lNewsImpactCurves[5], label = "d = 0.1, l = 5")
    ax2.plot(vXaxis, lNewsImpactCurves[6], label = "d = 0.1, l = 10")
    ax2.plot(vXaxis, lNewsImpactCurves[7], label = "d = 0.1, l = 50")
    ax2.set_ylim([0,20])
    ax2.tick_params(
        axis = 'x',
        labelbottom=False)
    ax2.tick_params(
        axis = 'y',
        labelleft=False)
    ax2.legend()
    ax3.plot(vXaxis, lNewsImpactCurves[8], label = "d = 0.2, l = 2")
    ax3.plot(vXaxis, lNewsImpactCurves[9], label = "d = 0.2, l = 5")
    ax3.plot(vXaxis, lNewsImpactCurves[10], label = "d = 0.2, l = 10")
    ax3.plot(vXaxis, lNewsImpactCurves[11], label = "d = 0.2, l = 50")
    ax3.set_ylim([0,20])
    ax3.set_xlabel("Xt")
    ax3.set_ylabel("Sigma2_t+1")
    ax3.legend()
    ax4.plot(vXaxis, lNewsImpactCurves[12], label = "d = 0.4, l = 2")
    ax4.plot(vXaxis, lNewsImpactCurves[13], label = "d = 0.4, l = 5")
    ax4.plot(vXaxis, lNewsImpactCurves[14], label = "d = 0.4, l = 10")
    ax4.plot(vXaxis, lNewsImpactCurves[15], label = "d = 0.4, l = 50")
    ax4.set_ylim([0,20])
    ax4.set_xlabel("Xt")
    ax4.legend()
    plt.tight_layout()
    plt.show()
    
    return

#%% 
###########################################################
### main
def main():
    # magic numbers
    path = r"data_ass_2.csv"
    df_test, df_real = loadin_data(path)
    output_Q1(df_real)
    output_Q2()
    # now call the functions that print all of the output for all questions

###########################################################
### start main
if __name__ == "__main__":
    main()

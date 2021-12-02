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
import datetime as dt


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
###########################################################
### output_Q2
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

###########################################################
### output_Q4
def output_Q4(df, estimates):
    """
    Function that prints all the plots for question 4.

    Parameters
    ----------
    df : dataframe of returns, used for getting estimated volatilities
    estimates : output of parameter estimates from question 3
    
    Returns
    -------
    None.

    """  
    series = df.columns
    # to do:
        # get vola's for all series...
        # get \sigma_2500
        # get impact curve for \sigma_2500
        # plot vola's w vertical line at t=2500
        
    def sigma_t(xlag1, mu, omega, beta, alpha, delta, Lambda, sigmalag1, lev):
        # calculates the next sigma based on the xt and params
        # every param is scalar, but lev is a bool
        diffsq = (xlag1-mu)**2
        if xlag1 > 0:
            delta = 0
        
        if lev == False:
            delta = 0
        sigma_t = omega + (beta + ((alpha*diffsq + delta*diffsq)/(sigmalag1 + (1/Lambda)*diffsq)))*sigmalag1
        
        return sigma_t
    
    volas_lev = np.ones((len(df), 3)) # volatilities with leverage
    volas_nolev = np.ones((len(df), 3)) # volatilities without leverage
    fig, ax = plt.subplots(3,2, figsize=(15,10))
    
    for i in range(3):
        mu = estimates[i, 0]
        omega = estimates[i, 1] 
        beta = estimates[i, 2]
        alpha = estimates[i, 3]
        delta = estimates[i, 4]
        Lambda = estimates[i, 5]
        sigma1 = estimates[i, 6]
        
        volas_lev[0,i] = sigma1
        volas_nolev[0,i] = sigma1
        
        for j in range(1, len(volas_lev)):
            volas_lev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_lev[j-1,i], True)
            volas_nolev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_nolev[j-1,i], False)
        
        # get range of xt...
        # get answers of sigmat+1 based on this xt, for lev and no lev
        # plot then
        rangext = np.linspace(-0.05, 0.05, 1000)
        NIC_lev = np.empty((len(rangext),))
        NIC_nolev = np.empty((len(rangext),))
        for t in range(len(rangext)):
            NIC_lev[t] = sigma_t(rangext[t], mu, omega, beta, alpha, delta, Lambda, volas_lev[2500,i], True)
            NIC_nolev[t] = sigma_t(rangext[t], mu, omega, beta, alpha, delta, Lambda, volas_lev[2500,i], False)     
            #NIC_lev[t] = sigma_t(rangext[t], mu, omega, beta, alpha, delta, Lambda, 1, True)
            #NIC_nolev[t] = sigma_t(rangext[t], mu, omega, beta, alpha, delta, Lambda, 1, False)            

            
        
        ax[i, 0].plot(rangext, NIC_lev) # NIC here...
        ax[i, 0].plot(rangext, NIC_nolev)
        ax[i, 0].set_xlabel('xt')
        ax[i, 0].set_ylabel('sigma2_t+1')
        ax[i, 0].legend(['leverage','no leverage'])
        # ax labels, legend
        
        # title, axlabels, whatevs
        ax[i, 1].plot(df.index[5:], volas_lev[5:,i]) # and maybe nolev as well??
        ax[i, 1].plot(df.index[5:], volas_nolev[5:,i])
        ax[i, 1].axvline(pd.Timestamp('2009-12-10'), color='black', linestyle='--', linewidth=1)
        ax[i, 1].set_xlabel('sigma2')
        ax[i, 1].legend(['leverage', 'no leverage'])

        
        
    plt.tight_layout()
    plt.show()
    
    return 

###########################################################
### main
def main():
    # magic numbers
    path = r"data_ass_2.csv"
    df_test, df_real = loadin_data(path)

    # now call the functions that print all of the output for all questions
    output_Q1(df_real)
    output_Q2()
    
    #estimates = output_Q3(df)
    #output_Q4(df_test, estimates)
    
###########################################################
### start main
if __name__ == "__main__":
    main()

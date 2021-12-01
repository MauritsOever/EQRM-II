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
    # placeholder estimates...
    estimates = np.array([[0.0, 0.0, 0.9, 0.05, 0.4, 5, 1],
                          [0.0, 0.0, 0.9, 0.05, 0.4, 5, 1],
                          [0.0, 0.0, 0.9, 0.05, 0.4, 5, 1]])
    
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
        if xlag1 < 0:
            delta = 0
        
        if lev == True:
            delta = 0
        sigma_t = omega + (beta + ((alpha*diffsq + delta*diffsq)/(sigmalag1 + (1/Lambda)*diffsq)))*sigmalag1
        
        return sigma_t
    
    volas_lev = np.ones((len(df), 3)) # volatilities with leverage
    volas_nolev = np.ones((len(df), 3)) # volatilities without leverage
    fig, ax = plt.subplots(3,2)
    
    for i in range(1):
        mu = estimates[i, 0]
        omega = estimates[i, 1] 
        beta = estimates[i, 2]
        alpha = estimates[i, 3]
        delta = estimates[i, 4]
        Lambda = estimates[i, 5]
        sigma1 = estimates[i, 6]
        
        for j in range(1, len(volas_lev)):
            volas_lev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_lev[j-1,i], True)
            volas_nolev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_nolev[j-1,i], False)
        
        # get range of xt...
        # get answers of sigmat+1 based on this xt, for lev and no lev
        # plot then
        ax[i, 0].plot() # NIC here...
        # title, axlabels, whatevs
        ax[i, 1].plot(df.index, volas_lev[:,i]) # and maybe nolev as well??
        
        
        
    plt.tight_layout()
    plt.show()
        



    return 




#%% 
###########################################################
### main
def main():
    # magic numbers
    path = r"C:\Users\gebruiker\Documents\GitHub\EQRM-II\data_ass_2.csv"
    df_test, df_real = loadin_data(path)
    
    # now call the functions that print all of the output for all questions
    output_Q1(df_real)
    
    
###########################################################
### start main
if __name__ == "__main__":
    main()

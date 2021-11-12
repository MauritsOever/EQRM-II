# -*- coding: utf-8 -*-
"""
EQRM II
Assignment 1 main

Created on Tue Nov  9 12:01:46 2021

@author: Maurits van den Oever and Connor Stevens
"""

# load in packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import statsmodels.tsa.stattools as st




# load in data

def loadin_data(path):
    data = pd.read_csv(path, sep = ";").iloc[:,1:]
    # okay so the numbers have commas, so there interpreted as strings, lets see if we can change it
    for i in range(len(data.columns)):
        data.iloc[:,i] = data.iloc[:,i].apply(lambda x : x.replace(',', '.'))
        if i>0:    
            data.iloc[:,i] = pd.to_numeric(data.iloc[:,i])
        else:
            data.iloc[:,i] = pd.to_datetime(data.iloc[:,i])

    df = data.dropna(axis=0) 
    
    df['DJIA.Ret'] = np.log(df.iloc[:,1]) - np.log(df.iloc[:,1].shift(1))
    df['N225.Ret'] = np.log(df.iloc[:,2]) - np.log(df.iloc[:,2].shift(1))
    df['SSMI.Ret'] = np.log(df.iloc[:,3]) - np.log(df.iloc[:,3].shift(1))
    
    return df

###############################################################################
### output_Q1
def output_Q1(df):
    """
    Function that produces all output associated with question one

    Parameters
    ----------
    df : dataframe produced by function loadin_data(path)

    Returns
    -------
    None.

    """
    # for this question, we only use data until 2010
    # we dont have date indexing or whatever, so we dont know when to split the data
    # but we assume:
    df = df.iloc[:5000,:] # this needs to change xdd
    
    # magic numbers:
    cols = ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']
    
    
    # Q1 a
    # plot the prices series:
    fig, ax = plt.subplots(nrows = 1,ncols = 3, figsize = (15, 4))
    index = range(len(df))
    ax[0].plot(index, df['DJIA.Close'])
    #ax0.set_title('DJIA Close')
    ax[1].plot(index, df['N225.Close'])
    #ax1.set_title('N225 Close')
    ax[2].plot(index, df['SSMI.Close'])
    #ax2.set_title('SSMI Close')
    plt.tight_layout()
    plt.show()
    
    
    
    # Q1 b
    # perform dickey fuller tests on the price series:
    # employ DF stat from Tsay page 77
    # then check crit values
    for col in ['DJIA.Close', 'N225.Close', 'SSMI.Close']:
        mx = np.ones((len(df[col][1:-1]),2))
        mx[:,1] = df[col][1:-1]
        y = df[col][2:]
        phi_hat = np.linalg.inv(mx.T @ mx) @ mx.T @ y
        e = y - mx@phi_hat
        standard_errors = np.var(e) * np.linalg.inv(mx.T@mx)
        
        DF_stat = (phi_hat[1]-1) / np.sqrt(standard_errors[1,1])
        print('dickey fuller test statistic for', col, ' = ', DF_stat)
        if np.abs(DF_stat) > 2.86:
            print('so the series is stationary')
        else:
            print('So the series is non-stationary')
            
        print('')
    
    
        
    
    # Q1 c
    # plot returns of these series
    fig, ax = plt.subplots(nrows = 1,ncols = 3, figsize = (15, 4))
    index = range(len(df)-1)
    ax[0].plot(index, df['DJIA.Ret'][1:])
    #ax0.set_title('DJIA Close')
    ax[1].plot(index, df['N225.Ret'][1:])
    #ax1.set_title('N225 Close')
    ax[2].plot(index, df['SSMI.Ret'][1:])
    #ax2.set_title('SSMI Close')
    plt.tight_layout()
    plt.show()
    
    
    # Q1 d
    # sim iid gaus and student-t(4) and put in fourth panel of picture, notice anything different?
    # no scale specified so i guess just standard ~iid(0,1)?
    # sim
    series_gaus = np.random.normal(0,1, len(df)-1)
    series_t    = np.random.standard_t(4, len(df)-1)
    
    #plot
    fig, ax = plt.subplots(nrows = 1,ncols = 4, figsize = (15, 4))
    index = range(len(df)-1)
    ax[0].plot(index, df['DJIA.Ret'][1:])
    #ax0.set_title('DJIA Close')
    ax[1].plot(index, df['N225.Ret'][1:])
    #ax1.set_title('N225 Close')
    ax[2].plot(index, df['SSMI.Ret'][1:])
    #ax2.set_title('SSMI Close')
    ax[3].plot(index, series_t)
    ax[3].plot(index, series_gaus)
    plt.tight_layout()
    plt.show()
    # normal obvi doesnt catch the extreme values
    # student t doesnt capture skewness of data, normal data has way more extreme negative values
    # than positive
    
    
    # Q1 e
    # make table of summ stats, including nr_obs, mean, median, std, skew, kurt, min, max
    summstats_df = pd.DataFrame()
    for col in ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']:
        summstats_df[col] = [len(df[col][1:]), np.mean(df[col][1:]), np.median(df[col][1:]), np.std(df[col][1:]), 
                             sc.skew(df[col][1:]), sc.kurtosis(df[col][1:]), min(df[col][1:]), max(df[col][1:])]
        
    summstats_df.index = ['Number of obs', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max']
    
    print(summstats_df)
    # summstats_df.to_latex()
    
    
    # Q1 f
    # first 12 lags of ACF are signifant at 5% level??
    st.acf(df['DJIA.Ret'][1:], nlags=12)
    
    # for 1 to 12 lags, get sample mean and estimate correlations...
    acfs = np.empty((3,12))
    tstats = np.empty((3,12))

    for j in range(3):
        col = cols[j]
        for i in range(1, 13):
            series_t = np.array(df[col][1+i:])
            series_tminus = np.array(df[col][1:-i])
            
            mean = np.mean(df[col][1:])
            var = np.sum((df[col][1:]-mean)**2)/(len(df[col][1:]))
            cov = np.sum((series_t-mean)*(series_tminus-mean))/(len(series_t))
            acfs[j, i-1] =  cov/var # assume stationarity?  

        tstats[j,:] = acfs[j,:] / np.sqrt(1 + 2 * np.sum(acfs[j,:]**2)/len(acfs[j,:]))
    pvals = sc.norm.pdf(tstats) # no significance, to be expected...
    print('we should print some stuff here maybe for output...')
    
    # Q1 g
    # get acfs for 100 lags...
    acfs = np.empty((3,100))
    for j in range(3):
        col = cols[j]
        for i in range(1, 101):
            series_t = np.array(df[col][1+i:])
            series_tminus = np.array(df[col][1:-i])
            
            mean = np.mean(df[col][1:])
            var = np.sum((df[col][1:]-mean)**2)/(len(df[col][1:]))
            cov = np.sum((series_t-mean)*(series_tminus-mean))/(len(series_t))
            acfs[j, i-1] =  cov/var # assume stationarity?  
    print('we should probs plot these...')
    
    return
    
    
###############################################################################
### output_Q2
def output_Q2(df):
    


###########################################################
### main
def main():
    # magic numbers
    path = r"C:\Users\gebruiker\Documents\GitHub\EQRM-II\triv_ts.txt"
    df = loadin_data(path)
    
    output_Q1(df)


###########################################################
### start main
if __name__ == "__main__":
    main()

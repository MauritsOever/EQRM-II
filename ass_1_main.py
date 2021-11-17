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
import scipy.optimize as opt
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
    print('Question 1 a: ')
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
    print('')
    
    
    # Q1 b
    # perform dickey fuller tests on the price series:
    # employ DF stat from Tsay page 77
    # then check crit values
    print('Question 1 b: ')
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
            print('so the series ', col,' is stationary')
        else:
            print('So the series' , col,' is non-stationary')
            
        print('')
    
    for col in ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']:
        mx = np.ones((len(df[col][1:-1]),2))
        mx[:,1] = df[col][1:-1]
        y = df[col][2:]
        phi_hat = np.linalg.inv(mx.T @ mx) @ mx.T @ y
        e = y - mx@phi_hat
        standard_errors = np.var(e) * np.linalg.inv(mx.T@mx)
        
        DF_stat = (phi_hat[1]-1) / np.sqrt(standard_errors[1,1])
        print('dickey fuller test statistic for', col, ' = ', DF_stat)
        if np.abs(DF_stat) > 2.86:
            print('so the series ', col,' is stationary')
        else:
            print('So the series' , col,' is non-stationary')
            
        print('')
        
    
    # Q1 c
    # plot returns of these series
    print('Question 1 c: ')
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
    print('')
    
    # Q1 d
    # sim iid gaus and student-t(4) and put in fourth panel of picture, notice anything different?
    # no scale specified so i guess just standard ~iid(0,1)?
    # sim
    print('Question 1 d: ')
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
    print('Question 1 e: ')
    summstats_df = pd.DataFrame()
    for col in ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']:
        summstats_df[col] = [len(df[col][1:]), np.mean(df[col][1:]), np.median(df[col][1:]), np.std(df[col][1:]), 
                             sc.skew(df[col][1:]), sc.kurtosis(df[col][1:]), min(df[col][1:]), max(df[col][1:])]
        
    summstats_df.index = ['Number of obs', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'min', 'max']
    
    print(summstats_df)
    print('')
    # summstats_df.to_latex()
    
    
    # Q1 f
    # first 12 lags of ACF are signifant at 5% level??
    print('Question 1 f: ')
    
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
    print('')
    
    # Q1 g
    # get acfs for 100 lags...
    print('Question 1 g: ')
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
    print('')
    
    return
    
    
###############################################################################
### output_Q2
def output_Q4(df):
    """
    Function that produces all the output of Q4

    Parameters
    ----------
    df : dataframe produced by function loadin_data(path)

    Returns
    -------
    None.

    To do:
        - test linear version, see if similar
        - subquestion c boiii

    """
    rets = ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']
    df = df[rets][1:] #reduce size of df for easier looping 
    
    # Q4 a
    print('Question 4 a: ')
    # get cross covariances for each lag for each series...
    cross_covs = np.ones((25,9))
    column_count = 0
    
    for series1 in rets:
        for series2 in rets:
            for k in range(1,26):
                ret1 = df[series1][k:]
                ret2 = df[series2][:-k]
                #cross_covs[k-1, column_count] = np.sum((ret1-np.mean(ret1))*(ret2-np.mean(ret2)))/(len(ret1)) # slightly different values
                cross_covs[k-1, column_count] = np.cov(ret1,ret2)[0,1]
    
            column_count += 1
    
    fig, ax = plt.subplots(3,3, figsize= (10,10))
    column_count = 0
    for i in range(0,3):
        for j in range(0,3):
            ax[i,j].plot(cross_covs[:,column_count])
            ax[i,j].axhline()
            ax[i,j].set_title(rets[i]+', '+rets[j])
            column_count += 1
    plt.tight_layout()
    plt.show()
    
    
    
    # Q4 b
    # estimating the VAR(1) and VAR(2) model by estimator given in assignment...
    print('Question 4 b: ')
    print('')
    # get OLS ests for VAR(1) as initial values...
    y = np.array(df[rets][1:]).T # define data used and put in right shape
    yt = y[:,1:]
    Z = np.empty((4,len(yt.T)))
    Z[0,:] = 1
    Z[1:,:] = y[:,:-1]
    beta_OLS = ((yt@Z.T)@np.linalg.inv(Z@Z.T)).T
    
    params = np.empty(12)
    params[0:3] = beta_OLS[0,:]
    params[3:6] = beta_OLS[1:,0] 
    params[6:9] = beta_OLS[1:,1] 
    params[9:] = beta_OLS[1:,2]
    
    # now the ML routine...
    def log_lik_var1(y, params):
        mu = np.reshape(params[0:3], (3,1))
        phi = np.reshape(params[3:], (3,3))
        
        yt = y[:,1:]
        ylag1 = y[:,:-1]
        
        eps = yt - phi@ylag1 - mu
        
        sigma = (eps@eps.T)/len(yt.T)
        LLs = np.empty(len(yt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(yt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]
        
        return LLs
    
    print('Fitting VAR(1) model...')
    
    AvgNLL = lambda params : -np.mean(log_lik_var1(y, params)) # define function to be minimized 
    res_var1    = opt.minimize(AvgNLL, params, method='Powell') # algos that work: Powell, 
    print(res_var1.message)
    mu_hat_var1 = np.reshape(res_var1.x[0:3], (3,1))
    phi_hat_var1 = np.reshape(res_var1.x[3:], (3,3))
    # maybe print params...
    print('mu = ', mu_hat_var1)
    print('')
    print('phi hat = ', phi_hat_var1)
    print('')
    # get AIC, BIC and HIC
    eps_hat = y[:,1:] - phi_hat_var1@y[:,:-1] - mu_hat_var1
    sigma_hat_var1 = (eps_hat@eps_hat.T)/len(y[:,1:].T)
    print('Sigma hat = ', sigma_hat_var1)
    print('')
    m = 0.5*3*(3+1) + 3 + 1*3**2
    T = len(y[:,1:].T)
    
    AIC_var1 = np.log(np.linalg.det(sigma_hat_var1)) + 2*m/T
    AICc_var1 = np.log(np.linalg.det(sigma_hat_var1)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_var1 = np.log(np.linalg.det(sigma_hat_var1)) + m*np.log(T)/T
    
    criterions_var1 = np.array([AIC_var1, AICc_var1, BIC_var1])
    print('criterions of VAR(1) (AIC, AICc, BIC) are ', criterions_var1)
    print('')
    
    # get OLS estimates for VAR(2)
    yt = y[:,2:]
    ylag1 = y[:,1:-1]
    ylag2 = y[:,:-2]
    Z = np.empty((7,len(yt.T)))
    Z[0,:] = 1
    Z[1:4,:] = ylag1
    Z[4:, :] = ylag2
    beta_OLS = ((yt@Z.T)@np.linalg.inv(Z@Z.T)).T
    params = np.zeros(21)
    params[0:3] = beta_OLS[0,:]
    params[3:6] = beta_OLS[1:4,0] 
    params[6:9] = beta_OLS[1:4,1] 
    params[9:12] = beta_OLS[1:4,2]
    params[12:15] = beta_OLS[4:,0]
    params[15:18] = beta_OLS[4:,1]
    params[18:] = beta_OLS[4:,2]
    
    # now do VAR(2)
    def log_lik_var2(y, params):
        mu = np.reshape(params[0:3], (3,1))
        phi1 = np.reshape(params[3:12], (3,3))
        phi2 = np.reshape(params[12:], (3,3))
        
        yt = y[:,2:]
        ylag1 = y[:,1:-1]
        ylag2 = y[:,:-2]
        
        eps = yt - phi1@ylag1 - phi2@ylag2 - mu
        sigma = (eps@eps.T)/len(yt.T)
        LLs = np.empty(len(yt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(yt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]
        
        return LLs
    
    y = np.array(df[rets][1:]).T # define data used and put in right shape
    print('Fitting VAR(2)...')
    print('')
    AvgNLL = lambda params : -np.mean(log_lik_var2(y, params)) # define function to be minimized
    res_var2 = opt.minimize(AvgNLL, params, method='Powell')
    print(res_var2.message)
    mu = np.reshape(res_var2.x[0:3], (3,1))
    phi1 = np.reshape(res_var2.x[3:12], (3,3))
    phi2 = np.reshape(res_var2.x[12:], (3,3))
    
    print('mu hat = ', mu)
    print('')
    print('phi1 hat = ', phi1)
    print('')
    print('phi1 hat = ', phi2)
    print('')
    
    eps_hat = y[:,2:] - phi1@y[:,1:-1] - phi2@y[:,:-2] - mu
    sigma_hat_var2 = (eps_hat@eps_hat.T)/len(y[:,2:].T)
    print('sigma hat = ', sigma_hat_var2)
    
    m = 0.5*3*(3+1) + 3 + 2*3**2
    T = len(y[:,2:].T)
    
    AIC_var2 = np.log(np.linalg.det(sigma_hat_var2)) + 2*m/T
    AICc_var2 = np.log(np.linalg.det(sigma_hat_var2)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_var2 = np.log(np.linalg.det(sigma_hat_var2)) + m*np.log(T)/T
    
    criterions_var2 = np.array([AIC_var2, AICc_var2, BIC_var2])
    
    print('criterions of VAR(2) (AIC, AICc, BIC) are ', criterions_var2)
    print('')
    
    return

###########################################################
### main
def main():
    # magic numbers
    path = r"C:\Users\gebruiker\Documents\GitHub\EQRM-II\triv_ts.txt"
    df = loadin_data(path)
    
    # output_Q1(df)
    
    output_Q4(df)

###########################################################
### start main
if __name__ == "__main__":
    main()

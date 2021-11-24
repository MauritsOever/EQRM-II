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
    # okay so the numbers have commas, so they're interpreted as strings, lets see if we can change it
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
    df = df[df['Day']<'01-01-2011']
    
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
    #print(summstats_df.to_latex())
    
    
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

        tstats[j,:] = acfs[j,:] / (np.sqrt(1 + 2 * np.sum(acfs[j,:]**2))/(len(df)-1))
    pvals = sc.norm.pdf(tstats) # no significance, to be expected...
    
    q1f = pd.DataFrame()
    q1f['DJIA acfs'] = acfs[0,:]
    q1f['DJIA tstats'] = tstats[0,:]
    q1f['DJIA pvals'] = pvals[0,:]
    q1f['N225 acfs'] = acfs[1,:]
    q1f['N225 tstats'] = tstats[1,:]
    q1f['N225 pvals'] = pvals[1,:]
    q1f['SSMI acfs'] = acfs[2,:]
    q1f['SSMI tstats'] = tstats[2,:]
    q1f['SSMI pvals'] = pvals[2,:]
    q1f['index'] = range(1,13)
    q1f = q1f.set_index('index')
    q1f = q1f.round(decimals=4)
    print(q1f.T)
    #print(q1f.T.to_latex())
    
    
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
            
    fig, ax = plt.subplots(nrows = 1,ncols = 3, figsize = (15, 4))
    index = range(1,101)
    ax[0].plot(index, acfs[0,:])
    ax[0].set_title('DJIA')
    ax[1].plot(index, acfs[1,:])
    ax[1].set_title('N225')
    ax[2].plot(index, acfs[2,:])
    ax[2].set_title('SSMI')
    plt.tight_layout()
    plt.show()
    print('')
    
    return
    
    
###############################################################################
### output_Q4
def output_Q4(df):
    """
    Function that produces all the output of Q4

    Parameters
    ----------
    df : dataframe produced by function loadin_data(path)

    Returns
    -------
    estimates   :   dictionary of estimated objects needed for Q5

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
    
    fig, ax = plt.subplots(3,3, figsize= (15,10))
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
    res_var1    = opt.minimize(AvgNLL, params, method='SLSQP') # algos that work: Powell, 
    print(res_var1.message)
    mu_hat_var1 = np.reshape(res_var1.x[0:3], (3,1))
    phi_hat_var1 = np.reshape(res_var1.x[3:], (3,3))
    # maybe print params...
    print('mu = ', mu_hat_var1.round(decimals=4))
    print('')
    print('phi hat = ', phi_hat_var1.round(decimals=4))
    print('')
    # get AIC, BIC and HIC
    eps_hat = y[:,1:] - phi_hat_var1@y[:,:-1] - mu_hat_var1
    sigma_hat_var1 = (eps_hat@eps_hat.T)/len(y[:,1:].T)
    print('Sigma hat = ', sigma_hat_var1.round(decimals=4))
    print('')
    m = 0.5*3*(3+1) + 3 + 1*3**2
    T = len(y[:,1:].T)
    
    AIC_var1 = np.log(np.linalg.det(sigma_hat_var1)) + 2*m/T
    AICc_var1 = np.log(np.linalg.det(sigma_hat_var1)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_var1 = np.log(np.linalg.det(sigma_hat_var1)) + m*np.log(T)/T
    
    criterions_var1 = np.array([AIC_var1, AICc_var1, BIC_var1])
    print('criterions of VAR(1) (AIC, AICc, BIC) are ', criterions_var1.round(decimals=4))
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
    res_var2 = opt.minimize(AvgNLL, params, method='SLSQP')
    print(res_var2.message)
    mu = np.reshape(res_var2.x[0:3], (3,1))
    phi1 = np.reshape(res_var2.x[3:12], (3,3))
    phi2 = np.reshape(res_var2.x[12:], (3,3))
    
    print('mu hat = ', mu.round(decimals=4))
    print('')
    print('phi1 hat = ', phi1.round(decimals=4))
    print('')
    print('phi2 hat = ', phi2.round(decimals=4))
    print('')
    
    eps_hat = y[:,2:] - phi1@y[:,1:-1] - phi2@y[:,:-2] - mu
    sigma_hat_var2 = (eps_hat@eps_hat.T)/len(y[:,2:].T)
    print('sigma hat = ', sigma_hat_var2.round(decimals=4))
    print('')
    m = 0.5*3*(3+1) + 3 + 2*3**2
    T = len(y[:,2:].T)
    
    AIC_var2 = np.log(np.linalg.det(sigma_hat_var2)) + 2*m/T
    AICc_var2 = np.log(np.linalg.det(sigma_hat_var2)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_var2 = np.log(np.linalg.det(sigma_hat_var2)) + m*np.log(T)/T
    
    criterions_var2 = np.array([AIC_var2, AICc_var2, BIC_var2])
    
    print('criterions of VAR(2) (AIC, AICc, BIC) are ', criterions_var2.round(decimals=4))
    print('')
    
    #question 4 c
    print('Question 4 c: ')
    print('')
    #initialize 0 matrix P
    # python implementation of the Cholesky-Banachiewicz algorithm
    def cholesky_decomp(matrix):
        P_hat = np.zeros((3,3))
        for i in range(3):
            for k in range(i+1):
                tmp_sum = sum(P_hat[i][j] * P_hat[k][j] for j in range(k))
                if (i == k): # Diagonal elements
                    P_hat[i][k] = np.sqrt(matrix[i][i] - tmp_sum)
                else:
                    P_hat[i][k] = (1.0 / P_hat[k][k] * (matrix[i][k] - tmp_sum))
        return P_hat

    P_hat_var1 = cholesky_decomp(sigma_hat_var1)
    P_hat_var2 = cholesky_decomp(sigma_hat_var2)
    
    print('P_hat for VAR(1) = ')
    print(P_hat_var1.round(decimals=4))
    print('')
    
    print('P_hat for VAR(2) = ')
    print(cholesky_decomp(sigma_hat_var2).round(decimals=4))
    print('')
    
    Q5_ests = {'P_hat_var1':P_hat_var1, 'P_hat_var2':P_hat_var2, 'phi_hat_var1':phi_hat_var1,
               'phi1': phi1, 'phi2':phi2}
    
    
    return Q5_ests

###########################################################
### output_Q5
def output_Q5(estimates):
    """
    Function that produces all the output for Q5

    Parameters
    ----------
    estimates : dictionary created by function output_Q4, including all parameter 
                estimates for VAR(1 and 2) and corresponding sigmas...

    Returns
    -------
    None.

    """
    print('Question 5 IRF for VAR(1)')
    print('')
    IRFlen = 11
    selection1 = np.reshape(np.array([1,0,0]), (3,1))
    selection2 = np.reshape(np.array([0,1,0]), (3,1))
    selection3 = np.reshape(np.array([0,0,1]), (3,1))
    
    store1 = np.empty((IRFlen,3))
    store2 = np.empty((IRFlen,3))
    store3 = np.empty((IRFlen,3))
    
    for i in range(1, IRFlen+1):
        store1[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection1, (3,))
        store2[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection2, (3,))
        store3[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection3, (3,))
        
    fig, ax = plt.subplots(3,3, figsize= (15,10))
    ax[0,0].plot(store1[:,0])
    ax[0,0].set_title('Effect of DJIA shock on DJIA')
    ax[0,1].plot(store1[:,1])
    ax[0,1].set_title('Effect of N225 shock on DJIA')
    ax[0,2].plot(store1[:,2])
    ax[0,2].set_title('Effect of SSMI shock on DJIA')
    ax[1,0].plot(store2[:,0])
    ax[1,0].set_title('Effect of DJIA shock on N225')
    ax[1,1].plot(store2[:,1])
    ax[1,1].set_title('Effect of N225 shock on N225')
    ax[1,2].plot(store2[:,2])
    ax[1,2].set_title('Effect of SSMI shock on N225')
    ax[2,0].plot(store3[:,0])
    ax[2,0].set_title('Effect of DJIA shock on SSMI')
    ax[2,1].plot(store3[:,1])
    ax[2,1].set_title('Effect of N225 shock on SSMI')
    ax[2,2].plot(store3[:,2])
    ax[2,2].set_title('Effect of SSMI shock on SSMI')
    plt.tight_layout()
    plt.show()
    
    print('Question 5 IRF for VAR(2)')
    print('')
    ## okay now do IFRS for VAR(2)
    store1 = np.empty((IRFlen,3))
    store2 = np.empty((IRFlen,3))
    store3 = np.empty((IRFlen,3))

    for i in range(1, IRFlen+1):
        if i == 1:
            store1[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection1, (3,))
            store2[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection2, (3,))
            store3[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection3, (3,))
        else:
            store1[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection1 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,))
            store2[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection2 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,))
            store3[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection3 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,))


    fig, ax = plt.subplots(3,3, figsize= (15,10))
    ax[0,0].plot(store1[:,0])
    ax[0,0].set_title('Effect of DJIA shock on DJIA')
    ax[0,1].plot(store1[:,1])
    ax[0,1].set_title('Effect of N225 shock on DJIA')
    ax[0,2].plot(store1[:,2])
    ax[0,2].set_title('Effect of SSMI shock on DJIA')
    ax[1,0].plot(store2[:,0])
    ax[1,0].set_title('Effect of DJIA shock on N225')
    ax[1,1].plot(store2[:,1])
    ax[1,1].set_title('Effect of N225 shock on N225')
    ax[1,2].plot(store2[:,2])
    ax[1,2].set_title('Effect of SSMI shock on N225')
    ax[2,0].plot(store3[:,0])
    ax[2,0].set_title('Effect of DJIA shock on SSMI')
    ax[2,1].plot(store3[:,1])
    ax[2,1].set_title('Effect of N225 shock on SSMI')
    ax[2,2].plot(store3[:,2])
    ax[2,2].set_title('Effect of SSMI shock on SSMI')
    plt.tight_layout()
    plt.show()

    return
#%%
###########################################################
### main
def main():
    # magic numbers
    path = r"C:\Users\gebruiker\Documents\GitHub\EQRM-II\triv_ts.txt"
    df = loadin_data(path)
    
    output_Q1(df)
    
    estimates = output_Q4(df)
    output_Q5(estimates)

###########################################################
### start main
if __name__ == "__main__":
    main()

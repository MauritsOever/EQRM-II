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
import math
import matplotlib.pyplot as plt
import scipy.stats as sc
import scipy.optimize as opt
import statsmodels.tsa.stattools as st
import array_to_latex as a2l
import scipy.optimize as opt





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
### output_Q2
def Output_Q2(df):
    """Prints output used to answer question 2 part a."""
    def data_feeder(df, series):
        """Transforms dataframe into numpy array of appropriate layout for input
        in the subsequent maximum likelihood optimisation. Selectes data only up
        until the end of 2010, as specified in the assignment.

        Args:
            df (DataFrame): Contains prices and log returns of the three indices as
            well as a dated index.
            series (str): The name of the series to be extracted from the DataFrame.
            For example "DJIA.Ret".

        Returns:
            vY [array]: (Nx1) Numpy array containing the daily returns for the
            series.
            mX [array]: (Nx3) Numpy array with 1s in column zero, shifted return 
            observations (Y_t-1) in column one and residuals (e_t) in column two
        """
        df.rename(columns={"Day":"Date"}, inplace=True)
        df = df.set_index("Date")
        df = df.loc[: "2010-12-30"]
        y = df[series]
        y = pd.DataFrame({"t": df[series].shift(-2), "t-1": df[series].shift(-1),
        "t-2" : df[series].shift(0), "e_t" : 0})
        y.insert(loc = 0, column = "constant",value = int(1))
        y = y.dropna()
        mX = np.array(y[["constant", "t-1", "e_t"]])
        vY = np.array(y["t"])

        return vY, mX
################################################################################
    def LnLRegNorm(vP, vY, mX, p, q):
        """Calculates a vector of log-likelihoods for a given set of X and Y 
        variables. Additionally calculates recursive errors for MA type models.

        Args:
            vP (array)): Array containing the parameters of the regression. First
                parameter is always variance, followed by intercept. Susequent
                parameters occur in vP as the do in the respective ARMA equation.
            vY [array]: (Nx1) Numpy array containing the daily returns for the
                series.
            mX [array]: (Nx3) Numpy array with 1s in column zero, shifted return 
                observations (Y_t-1) in column one and residuals (e_t) in column two
            p (int): Order of AR component of ARMA model.
            q (int): Order of MA component of ARMA model.

        Returns:
        vLL (array): Vector of individual log-likelihood contributions of each
            observation.
        """
        (dS, vBeta)= vP[0], vP[1:]
        (iN, iK) = mX.shape

        # |contstant|y_t-1|e_t|
        #AR(1)
        if (p == 1 and q == 0):
            for i in range(1, iN):
                mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1])

        #AR(2)
        if (p == 2 and q == 0):
            for i in range(1, iN):
                mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1]
                - mX[i - 1, 1] * vBeta[2])

        #ARMA(1,1)
        if (p == 1 and q == 1):
            for i in range(1, iN):
                mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1] 
                - mX[i - 1, 2] * vBeta[2])

        #ARMA(2,1)
        if (p == 2 and q == 1):
            for i in range(1, iN):
                mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1] 
                - mX[i - 1, 1] * vBeta[2] - mX[i - 1, 2] * vBeta[3])
        #ARMA(1,2)
        if (p == 1 and q == 2):
            for i in range(2, iN):
                mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1] 
                - mX[i - 1, 2] * vBeta[2]- mX[i - 2, 2] * vBeta[3])

        #ARMA(2,2)
        if (p == 2 and q == 2):
            for i in range(2, iN):
                mX[i, -1] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1] 
                - mX[i - 1, 1] * vBeta[2]- mX[i - 1, 2] * vBeta[3]- mX[i - 2, 2] * vBeta[4])

        vE = mX[:,-1]

        vLL = np.zeros(iN)

        for j in range(0, iN):
            vLL[j] = -1/2 * np.log(2 * np.pi * (dS**2)) - (vE[j]**2)/(2 * dS**2)
        print('.', end='')
        return vLL
################################################################################
    def EstRegNorm(vY, mX, p, q):
        """Calculates the optimal maximum likelihood estimator for a given X and Y
        series.

        Args:
            vY [array]: (Nx1) Numpy array containing the daily returns for the
                series.
            mX [array]: (Nx3) Numpy array with 1s in column zero, shifted return 
                observations (Y_t-1) in column one and residuals (e_t) in column two
            p (int): Order of AR component of ARMA model.
            q (int): Order of MA component of ARMA model.


        Returns:
            res.x[list]: List of optimal parameters calculate for mle.
        """
        (iN, iK)= mX.shape
        #Initial guess.
        vP0 = np.full((p + q + 2), 0.02)
        print(len(vP0))
        SumNLnLReg= lambda vP: -np.sum(LnLRegNorm(vP, vY, mX, p, q))


        print ('Initial guess Log-Likelihood = {}'.format(-SumNLnLReg(vP0)))
        #optimize parameters such that they minimise the negative sum of the 
        #indiviual log likelihod contributions.
        res= opt.minimize(SumNLnLReg, vP0, method='Nelder-Mead')
        print ('\nResults_normal: ', res)
        print("\ndLL_normal=", -res.fun)
        return res.x, -res.fun
################################################################################
    def _gh_stepsize(vP):
        """
        Purpose:
            Calculate stepsize close (but not too close) to machine precision

        Inputs:
            vP      1D array of parameters

        Return value:
            vh      1D array of step sizes
        """
        vh = 1e-8*(np.fabs(vP)+1e-8)   # Find stepsize
        vh= np.maximum(vh, 5e-6)       # Don't go too small

        return vh
################################################################################
    def jacobian_2sided(fun, vP, *args):
        """
        Purpose:
        Compute numerical jacobian, using a 2-sided numerical difference

        Author:
        Charles Bos, following Kevin Sheppard's hessian_2sided, with
        ideas/constants from Jurgen Doornik's Num1Derivative

        Inputs:
        fun     function, return 1D array of size iN
        vP      1D array of size iP of optimal parameters
        args    (optional) extra arguments

        Return value:
        mG      iN x iP matrix with jacobian

        See also:
        numdifftools.Jacobian(), for similar output
        """
        iP = np.size(vP)
        vP= vP.reshape(iP)      # Ensure vP is 1D-array

        vF = fun(vP, *args)     # evaluate function, only to get size
        iN= vF.size

        vh= _gh_stepsize(vP)
        mh = np.diag(vh)        # Build a diagonal matrix out of h

        mGp = np.zeros((iN, iP))
        mGm = np.zeros((iN, iP))

        for i in range(iP):     # Find f(x+h), f(x-h)
            mGp[:,i] = fun(vP+mh[i], *args)
            mGm[:,i] = fun(vP-mh[i], *args)

        vhr = (vP + vh) - vP    # Check for effective stepsize right
        vhl = vP - (vP - vh)    # Check for effective stepsize left
        mG= (mGp - mGm) / (vhr + vhl)  # Get central jacobian

        return mG
################################################################################
    def hessian_2sided(fun, vP, *args):
        """
        Purpose:
        Compute numerical hessian, using a 2-sided numerical difference

        Author:
        Kevin Sheppard, adapted by Charles Bos

        Source:
        https://www.kevinsheppard.com/Python_for_Econometrics

        Inputs:
        fun     function, as used for minimize()
        vP      1D array of size iP of optimal parameters
        args    (optional) extra arguments

        Return value:
        mH      iP x iP matrix with symmetric hessian
        """
        iP = np.size(vP,0)
        vP= vP.reshape(iP)    # Ensure vP is 1D-array

        f = fun(vP, *args)
        vh= _gh_stepsize(vP)
        vPh = vP + vh
        vh = vPh - vP

        mh = np.diag(vh)            # Build a diagonal matrix out of vh

        fp = np.zeros(iP)
        fm = np.zeros(iP)
        for i in range(iP):
            fp[i] = fun(vP+mh[i], *args)
            fm[i] = fun(vP-mh[i], *args)

        fpp = np.zeros((iP,iP))
        fmm = np.zeros((iP,iP))
        for i in range(iP):
            for j in range(i,iP):
                fpp[i,j] = fun(vP + mh[i] + mh[j], *args)
                fpp[j,i] = fpp[i,j]
                fmm[i,j] = fun(vP - mh[i] - mh[j], *args)
                fmm[j,i] = fmm[i,j]

        vh = vh.reshape((iP,1))
        mhh = vh @ vh.T             # mhh= h h', outer product of h-vector

        mH = np.zeros((iP,iP))
        for i in range(iP):
            for j in range(i,iP):
                mH[i,j] = (fpp[i,j] - fp[i] - fp[j] + f + f - fm[i] - fm[j] + fmm[i,j])/mhh[i,j]/2
                mH[j,i] = mH[i,j]

        return mH
################################################################################
    def CalcResiduals(df, lParameters, vSeriesNames):
        for count, sSeriesName in enumerate(vSeriesNames):
            (vY, mX) = data_feeder(df, series = sSeriesName)
            (iN, iK) = mX.shape
            vBeta = lParameters[count]
            if (sSeriesName == "DJIA.Ret" or sSeriesName == "N225.Ret"):
                #AR(2)
                for i in range(0, iN):
                    mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1]
                    - mX[i - 1, 1] * vBeta[2])
                if sSeriesName == "DJIA.Ret":
                    vResiduals_DJIA = mX[:, 2]
            if sSeriesName == "SSMI.Ret":
                #ARMA(1,2)
                for i in range(2, iN):
                        mX[i, 2] = (vY[i] - mX[i, 0] * vBeta[0] - mX[i, 1] * vBeta[1] 
                        - mX[i - 1, 2] * vBeta[2]- mX[i - 2, 2] * vBeta[3])

            if sSeriesName == "DJIA.Ret":
                vResiduals_DJIA = mX[:, 2]
            if sSeriesName == "N225.Ret":
                vResiduals_N225 = mX[:, 2]
            if sSeriesName == "SSMI.Ret":
                vResiduals_SSMI = mX[2:, 2]
        return vResiduals_DJIA, vResiduals_N225, vResiduals_SSMI
################################################################################
    def ACF(vSeries, iLags):
        vAC = np.ones(iLags)
        for lag in range(1, iLags):
            vY_lag = vSeries[:-(lag)]
            vY = vSeries[lag:]
            dMean = np.mean(vY)
            dMean_lag = np.mean(vY_lag)
            dVariance = ((vY - dMean) @ (vY - dMean).T)/len(vY)
            dCovariance = ((vY - dMean) @ (vY_lag - dMean_lag).T)/len(vY)
            vAC[lag] = dCovariance/dVariance
        return vAC
################################################################################
    def PACF(vSeries, iLags):
        df = pd.DataFrame({"t":vSeries})

        vPACF = np.ones(iLags + 1)
        for phi in range(1, iLags + 1):
            df = pd.DataFrame({"t":vSeries})
            for lag in range(1, phi + 1):
                df["t-" + str(lag)] = df["t"].shift(lag)
            #print(df.head(10))
            df = df.iloc[iLags:, 1:]
            df.insert(loc=0, column="constant", value =1)
            mX = np.array(df)
            vY = vSeries[iLags:]

            vBeta = np.linalg.inv(mX.T@mX)@mX.T@vY
            vPACF[phi] = vBeta[phi]
        vPACF = vPACF[1:]
        return vPACF
################################################################################
    def LjungBox(vAutoCorrelation, iT):
        dChi2 = 1.145
        dFirstTerm = iT*(iT + 2)
        dSecondTerms = []
        for lag in range(1, 5):
            dSecondTerms.append((vAutoCorrelation[lag]**2)/(iT-lag))
            dSecondTerm = np.sum(dSecondTerms)
        Q = dFirstTerm * dSecondTerm
        if Q < dChi2:
            print("Q = " + str(Q) + "\nResiduals are white noise:" + "\n" + str(Q) 
            + "< " + str(dChi2))
        if Q > dChi2:
            print("Q = " + str(Q) + "\nResiduals are  not white noise:" + "\n" + str(Q) 
            + "> " + str(dChi2))
        return Q
################################################################################
    def JarqueBera(vSeries):
        iN = len(vSeries)
        dJB = (iN/6 * (sc.skew(vSeries)**2) + 1/4 * (sc.kurtosis(vSeries)-3)**2)
        return dJB
################################################################################

    # magic numbers
    lParameters = [(0.000145, -0.068644, -0.046275), (-0.000136, -0.035088, -0.033655), (0.000006, 0.140633, -0.135125, -0.041509)]
    vSeriesNames = ["DJIA.Ret", "N225.Ret", "SSMI.Ret"]
    path = r"triv_ts.txt"
    df = loadin_data(path)
    arma_models = [(1,0), (2,0), (1,1), (2,1), (1,2), (2,2)]
    series_list = ["DJIA.Ret", "N225.Ret", "SSMI.Ret"]
    #a)
    for series in series_list:
        print(series)
        (vY, mX) = data_feeder(df = df, series = series)
        if series == "DJIA.Ret":
            latex_output = np.zeros((15, 6))
        if (series == "N225.Ret" or series == "SSMI.Ret"):
            latex_extra = np.zeros((15, 6))
            latex_output = np.concatenate((latex_output, latex_extra), axis = 1)
        shift = 0
        if series == "N225.Ret":
            shift = 6
        if series == "SSMI.Ret":
            shift = 12
        for column, model in enumerate(arma_models):
            #Maximum likelihood estimator and reporting.
            (iN, iK) = mX.shape
            (p,q) = model
            print("\nARMA({}, {})".format(p,q))
            (vP_MLE, dLL) = EstRegNorm(vY, mX, p, q)
            print("\nSigma: {}".format(vP_MLE[0]) + "\nMu: {}".format(vP_MLE[1])
            + "\nFirst {} are AR coefficients, subsequent {} are MA "
            "coefficients:".format(p, q) + "\n{}".format(vP_MLE[2:]) 
            + "\nMLE Log-Likelihod = {}".format(dLL))

            #Information criterion and reporting.
            aic = 2 * (p+q+1) - 2 * dLL
            bic = (p+q+1) * np.log(iN) - 2 * dLL

            print("\nAIC = {} \nBIC = {}".format(
            aic, bic))

            #Sandwich estimator standard errors and reporting.
            SumNLnLReg= lambda vP: -np.mean(LnLRegNorm(vP, vY = vY, mX = mX, p = p, q = q))
            mH = -hessian_2sided(SumNLnLReg, vP = vP_MLE)
            mG= jacobian_2sided(LnLRegNorm,vP_MLE, vY, mX, p, q)
            mG2 = (mG.T @ mG) /iN
            mH_inv = np.linalg.inv(mH)
            mVhat = (mH_inv @ mG2 @ mH_inv)/iN
            print("\nSigma standard error: {}".format(np.sqrt(mVhat[0,0])) + "\nMu " 
            + "standard error: {}".format(vP_MLE[1]) + "\nFirst {} are AR "
            "standard errors, subsequent {} are MA standard errors :".format(p, q) 
            + "\n{}".format(np.diagonal(np.sqrt(mVhat[2:, 2:]))))


            #Following code is used for latex table construction
            standard_errors = np.sqrt(np.diagonal(mVhat))
            latex_output[0, column + shift] = vP_MLE[0]
            latex_output[1, column + shift] =  standard_errors[0]
            print("\nSigma t-value = " + str(vP_MLE[0]/standard_errors[0]))
            latex_output[2, column + shift] = vP_MLE[1]
            latex_output[3, column + shift] = standard_errors[1]
            print("\nMu t-value = " + str(vP_MLE[1]/standard_errors[1]))
            if p == 1:
                latex_output[4, column + shift] = vP_MLE[2]
                latex_output[5, column + shift] = standard_errors[2]
                print("\nPhi_1 t-value = " + str(vP_MLE[2]/standard_errors[2]))
            if p == 2:
                latex_output[4, column + shift] = vP_MLE[2]
                latex_output[5, column + shift] = standard_errors[2]
                print("\nPhi_1 t-value = " + str(vP_MLE[2]/standard_errors[2]))
                latex_output[6, column + shift] = vP_MLE[3]
                latex_output[7, column + shift] = standard_errors[3]
                print("\nPhi_2 t-value = " + str(vP_MLE[3]/standard_errors[3]))
            if (p ==1 and q == 1):
                latex_output[8, column + shift] = vP_MLE[3]
                latex_output[9, column + shift] = standard_errors[3]
                print("\nTheta_1 t-value = " + str(vP_MLE[3]/standard_errors[3]))
            if (p ==1 and q == 2):
                latex_output[8, column + shift] = vP_MLE[3]
                latex_output[9, column + shift] = standard_errors[3]
                print("\nTheta_1 t-value = " + str(vP_MLE[3]/standard_errors[3]))
                latex_output[10, column + shift] = vP_MLE[4]
                latex_output[11, column + shift] = standard_errors[4]
                print("\nTheta_2 t-value = " + str(vP_MLE[4]/standard_errors[4]))
            if (p ==2 and q == 1):
                latex_output[8, column + shift] = vP_MLE[4]
                latex_output[9, column + shift] = standard_errors[4]
                print("\nTheta_1 t-value = " + str(vP_MLE[4]/standard_errors[4]))
            if (p ==2 and q == 2):
                latex_output[8, column + shift] = vP_MLE[4]
                latex_output[9, column + shift] = standard_errors[4]
                print("\nTheta_1 t-value = " + str(vP_MLE[4]/standard_errors[4]))
                latex_output[10, column + shift] = vP_MLE[5]
                latex_output[11, column + shift] = standard_errors[5]
                print("\nTheta_2 t-value = " + str(vP_MLE[5]/standard_errors[5]))
            latex_output[12,column + shift] = dLL
            latex_output[13, column + shift] = aic
            latex_output[14,column + shift] = bic
            
            np.set_printoptions(precision=6, suppress = True)

            latex = a2l.to_ltx(latex_output, frmt = '{:.6f}', mathform = False)
            print(latex)
    #c)
    (vResiduals_DJIA, vResiduals_N225, vResiduals_SSMI) = CalcResiduals(df, lParameters, vSeriesNames)

    ACF_list = []
    PACF_list = []
    Residuals_list = [vResiduals_DJIA, vResiduals_N225, vResiduals_SSMI]
    for series in Residuals_list:
        ACF_list.append(tuple(ACF(series, 25)))
        PACF_list.append(tuple(PACF(series, 25)))

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(Residuals_list[0])
    ax1.set_title("DJIA AR(2)")
    ax2.plot(Residuals_list[1])
    ax2.set_title("N225 AR(2)")
    ax3.plot(Residuals_list[2])
    ax3.set_title("SSMI ARMA(1,2)")
    plt.tight_layout()
    plt.plot()

    x = list(range(0, 25))
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    ax1.bar(x, height = ACF_list[0])
    ax1.set_title("DJIA.Ret ACF")
    ax1.axhline(2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax1.axhline(-2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax3.bar(x, height = ACF_list[1])
    ax3.set_title("N225.Ret ACF")
    ax3.axhline(2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax3.axhline(-2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax5.bar(x, height = ACF_list[2])
    ax5.set_title("SSMI.Ret ACF")
    ax5.axhline(2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax5.axhline(-2/np.sqrt(3011), color = "r", linestyle = "dashed")
    ax2.bar(x, height = PACF_list[0])
    ax2.set_title("DJIA.Ret PACF")
    ax2.axhline(2/np.sqrt(3021), color = "r", linestyle = "dashed")
    ax2.axhline(-2/np.sqrt(3021), color = "r", linestyle = "dashed")
    ax4.bar(x, height = PACF_list[1])
    ax4.set_title("N225.Ret PACF")
    ax4.axhline(2/np.sqrt(3021), color = "r", linestyle = "dashed")
    ax4.axhline(-2/np.sqrt(3021), color = "r", linestyle = "dashed")
    ax6.bar(x, height = PACF_list[2])
    ax6.set_title("SSMI.Ret PACF")
    ax6.axhline(2/np.sqrt(3021), color = "r", linestyle = "dashed")
    ax6.axhline(-2/np.sqrt(3021), color = "r", linestyle = "dashed")
    plt.tight_layout()

    for vACF in ACF_list:
        LjungBox(vAutoCorrelation = vACF, iT = 3021)

    print("JB Statistic for DJIA, N225 and SSMI:")
    for vResidual in Residuals_list:
        print("\n"+str(JarqueBera(vResidual)))
            

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
            ax[i,j].set_title(rets[i]+', '+rets[j]) # loop that plots all the cross covariances
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
    params[9:] = beta_OLS[1:,2] # you can only pass an array of shape (n,) to sc.opt so gotta squish them in params...
    
    # now the ML routine...
    def log_lik_var1(y, params):
        # function that gets vector of log likelihood based on params for the var 1
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

    P_hat_var1 = cholesky_decomp(sigma_hat_var1) # get P hat
    P_hat_var2 = cholesky_decomp(sigma_hat_var2) 
    
    print('P_hat for VAR(1) = ')
    print(P_hat_var1.round(decimals=4))
    print('')
    
    print('P_hat for VAR(2) = ')
    print(cholesky_decomp(sigma_hat_var2).round(decimals=4))
    print('')
    
    Q5_ests = {'P_hat_var1':P_hat_var1, 'P_hat_var2':P_hat_var2, 'phi_hat_var1':phi_hat_var1,
               'phi1': phi1, 'phi2':phi2} # needed for Q5 orthogonal IRF's
    
    
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
        store1[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection1, (3,)) # implement var1 IRF formula
        store2[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection2, (3,))
        store3[i-1,:] = np.reshape(estimates['phi_hat_var1']**(i-1)@estimates['P_hat_var1']@selection3, (3,))
        
    fig, ax = plt.subplots(3,3, figsize= (15,10))
    ax[0,0].plot(store1[:,0])
    ax[0,0].set_title('Effect of DJIA shock on DJIA') # plot them ALL
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
    plt.show() # looks pretty good right
    
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
            store1[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection1 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,)) # add second phi for h>1
            store2[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection2 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,))
            store3[i-1,:] = np.reshape(estimates['phi1']**(i-1)@estimates['P_hat_var2']@selection3 + estimates['phi2']**(i-1)@estimates['P_hat_var2']@selection1, (3,))


    fig, ax = plt.subplots(3,3, figsize= (15,10))
    ax[0,0].plot(store1[:,0])
    ax[0,0].set_title('Effect of DJIA shock on DJIA') # plot them
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
    plt.show() # okay not bad

    return

###############################################################################
### output_Q6
def output_Q6(df):
    """

    Parameters
    ----------
    df : dataframe of returns and closing prices

    Returns
    -------
    None.

    """
    print('Question 6a: ')
    print('')
    df = df.iloc[1:,:]
    # define data:
    y = np.array(df.iloc[:,1:4].T)
    dy = np.array(df.iloc[:,4:].T)
    
    # okay start with VECM(1)
    ylag1 = y[:,:-1]
    dyt = dy[:,1:]
    
    # initialize parameters
    params = np.zeros(12) # initialize, just took zeros because of time constraints
    
    
    def log_lik_vecm1(dyt, ylag1, params):
        mu = np.reshape(params[0:3], (3,1))
        PI = np.reshape(params[3:12], (3,3))
        
        eps = dyt - PI@ylag1 - mu
        sigma = (eps@eps.T)/len(dyt.T)

        LLs = np.empty(len(dyt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(dyt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]

        return LLs
    
    print('Fitting a VECM(1): ')
    print('')
    AvgNLL = lambda params : -np.mean(log_lik_vecm1(dyt, ylag1, params)) # define function to be minimized 
    res_vecm1   = opt.minimize(AvgNLL, params, method='Powell') # algos that work: Powell, SLSQP
    print(res_vecm1.message)
    mu_hat = np.reshape(res_vecm1.x[0:3], (3,1)) # get ests from res
    PI_hat1 = np.reshape(res_vecm1.x[3:12], (3,3))
    eps = dyt - PI_hat1@ylag1 - mu_hat
    sigma_hat = (eps@eps.T)/len(dyt.T)
    
    print('mu hat = ', mu_hat.round(decimals=4))
    print('')
    print('PI hat = ', PI_hat1.round(decimals=4))
    print('')
    print('sigma hat = ', sigma_hat.round(decimals=4))
    print('')
    
    m = 18
    T = len(dy[:,1:].T)
    
    AIC_vecm1 = np.log(np.linalg.det(sigma_hat)) + 2*m/T # get information criteria
    AICc_vecm1 = np.log(np.linalg.det(sigma_hat)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_vecm1 = np.log(np.linalg.det(sigma_hat)) + m*np.log(T)/T
    
    criterions_vecm1 = np.array([AIC_vecm1, AICc_vecm1, BIC_vecm1])
    print('criterions of VECM(1) (AIC, AICc, BIC) are ', criterions_vecm1.round(decimals=4))
    print('')

    # okay now for VECM2
    # define series used
    dyt = dy[:,1:]
    dylag1 = dy[:,:-1]
    ylag1 = y[:,:-1]
    params = np.zeros(21)

    def log_lik_vecm2(dyt, dylag1, ylag1, params):
        mu = np.reshape(params[0:3], (3,1))
        PI = np.reshape(params[3:12], (3,3))
        Gamma1 = np.reshape(params[12:], (3,3))
        
        eps = dyt - PI@ylag1 - Gamma1@dylag1 - mu
        sigma = (eps@eps.T)/len(dyt.T)

        LLs = np.empty(len(dyt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(dyt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]
            
        return LLs
    print('Fitting a VECM(2): ')
    print('')
    AvgNLL = lambda params : -np.mean(log_lik_vecm2(dyt, dylag1, ylag1, params))
    res_vecm2 = opt.minimize(AvgNLL, params, method='Powell')
    print(res_vecm2.message)
    mu_hat = np.reshape(res_vecm2.x[0:3], (3,1))
    PI_hat2 = np.reshape(res_vecm2.x[3:12], (3,3))
    Gamma1_hat = np.reshape(res_vecm2.x[12:], (3,3))
    
    eps = dyt - PI_hat2@ylag1 - Gamma1_hat@dylag1 - mu_hat
    sigma_hat = (eps@eps.T)/len(dyt.T)
    print('mu hat = ', mu_hat.round(decimals=4))
    print('')
    print('PI hat = ', PI_hat2.round(decimals=4))
    print('')
    print('Gamma1 hat = ', Gamma1_hat.round(decimals=4))
    print('')
    print('sigma hat = ', sigma_hat.round(decimals=4))
    print('')
    
    m = 27
    T = len(dy[:,1:].T)
    
    AIC_vecm2 = np.log(np.linalg.det(sigma_hat)) + 2*m/T
    AICc_vecm2 = np.log(np.linalg.det(sigma_hat)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_vecm2 = np.log(np.linalg.det(sigma_hat)) + m*np.log(T)/T
    
    criterions_vecm2 = np.array([AIC_vecm2, AICc_vecm2, BIC_vecm2])
    print('criterions of VECM(2) (AIC, AICc, BIC) are ', criterions_vecm2.round(decimals=4))
    print('')
    
    
    # okay now for VECM3
    # define series used
    dyt = dy[:,2:]
    dylag1 = dy[:,1:-1]
    dylag2 = dy[:,:-2]
    ylag1 = y[:,1:-1]
    
    params = np.zeros(30)
    
    def log_lik_vecm3(dyt, dylag1, dylag2, ylag1, params):
        mu = np.reshape(params[0:3], (3,1))
        PI = np.reshape(params[3:12], (3,3))
        Gamma1 = np.reshape(params[12:21], (3,3))
        Gamma2 = np.reshape(params[21:], (3,3))
        
        eps = dyt - PI@ylag1 - Gamma1@dylag1 - Gamma2@dylag2 - mu
        sigma = (eps@eps.T)/len(dyt.T)

        LLs = np.empty(len(dyt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(dyt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]
            
        return LLs
    print('Fitting a VECM(3): ')
    print('')
    AvgNLL = lambda params : -np.mean(log_lik_vecm3(dyt, dylag1, dylag2, ylag1, params))
    res_vecm3 = opt.minimize(AvgNLL, params, method='Powell')
    print(res_vecm3.message)
    mu_hat = np.reshape(res_vecm3.x[0:3], (3,1))
    PI_hat3 = np.reshape(res_vecm3.x[3:12], (3,3))
    Gamma1_hat = np.reshape(res_vecm3.x[12:21], (3,3))
    Gamma2_hat = np.reshape(res_vecm3.x[21:], (3,3))
    
    eps = dyt - PI_hat3@ylag1 - Gamma1_hat@dylag1 - Gamma2_hat@dylag2 - mu_hat
    sigma_hat = (eps@eps.T)/len(dyt.T)
    print('mu hat = ', mu_hat.round(decimals=4))
    print('')
    print('PI hat = ', PI_hat3.round(decimals=4))
    print('')
    print('Gamma1 hat = ', Gamma1_hat.round(decimals=4))
    print('')
    print('Gamma2 hat = ', Gamma2_hat.round(decimals=4))
    print('')
    print('sigma hat = ', sigma_hat.round(decimals=4))
    print('')
    
    m = 36
    T = len(dy[:,1:].T)
    
    AIC_vecm3 = np.log(np.linalg.det(sigma_hat)) + 2*m/T
    AICc_vecm3 = np.log(np.linalg.det(sigma_hat)) + 2*m/T*((T+2*m)/(T-m-1)) 
    BIC_vecm3 = np.log(np.linalg.det(sigma_hat)) + m*np.log(T)/T
    
    criterions_vecm3 = np.array([AIC_vecm3, AICc_vecm3, BIC_vecm3])
    print('criterions of VECM(2) (AIC, AICc, BIC) are ', criterions_vecm3.round(decimals=4))
    print('')
    
    print('Done fitting all the models!')
    print('')
    
    
    
    # question 4b
    print('Question 6b: ')
    # first get LR for all PI's...
    eig1 = np.linalg.eig(PI_hat1)[0] # get eigenvalues
    eig2 = np.linalg.eig(PI_hat2)[0]
    eig3 = np.linalg.eig(PI_hat3)[0]
    
    LR = np.empty((3,3))
    
    for i in range(3):
        sumk1 = np.sum(np.log(np.ones(3-i)-eig1[i:])) # get Johansen trace test statistic,
        LR1 = -(len(dy.T)-1)*sumk1
        sumk2 = np.sum(np.log(np.ones(3-i)-eig2[i:])) # we suspect the obtained statistics are wrong :(, but we don't know why
        LR2 = -(len(dy.T)-2)*sumk2
        sumk3 = np.sum(np.log(np.ones(3-i)-eig3[i:]))
        LR3 = -(len(dy.T)-3)*sumk3   
        
        LR[:,i] = np.array([LR1, LR2, LR3])
    
    print('LR test statistics = ',LR)
    
    return



#%%
###########################################################
### main
def main():
    # magic numbers
    path = r"triv_ts.txt"
    df = loadin_data(path)
    
    # now call the functions that print all of the output for all questions
    output_Q1(df)
    Output_Q2(df)
    estimates = output_Q4(df)
    output_Q5(estimates)
    output_Q6(df)

###########################################################
### start main
if __name__ == "__main__":
    main()

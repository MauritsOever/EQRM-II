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
import statsmodels.tsa.stattools as st
import array_to_latex as a2l
import scipy.optimize as opt





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
    None.

    """
    rets = ['DJIA.Ret', 'N225.Ret', 'SSMI.Ret']
    df = df[rets][1:] #reduce size of df for easier looping 
    
    # Q4 a
    
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
                print(len(ret2))
    
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
    # estimating the VAR(1) and VAR(2) model by ML
    
    # start with VAR(1) 
    def log_lik(y, phi, mu):
        yt = y[:,1:]
        ylag1 = y[:,:-1]
        
        eps = yt - phi@ylag1 - mu
        
        sigma = (eps@eps.T)/len(yt.T)
        LLs = np.empty(len(yt.T))
        
        for t in range(len(LLs)):
            LLs[t] = -(3*len(yt.T)/2)*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(sigma)) -0.5*eps[:,t].T@np.linalg.inv(sigma)@eps[:,t]
        
        return LLs
    
    mu = np.zeros((3,1))
    phi = np.zeros((3,3))
    y = np.array(df[rets][1:]).T
    
    params = [mu, phi]
    
    sc 
    
    
    
    return

###########################################################
### main
def main():
    # magic numbers
    path = r"triv_ts.txt"
    df = loadin_data(path)
    
    output_Q1(df)
    Output_Q2(df)


###########################################################
### start main
if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
EQRM II
Assignment 1 main

Created on Mon Nov 29 16:00:04 2021

@author: Donald Hagestein, Connor Stevens and Maurits van den Oever
"""

###############################################################################
# import pachages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import datetime as dt
from numpy.ma.core import concatenate
from scipy.special import loggamma
import scipy.optimize as opt
import pandas as pd
from scipy.stats import t
import scipy
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(suppress=True)
###############################################################################



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

    df = df*100
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
            self.vSig2tPlus1 = [Sig2Hat(self.dOmega,
            self.dBeta,
            self.dAlpha,
            x,
            self.dMu,
            self.dDelta,
            self.dSig2Initial,
            self.dLambda) for x in self.vX]

    #Pre-determined lambda and delta values.
    vLambda = np.array([2, 5, 10, 50])
    vDelta = np.array([0, 0.1, 0.2, 0.4])
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




def output_Q3(data):
    """
    Function that handles all output for question 3

    Parameters
    ----------
    data : Dataframe of returns

    Returns
    -------
    None.

    """
    print("\nQuestion 3: \n")
    #### functions inside of functions ####
    def printMatrix(s):
        # Do heading
        print("      ", end="")
        for j in range(len(s[0])):
            print("%7d " % j, end="")
        print()
        print("     ", end="")
        for j in range(len(s[0])):
            print("------", end="")
        print()
        # Matrix contents
        for i in range(len(s)):
            print("%3d |" % (i), end="")  # Row nums
            for j in range(len(s[0])):
                print("  ", round(s[i][j], 5), end="")
            print()
            
    def variance(serie):
        n=len(serie)
        average=np.sum(serie)/n
        serie=serie-average
        serie=np.square(serie)
        return np.sum(serie)/n
    
    def indicator(value):
        placeholder=0
        if value<0:
            placeholder=1
        return placeholder
    
    def t_logLikelihood(error,vega):
        a=loggamma((vega+1)/2)
        b=loggamma(vega/2)+0.5*np.log(vega*np.pi)
        c=(-(vega+1)/2)*np.log(1+(error**2)/vega)
        return (a-b)+c
    
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
    
    def transformParameters_withLeverage(subparameters):
        parameters=np.zeros(len(subparameters))
        parameters[0] = subparameters[0]
        parameters[1] = np.exp(subparameters[1])
        parameters[2] = (1+np.exp(-subparameters[2]))**-1
        parameters[3] = (1+np.exp(-subparameters[3]))**-1
        parameters[4] = -1+2*(1+np.exp(-subparameters[4]))**-1
        parameters[5] = np.exp(subparameters[5])
        return parameters
    
    def transformParameters_withoutLeverage(subparameters):
        parameters=np.zeros(len(subparameters))
        parameters[0] = subparameters[0]
        parameters[1] = np.exp(subparameters[1])
        parameters[2] = (1+np.exp(-subparameters[2]))**-1
        parameters[3] = (1+np.exp(-subparameters[3]))**-1
        parameters[4] = np.exp(subparameters[4])
        return parameters
    
    def RobustGARCHwithLeverageEffect(subparameters,serie,average):
        parameters=transformParameters_withLeverage(subparameters)
        x=serie
        n=len(x)
        mu = parameters[0]
        omega = parameters[1]
        alpha = parameters[2]
        beta = parameters[3]
        delta = parameters[4]
        vega = parameters[5]
        sigma_sq1=variance(x[0:50])
        vSigma_sq=np.zeros(n)
        vSigma_sq[0]=sigma_sq1
        vLlik=np.zeros(n)
        vLlik[0]=t_logLikelihood((x[0]-mu)/(vSigma_sq[0]**0.5),vega)-0.5*np.log(vSigma_sq[0])
        i=1
        while i<n:
            vSigma_sq[i]=omega+(beta+(alpha*((x[i-1]-mu)**2)+delta*((x[i-1]-mu)**2)*indicator(x[i-1]))/(vSigma_sq[i-1]+(1/vega)*((x[i-1]-mu)**2)))*vSigma_sq[i-1]
            vLlik[i]=t_logLikelihood((x[i]-mu)/(vSigma_sq[i]**0.5),vega)-0.5*np.log(vSigma_sq[i])
            i=i+1
    
        if average==True:
            return -np.mean(vLlik)
        else:
            return vLlik
    
    def RobustGARCHwithoutLeverageEffect(subparameters,serie,average):
        parameters=transformParameters_withoutLeverage(subparameters)
        x=serie
        n=len(x)
        mu = parameters[0]
        omega = parameters[1]
        alpha = parameters[2]
        beta = parameters[3]
        vega = parameters[4]
        sigma_sq1=variance(x[0:50])
        vSigma_sq=np.zeros(n)
        vSigma_sq[0]=sigma_sq1
        vLlik=np.zeros(n)
        vLlik[0]=t_logLikelihood((x[0]-mu)/(vSigma_sq[0]**0.5),vega)-0.5*np.log(vSigma_sq[0])
        i=1
        while i<n:
            vSigma_sq[i]=omega+(beta+(alpha*((x[i-1]-mu)**2))/(vSigma_sq[i-1]+(1/vega)*((x[i-1]-mu)**2)))*vSigma_sq[i-1]
            vLlik[i]=t_logLikelihood((x[i]-mu)/(vSigma_sq[i]**0.5),vega)-0.5*np.log(vSigma_sq[i])
            i=i+1
    
        if average==True:
            return -np.mean(vLlik)
        else:
            return vLlik
    
    
    def estimate_RobustGARCHwithLeverageEffect(serie):
        n=len(serie)
        x0 = [0]*6
        average=True
        res = scipy.optimize.minimize(RobustGARCHwithLeverageEffect,x0,args=(serie,average),method='SLSQP')
        subparameters=res.x
        parameters=transformParameters_withLeverage(subparameters)
    
        llik_sum=np.sum(RobustGARCHwithLeverageEffect(subparameters,serie,False))
        llik_avg=np.mean(RobustGARCHwithLeverageEffect(subparameters,serie,False))
        k=5
        AIC=2*k-2*llik_sum
        BIC=k*np.log(n)-2*llik_sum
    
        llikAVG = lambda subparameters: np.mean(RobustGARCHwithLeverageEffect(subparameters,serie=serie,average=False))
        llikVector = lambda subparameters: RobustGARCHwithLeverageEffect(subparameters, serie=serie, average=False)
        parameterTransformer = lambda subparameters: transformParameters_withLeverage(subparameters)
        H = -hessian_2sided(llikAVG,vP=subparameters)
        G = jacobian_2sided(llikVector, vP=subparameters)
        G2 = (G.T @ G) / n
        H_inv = np.linalg.inv(H)
        Vhat = (H_inv @ G2 @ H_inv) / n
        subparameterVariance=Vhat
        K=jacobian_2sided(parameterTransformer, vP=subparameters)
        parameterVariance=K @ subparameterVariance @K.T
        parameters_STE=np.sqrt(np.diagonal(parameterVariance))
    
        # print(parameters)
        # print(parameters_STE)
    
        output=np.zeros(15)
        output[0]=parameters[0]
        output[1]=parameters_STE[0]
        output[2]=parameters[1]
        output[3]=parameters_STE[1]
        output[4]=parameters[2]
        output[5]=parameters_STE[2]
        output[6]=parameters[3]
        output[7]=parameters_STE[3]
        output[8]=parameters[4]
        output[9]=parameters_STE[4]
        output[10]=parameters[5]
        output[11]=parameters_STE[5]
        output[12] = llik_sum
        output[13] = AIC
        output[14] = BIC
        return output
    
    def estimate_RobustGARCHwithoutLeverageEffect(serie):
        n=len(serie)
        x0 = [0]*5
        average=True
        res = scipy.optimize.minimize(RobustGARCHwithoutLeverageEffect,x0,args=(serie,average),method='SLSQP')
        subparameters=res.x
        parameters=transformParameters_withoutLeverage(subparameters)
    
        llik_sum=np.sum(RobustGARCHwithoutLeverageEffect(subparameters,serie,False))
        llik_avg=np.mean(RobustGARCHwithoutLeverageEffect(subparameters,serie,False))
        k=4
        AIC=2*k-2*llik_sum
        BIC=k*np.log(n)-2*llik_sum
    
        llikAVG = lambda subparameters: np.mean(RobustGARCHwithoutLeverageEffect(subparameters,serie=serie,average=False))
        llikVector = lambda subparameters: RobustGARCHwithoutLeverageEffect(subparameters, serie=serie, average=False)
        parameterTransformer = lambda subparameters: transformParameters_withoutLeverage(subparameters)
        H = -hessian_2sided(llikAVG,vP=subparameters)
        G = jacobian_2sided(llikVector, vP=subparameters)
        G2 = (G.T @ G) / n
        H_inv = np.linalg.inv(H)
        Vhat = (H_inv @ G2 @ H_inv) / n
        subparameterVariance=Vhat
        K=jacobian_2sided(parameterTransformer, vP=subparameters)
        parameterVariance=K @ subparameterVariance @K.T
        parameters_STE=np.sqrt(np.diagonal(parameterVariance))
    
        # print(parameters)
        # print(parameters_STE)
    
        output=np.zeros(15)
        output[0]=parameters[0]
        output[1]=parameters_STE[0]
        output[2]=parameters[1]
        output[3]=parameters_STE[1]
        output[4]=parameters[2]
        output[5]=parameters_STE[2]
        output[6]=parameters[3]
        output[7]=parameters_STE[3]
        output[8]=None
        output[9]=None
        output[10]=parameters[4]
        output[11]=parameters_STE[4]
        output[12] = llik_sum
        output[13] = AIC
        output[14] = BIC
        return output
    
    def question3_subTable(serie):
        table=np.zeros((15,2))
        table[:, 0] = estimate_RobustGARCHwithoutLeverageEffect(serie)
        table[:, 1] = estimate_RobustGARCHwithLeverageEffect(serie)
        return table
    
    #q3 body here
    data=data[:2500].to_numpy()
    data=data*100
    print('len data: ',len(data))
    stock1 = data[:, 0]
    stock2 = data[:, 1]
    stock3 = data[:, 2]

    table=np.zeros((15,6))
    table[:,0:2] = question3_subTable(stock1)
    table[:,2:4] = question3_subTable(stock2)
    table[:,4:6] = question3_subTable(stock3)

    printMatrix(table)
    df = pd.DataFrame(table)
    writer = pd.ExcelWriter(r'C:\Users\gebruiker\Documents\GitHub\EQRM-II\iets.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    
    return table


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
    print('Question 4: ')
    print('')
    print('Plots: ')
    df = df*100
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
        number = i*2
        mu = estimates[0, number]
        omega = estimates[2, number] 
        beta = estimates[6, number]
        alpha = estimates[4, number]
        delta = estimates[8, number]
        Lambda = estimates[10, number]
        sigma1 = estimates[11, number]
        
        mu_lev = estimates[0, number+1]
        omega_lev = estimates[2, number+1]
        beta_lev = estimates[6, number+1]
        alpha_lev = estimates[4, number+1]
        delta_lev = estimates[8, number+1]
        Lambda_lev = estimates[10, number+1]
        sigma1_lev = estimates[11, number+1]
        
        volas_lev[0,i] = sigma1_lev
        volas_nolev[0,i] = sigma1
        
        for j in range(1, len(volas_lev)):
            volas_lev[j,i] = sigma_t(df.iloc[j-1,i], mu_lev, omega_lev, beta_lev, alpha_lev, delta_lev, Lambda_lev, volas_lev[j-1,i], True)
            volas_nolev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_nolev[j-1,i], False)
        
        # get range of xt...
        # get answers of sigmat+1 based on this xt, for lev and no lev
        # plot then
        rangext = np.linspace(-50, 50, 1000)
        NIC_lev = np.empty((len(rangext),))
        NIC_nolev = np.empty((len(rangext),))
        for t in range(len(rangext)):
            NIC_lev[t] = sigma_t(rangext[t], mu_lev, omega_lev, beta_lev, alpha_lev, delta_lev, Lambda_lev, volas_lev[2500,i], True)
            NIC_nolev[t] = sigma_t(rangext[t], mu, omega, beta, alpha, delta, Lambda, volas_lev[2500,i], False)     
            
        
        ax[i, 0].plot(rangext, NIC_lev) # NIC here...
        ax[i, 0].plot(rangext, NIC_nolev)
        ax[i, 0].set_xlabel('xt')
        ax[i, 0].set_ylabel('sigma2_t+1')
        ax[i, 0].legend(['leverage','no leverage'])
        ax[i, 0].set_title('NIC for ' + df.columns[i].replace('_ret', ''))
        # ax labels, legend
        
        # title, axlabels, whatevs
        ax[i, 1].plot(df.index[5:], volas_lev[5:,i]) # and maybe nolev as well??
        ax[i, 1].plot(df.index[5:], volas_nolev[5:,i])
        ax[i, 1].axvline(pd.Timestamp('2009-12-10'), color='black', linestyle='--', linewidth=1)
        ax[i, 1].set_xlabel('sigma2')
        ax[i, 1].legend(['leverage', 'no leverage'])
        ax[i, 1].set_title('Estimated and forecasted volatilities for ' + df.columns[i].replace('_ret', ''))

        
        
    plt.tight_layout()
    plt.show()
    
    return 

###########################################################
### output_Q5
def output_Q5(df, estimates):
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
    # okay so:
    # sim 1, 5 and 20 eps
    # get volas for both models, for all series...
    # fucc mee xdd
    # then sim volas with eps
    # then do that 10k times
    # then get 1%, 5%, and 10%
    
    # magic numbers:
    df = df*100
    simsize = 10000
    
    print('Question 5: ')
    print('')
    
    def sigma_t(xlag1, mu, omega, beta, alpha, delta, Lambda, sigmalag1, lev):
        # calculates the next sigma based on the xt and params
        # every param is scalar, but lev is a bool
        diffsq = (xlag1-mu)**2
        if diffsq < 0:
            print('the error is here')
        
        if xlag1 > 0:
            delta = 0
        
        if lev == False:
            delta = 0
        sigma_t = omega + (beta + ((alpha*diffsq + delta*diffsq)/(sigmalag1 + (1/Lambda)*diffsq)))*sigmalag1
        
        return sigma_t
    
    def x_t(sigmat, epst, mu):
        x_t = mu + np.sqrt(sigmat)*epst
        return x_t
        
    numbers = np.empty((6,9))
    
    
    for i in range(3):
        number = i*2
        mu = estimates[0, number]
        omega = estimates[2, number] 
        beta = estimates[6, number]
        alpha = estimates[4, number]
        delta = estimates[8, number]
        Lambda = estimates[10, number]
        sigma1 = estimates[11, number]
        
        mu_lev = estimates[0, number+1]
        omega_lev = estimates[2, number+1]
        beta_lev = estimates[6, number+1]
        alpha_lev = estimates[4, number+1]
        delta_lev = estimates[8, number+1]
        Lambda_lev = estimates[10, number+1]
        sigma1_lev = estimates[11, number+1]
                
        volas_lev = np.empty((len(df), 3)) # volatilities with leverage
        volas_nolev = np.empty((len(df), 3)) # volatilities without leverage

        volas_lev[0,i] = sigma1_lev
        volas_nolev[0,i] = sigma1
        
        for j in range(1, len(volas_lev)):
            volas_lev[j,i] = sigma_t(df.iloc[j-1,i], mu_lev, omega_lev, beta_lev, alpha_lev, delta_lev, Lambda_lev, volas_lev[j-1,i], True)
            volas_nolev[j,i] = sigma_t(df.iloc[j-1,i], mu, omega, beta, alpha, delta, Lambda, volas_nolev[j-1,i], False)
        
        # get index for 2020 april 1 in df
        index = list(df.index).index(pd.Timestamp('2020-04-01'))
        
        sigmat_init_lev = volas_lev[index, i]
        sigmat_init_nolev = volas_nolev[index, i]
        xt_init = df.iloc[index, i]
        print(xt_init)
            
        sims_sigma_lev = np.full((simsize, 21), sigmat_init_lev)
        sims_sigma_nolev = np.full((simsize, 21), sigmat_init_nolev)
        sims_xt_lev = np.full((simsize, 21), xt_init)
        sims_xt_nolev = np.full((simsize, 21), xt_init)
        
        for n in range(simsize):
            eps = sc.t.rvs(Lambda, 0, 1, 21) # perhaps simulated wrong...
            for j in range(1,21):
                sims_sigma_lev[n,j] = sigma_t(sims_xt_lev[n,j-1], mu_lev, omega_lev, beta_lev, alpha_lev, delta_lev, Lambda_lev, sims_sigma_lev[n, j-1], True)
                sims_sigma_nolev[n,j] = sigma_t(sims_xt_nolev[n,j-1], mu, omega, beta, alpha, delta, Lambda, sims_sigma_lev[n, j-1], False)
                sims_xt_lev[n,j] = mu + np.sqrt(sims_sigma_lev[n,j])*eps[j]
                sims_xt_nolev[n,j] = mu + np.sqrt(sims_sigma_nolev[n,j])*eps[j]
        
        sims_xt_lev *= (1/100)    
        sims_xt_nolev *= (1/100)    
        
        xt_h1_lev = sims_xt_lev[:,1]
        xt_h5_lev = (np.apply_along_axis(np.product, 1, (sims_xt_lev[:,1:6]+1))) -1
        xt_h20_lev = (np.apply_along_axis(np.product, 1, (sims_xt_lev[:,1:21]+1))) -1
        
        xt_h1_nolev = sims_xt_nolev[:,1]
        xt_h5_nolev = (np.apply_along_axis(np.product, 1, (sims_xt_nolev[:,1:6]+1))) -1
        xt_h20_nolev = (np.apply_along_axis(np.product, 1, (sims_xt_nolev[:,1:21]+1))) -1
        
        # print('for series', df.columns[i].replace('_ret','')) # this statement and loop print output for Q5
        # for q in [1,5,10]:
        #     print('leveraged GARCH model: ')
        #     print('for', q, '%, VaR for h = 1, 5, 20:', np.quantile(xt_h1_lev, q/100), np.quantile(xt_h5_lev, q/100), np.quantile(xt_h20_lev, q/100))
        #     print('')
        #     print('unleveraged GARCH model: ')
        #     print('for', q, '%, VaR for h = 1, 5, 20:', np.quantile(xt_h1_nolev, q/100), np.quantile(xt_h5_nolev, q/100), np.quantile(xt_h20_nolev, q/100))
        #     print('')
            
        # store numbers into big object, then print as table...
        numb = i*2
        numbers[numb,:] = np.array([np.quantile(xt_h1_lev, 0.01), np.quantile(xt_h1_lev, 0.05), np.quantile(xt_h1_lev, 0.1),
                                 np.quantile(xt_h5_lev, 0.01), np.quantile(xt_h5_lev, 0.05), np.quantile(xt_h5_lev, 0.1),
                                 np.quantile(xt_h20_lev, 0.01), np.quantile(xt_h20_lev, 0.05), np.quantile(xt_h20_lev, 0.1)])
        numbers[numb+1,:] = np.array([np.quantile(xt_h1_nolev, 0.01), np.quantile(xt_h1_nolev, 0.05), np.quantile(xt_h1_nolev, 0.1),
                                 np.quantile(xt_h5_nolev, 0.01), np.quantile(xt_h5_nolev, 0.05), np.quantile(xt_h5_nolev, 0.1),
                                 np.quantile(xt_h20_nolev, 0.01), np.quantile(xt_h20_nolev, 0.05), np.quantile(xt_h20_nolev, 0.1)])
    
    numbers = numbers*100
    # this block of code prints a latex table for Q5
    print(' & & h=1 & & & h=5 & & & h=20 & & \\\ ')
    print('q =  & & 0.01 & 0.05 & 0.10 & 0.01 & 0.05 & 0.10 & 0.01 & 0.05 & 0.10 \\\ ')
    print(' \\midrule ')
    print(df.columns[0].replace('_ret',''), ' & $\\delta \\neq 0 $ &',  '&'.join([str(entry) for entry in numbers[0,:].round(decimals=4)]), ' \\\ ')
    print(' & $ \\delta = 0 $ &', '&'.join([str(entry) for entry in numbers[1,:].round(decimals=4)]), '\\\ ')
    print('&&&&&&&&&& \\\ ')
    print(df.columns[1].replace('_ret',''), ' & $\\delta \\neq 0 $ &',  '&'.join([str(entry) for entry in numbers[2,:].round(decimals=4)]), '\\\ ')
    print(' & $ \\delta = 0 $ &', '&'.join([str(entry) for entry in numbers[3,:].round(decimals=4)]), '\\\ ')
    print('&&&&&&&&&& \\\ ')
    print(df.columns[2].replace('_ret',''), ' & $\\delta \\neq 0 $ &',  '&'.join([str(entry) for entry in numbers[4,:].round(decimals=4)]), '\\\ ')
    print(' & $ \\delta = 0 $ &', '&'.join([str(entry) for entry in numbers[5,:].round(decimals=4)]), '\\\ ')

    return

###########################################################
### output_Q7
def output_Q7(df_real):
    """
    Function that prints output for question 7

    Args:
        df (DataFrame): dataframe of returns
    """
    print("\nQuestion 7: \n")
    ################################################################################
    ### Sigma_tplus1_calculation(vX_t, dBeta, mOmega, mA, dLambda, mSigma_t) = mSigma_tplus1
    def Sigma_tplus1_calculation(vX_t, dBeta, mOmega, mA, dLambda, mSigma_t):
        #Force column vector.
        vX_t.shape = (3,1)

        #Split into parts for debugging
        mTerm1 = (1 - dBeta) * mOmega

        mTerm2_numerator = vX_t @ vX_t.T

        dTerm2_denominator = 1 + (vX_t.T @ np.linalg.inv(mSigma_t) @ vX_t) * dLambda**-1

        mTerm2 = mA @ ((mTerm2_numerator / dTerm2_denominator) - mSigma_t) @ mA.T

        mTerm3 = dBeta * mSigma_t

        mSigma_tplus1 = mTerm1 + mTerm2 + mTerm3

        return mSigma_tplus1

    ################################################################################
    ### Multivariate_t_log_likelihood(dLambda, iK, mSigma_t, vX_t) = dLog_likelihood
    def Multivariate_t_log_likelihood(dLambda, iK, mSigma_t, vX_t):
        #Force column vector.
        vX_t.shape = (3,1)

        # #Break up into terms for debugging.
        # dTerm1 = loggamma(1/2 * (dLambda + iK)) - loggamma(dLambda/2) 
        # dTerm2 = - 1/2 * np.log(np.linalg.det(np.pi * dLambda * mSigma_t))
        # dTerm3 = -1/2 * (dLambda + iK) * np.log(1 + dLambda**-1 * (vX_t.T @ np.linalg.inv(mSigma_t) @ vX_t))

        #Break up into terms for debugging.
        dTerm1 = loggamma(1/2 * (dLambda + iK))
        dTerm2 = -loggamma(dLambda/2)
        dTerm3 = -iK/2 * np.log(dLambda * np.pi)
        dTerm4 = -1/2 * np.log(np.linalg.det(mSigma_t))
        dTerm5 = -1/2 * (dLambda + iK) * np.log(1 + dLambda**-1 * (vX_t.T @ np.linalg.inv(mSigma_t) @ vX_t))

        #Sum for final log-likelihood contribution.
        dLog_likelihood = dTerm1 + dTerm2 + dTerm3 + dTerm4 + dTerm5

        return dLog_likelihood

    ################################################################################
    ### Parametrize(vTheta, sFrom_to, dModel) = vTrue_parameters = [dBeta, dLambda, vA]
    def Parametrize(vTheta):

        #Save as array for easy combining later.
        dBeta = (1 + np.exp(-vTheta[0]))**-1
        dLambda = np.exp(vTheta[1])

        #Model specification 1.
        if len(vTheta) == 3:
            #Save as array for easy combining later.
            vA_parameters = [np.array((1 + np.exp(-vTheta[2]))**-1)]

        #Model specification 2.
        if len(vTheta) == 5:
            #Save as array for easy combining later.
            vA_parameters = np.array((1 + np.exp(-vTheta[2:]))**-1)
        
        #Model specification 3.
        if len(vTheta) == 8:
            
            #Masks for indices of diagonal and off-diagonal elements in flattened
            #array version of A-matrix.
            vDaig_mask = np.array([0, 2, 5])
            vOff_diag_mask = np.array([1, 3, 4])

            #Different treatment for diagonal and off-diagonal elements of A-matrix.
            vA_subparameters = vTheta[2:]

            #Inf used to highlight errors in the indexing.
            vA_parameters = np.full_like(vTheta[2:], fill_value = np.inf)
            
            #Diagonal treatment.
            vA_parameters[vDaig_mask] = (1 + np.exp(-vA_subparameters[vDaig_mask]))**-1

            #Off-diagonal treatment.
            vA_parameters[vOff_diag_mask] = 1/3 * (-1 + 2 / (1 + np.exp(-vA_subparameters[vOff_diag_mask])))

        #Return parameters in the order and form they were provided.
        vParams_parametrized = np.insert(vA_parameters, 0, [dBeta, dLambda])
        
        return vParams_parametrized

    ################################################################################
    ###Log_likelihood_function(
    # vTheta, mXtilde, iK, iN, mOmega, mSigma_starting) = vLog_likelihood_contributions
    def Log_likelihood_function(vTheta, mXtilde, iK, iN, mOmega, mSigma_starting):

        if len(vTheta) == 3:
            #Re-parametrized  model parameters prior to optimisation.
            vTheta_new = Parametrize(vTheta)
            dBeta = vTheta_new[0]
            dLambda = vTheta_new[1]
            dA11 = vTheta_new[2]

            #Pre-specified A-matrix A11 * I.
            mA = dA11 * np.identity(3)
        
        if len(vTheta) == 5:
            #Re-parametrized  model parameters prior to optimisation.
            vTheta_new = Parametrize(vTheta)
            dBeta = vTheta_new[0]
            dLambda = vTheta_new[1]
            vA_flat = vTheta_new[2:]
            
            #Pre-specified diagonal A-matrix.
            mA = np.diagflat(vA_flat)

        if len(vTheta) == 8:
            #Re-parametrized  model parameters prior to optimisation.
            vTheta_new = Parametrize(vTheta)
            dBeta = vTheta_new[0]
            dLambda = vTheta_new[1]

            #Vector for holding flattened A-matrix.
            vA_flat = np.zeros(9)

            #Index mask for positions of lower-triangular elements.
            vIndex_mask = [0, 3, 4, 6, 7, 8]

            #Values to be put in lower triangular matrix extracted from vTheta.
            vA_lower_triangular = vTheta_new[2:]

            #Places values from vA_lower_triangular into positions in vIndex_mask.
            for iCount in range(0, len(vA_lower_triangular)):
                vA_flat[vIndex_mask[iCount]] = vA_lower_triangular[iCount]

            #Reshape pre-specified lower-trinagular A-matrix.
            mA = vA_flat.reshape(3,3)

        #Empty list to be filled with each of the 2500 covariance matrices.
        lSigmas = []

        #Set starting sigma.
        lSigmas.append(mSigma_starting)

        for t in range(1, iN):
            #Calculate covariance matrix using predefined function.
            lSigmas.append(Sigma_tplus1_calculation(
                mXtilde[t - 1, :],
                dBeta,
                mOmega,
                mA,
                dLambda,
                lSigmas[t - 1]))

        #Empty vector to store log-likelihood contributions.
        vLog_likelihood_contributions = np.zeros(iN)

        #Loop through observations and calculate log-likelihood contributions.
        for t in range(0, iN):
            vX_t = mXtilde[t, :]
            mSigma_t = lSigmas[t]
            vLog_likelihood_contributions[t] = Multivariate_t_log_likelihood(
                dLambda,
                iK,
                mSigma_t,
                vX_t)
            
        return vLog_likelihood_contributions

    ################################################################################
    ### vh= _gh_stepsize(vP)
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
        vh = np.maximum(vh, 5e-6)      # Don't go too small
        
        return vh

    ################################################################################
    ### vG= gradient_2sided(fun, vP, *args)
    def gradient_2sided(fun, vP, *args):
        """
        Purpose:
            Compute numerical gradient, using a 2-sided numerical difference
            Author:Charles Bos, following Kevin Sheppard's hessian_2sided, with
            ideas/constants from Jurgen Doornik's Num1Derivative
            
        Inputs:
            fun     function, as used for minimize()
            vP      1D array of size iP of optimal parameters
            args    (optional) extra arguments
        
        Return value:
            vG      iP vector with gradient
            
        See also:
            scipy.optimize.approx_fprime, for forward difference
        """
        
        iP   =  np.size(vP)
        vP   =  vP.reshape(iP)      # Ensure vP is 1D-array
        
        #  f  = fun(vP, *args)      # central function value is not needed
        vh= _gh_stepsize(vP)
        mh   =  np.diag(vh)         # Build a  diagonal matrix out of h
        
        fp = np.zeros(iP)
        fm = np.zeros(iP)
        for i in range(iP):         # Find f(x+h), f(x-h)
            fp[i] =  fun(vP+mh[i], *args)
            fm[i] =  fun(vP-mh[i], *args)
            
        vhr = (vP +  vh) - vP       # Check for effective stepsize right
        vhl = vP - (vP - vh)        # Check for effective stepsize left
        vG= (fp -  fm) /  (vhr +  vhl)  # Get central gradient
        
        return vG

    ################################################################################
    ### mG= jacobian_2sided(fun, vP, *args)
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
            mG      iN x  iP   matrix with jacobian
            
        See also:numdifftools.Jacobian(), for similar output
        """
        iP = np.size(vP)
        vP = vP.reshape(iP)        # Ensure vP is 1D-array
        vF = fun(vP, *args)        # evaluate function, only to get size
        iN = vF.size
        vh= _gh_stepsize(vP)
        mh   =  np.diag(vh)        # Build a  diagonal matrix out of h
        mGp = np.zeros((iN, iP))
        mGm = np.zeros((iN, iP))
        for i in   range(iP):     # Find f(x+h), f(x-h)
            mGp[:,i] =  fun(vP+mh[i], *args)
            mGm[:,i] =  fun(vP-mh[i], *args)
        vhr = (vP +  vh) - vP    # Check for effective stepsize right
        vhl = vP   -  (vP -  vh)    # Check for effective stepsize left
        mG= (mGp -  mGm) / (vhr +  vhl)  # Get central jacobian
        return mG

    ################################################################################
    ### mH= hessian_2sided(fun, vP, *args)
    def hessian_2sided(fun, vP, *args):
        """
        Purpose:
            Compute numerical hessian, using a  2-sided numerical difference
            
        Author:Kevin Sheppard, adapted by Charles Bos
        
        Source:https://www.kevinsheppard.com/Python_for_Econometrics
        
        Inputs:
            fun     function, as used for minimize()
            vP      1D array of size iP of optimal parameters
            args    (optional) extra arguments
            
        Return value:
            mH      iP x  iP matrix with symmetric hessian
        """
        iP = np.size(vP,0)
        vP= vP.reshape(iP)    # Ensure vP is 1D-array
        f = fun(vP, *args)
        vh= _gh_stepsize(vP)
        vPh = vP + vh
        vh = vPh - vP
        
        mh = np.diag(vh)      # Build a  diagonal matrix out of vh
        
        fp   =  np.zeros(iP)
        fm   =  np.zeros(iP)
        for i in range(iP):
            fp[i] =  fun(vP+mh[i], *args)
            fm[i] =  fun(vP-mh[i], *args)
        
        fpp = np.zeros((iP,iP))
        fmm = np.zeros((iP,iP))
        for i in   range(iP):
            for j in   range(i,iP):
                fpp[i,j] =  fun(vP +  mh[i] +  mh[j], *args)
                fpp[j,i] =  fpp[i,j]
                fmm[i,j] =  fun(vP -  mh[i] -  mh[j], *args)
                fmm[j,i] =  fmm[i,j]
                
        vh   =  vh.reshape((iP,1))
        mhh = vh   @  vh.T             # mhh= h  h', outer product of h-vector
        
        mH   =  np.zeros((iP,iP))
        for i in range(iP):
            for j in range(i,iP):
                mH[i,j] =  (fpp[i,j] -  fp[i] - fp[j] +  f  +  f  - fm[i] -  fm[j] + fmm[i,j])/mhh[i,j]/2
                mH[j,i] =  mH[i,j]
                
        return mH

    ################################################################################
    ### Standard_errors(vTheta_star) = mCov
    def Standard_errors(vTheta_star):
            
            #Define objective function for Hessian.
            dAve_log_likelihood = lambda vTheta: np.mean(Log_likelihood_function(
                vTheta,
                mXtilde,
                iK,
                iN,
                mOmega,
                mSigma_starting))
            
            #Define objective function for Jacobian.
            vLog_likelihood = lambda vTheta: Log_likelihood_function(
                vTheta,
                mXtilde,
                iK,
                iN,
                mOmega,
                mSigma_starting)
            
            # mH= -hessian_2sided(dAve_log_likelihood, vTheta_star)
            # mG = jacobian_2sided(vLog_likelihood, vTheta_star)
            # mG2 = (mG.T @ mG) / iN
            # mH_inv = np.linalg.inv(mH)
            # mVhat = (mH_inv @ mG2 @ mH_inv) / iN
            # vTheta_variance = mVhat

            # mK = jacobian_2sided(vParametrized_params, vTheta_star)
            # mTheta_true_variance = mK @ vTheta_variance @ mK.T
            # vTrue_se = np.sqrt(np.diagonal(mTheta_true_variance))

            #Calculate inverse hessian.
            mH= -hessian_2sided(dAve_log_likelihood, vTheta_star)
            mCov = np.linalg.inv(mH)

            #Force symmetricality.
            mCov = (mCov +  mCov.T)/2

            # compute the outer product of gradients of the average log likelihood
            mG = jacobian_2sided(vLog_likelihood, vTheta_star)

            mG = np.dot(mG.T, mG) / iN
            mG = np.dot(mG, mCov)
            mCov = np.dot(mCov, mG) / iN

            ##Standard errors via delta method.
            mJ = jacobian_2sided(Parametrize, vTheta_star)
            mTrue_cov = mJ @ mCov @ mJ.T
            vTrue_se = np.sqrt(np.diagonal(mTrue_cov))

            return vTrue_se

    ################################################################################
    ### Model1(dBeta_starting, dLambda_starting, dA_starting) = Ouptut
    def Model1(dBeta_starting, dLambda_starting, dA_starting):

        print("\nOptimising model specification 1: A11 * I")

        #Define objective function.
        dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(
            vTheta,
            mXtilde,
            iK,
            iN,
            mOmega,
            mSigma_starting))

        #Define starting values in parameter vector.
        vTheta_starting = np.array([dBeta_starting, dLambda_starting, dA_starting])

        #Optimise.
        res= opt.minimize(
            dAve_log_likelihood,
            vTheta_starting,
            method='Nelder-Mead')

        print("\nOptimization results:")
        print(res)

        #Transform parameters back.
        dBeta_result = (1 + np.exp(-res.x[0]))**-1
        dLambda_result = np.exp(res.x[1])
        dA11_result = (1 + np.exp(-res.x[2]))**-1

        print("\nLog-Likelihood: " + str(res.fun * 2500))
        print("AIC: " + str(-2*(res.fun * 2500) + 2 * 3))
        print("BIC: " + str(-2*(res.fun * 2500) + 2 * np.log(2500) * 3))
        print("\ndLambda: " + str(dLambda_result))
        print("\ndBeta: " + str(dBeta_result))
        print("\nmA: " + str(dA11_result))

        vTheta_star = np.array([res.x[0], res.x[1], res.x[2]])

        #Calculate covariance matrix and standard errors.
        vTrue_se = Standard_errors(vTheta_star)

        vT_stat = []
        for iCount in range(0, len(vTrue_se)):
            vT_stat.append(vTheta_star[iCount]/vTrue_se[iCount])
        
        vP_value = []
        # p-value for 2-sided test
        for iCount in range(0, len(vT_stat)):
            vP_value.append(2*(1 - t.cdf(abs(vT_stat[iCount]), dLambda_result)))

        print("\nStandard errors: \n" + str(vTrue_se))

        print("\nP-values: \n" + str(vP_value))

        print("\nEnd of model specification 1.")

        return

    ################################################################################
    ###def Model2(dBeta_starting, dLambda_starting, dA_starting)
    def Model2(dBeta_starting, dLambda_starting, vA_starting):

        print("\nOptimising model specification 2: diag(A11, A22, A33)")

        #Define objective function.
        dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(
            vTheta,
            mXtilde,
            iK,
            iN,
            mOmega,
            mSigma_starting))

        #Define starting values in parameter vector.
        vTheta_starting = np.insert(vA_starting, 0, [dBeta_starting ,dLambda_starting])

        #Optimise.
        res= opt.minimize(
            dAve_log_likelihood,
            vTheta_starting,
            method='Nelder-Mead')

        print("\nOptimization results:")
        print(res)

        #Transform parameters back.
        dBeta_result = (1 + np.exp(-res.x[0]))**-1
        dLambda_result = np.exp(res.x[1])
        mA_result = np.diag((1 + np.exp(-res.x[2:]))**-1)

        print("\nLog-Likelihood: " + str(res.fun * 2500))
        print("AIC: " + str(-2*(res.fun * 2500) + 2 * 5))
        print("BIC: " + str(-2*(res.fun * 2500) + 2 * np.log(2500) * 5))
        print("\ndLambda: " + str(dLambda_result))
        print("\ndBeta: " + str(dBeta_result))
        print("\nmA: \n" + str(mA_result))

        #Optimal subparameters.
        vTheta_star = res.x

        #Calculate covariance matrix and standard errors.
        vTrue_se = Standard_errors(vTheta_star)

        vT_stat = []
        for iCount in range(0, len(vTrue_se)):
            vT_stat.append(vTheta_star[iCount]/vTrue_se[iCount])
        
        vP_value = []
        # p-value for 2-sided test
        for iCount in range(0, len(vT_stat)):
            vP_value.append(2*(1 - t.cdf(abs(vT_stat[iCount]), dLambda_result)))

        print("\nStandard errors: \n" + str(vTrue_se))

        print("\nP-values: \n" + str(vP_value))

        print("\nEnd of model specification 2.")

        return

    ################################################################################
    ###def Model3(dBeta_starting, dLambda_starting, dA_starting)
    def Model3(dBeta_starting, dLambda_starting, vA_starting):

        print("\nOptimising model specification 3: lower triangular(A11, A21, A22, A31, A32, A33)")

        #Define objective function.
        dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(
            vTheta,
            mXtilde,
            iK,
            iN,
            mOmega,
            mSigma_starting))

        #Define starting values in parameter vector.
        vTheta_starting = np.insert(vA_starting, 0, [dBeta_starting ,dLambda_starting])

        #Optimise.
        res= opt.minimize(
            dAve_log_likelihood,
            vTheta_starting,
            method='Nelder-Mead')

        print("\nOptimization results:")
        print(res)
    
        vTrue_params = Parametrize(res.x)

        dBeta_result = vTrue_params[0]
        dLambda_result = vTrue_params[1]
        vA_lower_triangular = vTrue_params[2:]

        #Vector for holding flattened A-matrix.
        vA_flat = np.zeros(9)

        #Index mask for positions of lower-triangular elements.
        vIndex_mask = [0, 3, 4, 6, 7, 8]

        #Places values from vA_lower_triangular into positions in vIndex_mask.
        for iCount in range(0, len(vA_lower_triangular)):
            vA_flat[vIndex_mask[iCount]] = vA_lower_triangular[iCount]

        #Reshape pre-specified lower-trinagular A-matrix.
        mA = vA_flat.reshape(3,3)

        print("\nLog-Likelihood: " + str(res.fun * 2500))
        print("AIC: " + str(-2*(res.fun * 2500) + 2 * 8))
        print("BIC: " + str(-2*(res.fun * 2500) + 2 * np.log(2500) * 8))
        print("\ndLambda: " + str(dLambda_result))
        print("\ndBeta: " + str(dBeta_result))
        print("\nmA: \n" + str(vA_lower_triangular))
        print("\nmA: \n" + str(mA))

        vTheta_star = res.x

        #Calculate covariance matrix and standard errors.
        vTrue_se = Standard_errors(vTheta_star)

        vT_stat = []
        for iCount in range(0, len(vTrue_se)):
            vT_stat.append(vTheta_star[iCount]/vTrue_se[iCount])
        
        vP_value = []
        # p-value for 2-sided test
        for iCount in range(0, len(vT_stat)):
            vP_value.append(2*(1 - t.cdf(abs(vT_stat[iCount]), dLambda_result)))

        print("\nStandard errors: \n" + str(vTrue_se))

        print("\nP-values: \n" + str(vP_value))

        print("\nEnd of model specification 3.")

        return

    ################################################################################
    #Magic numbers.
    # path = r"data_ass_2.csv"
    # df_test, df_real = loadin_data(path)

    #Full dataset for calculating mOmega.
    mFull = np.array(df_real)
    mFull_de_mean = mFull - np.mean(mFull, axis = 0)

    #Use first 2500 observations.
    mSample = np.array(df_real.iloc[0: 2500, :], dtype=np.float64) * 100

    #De-mean each column.
    mXtilde = mSample - np.mean(mFull, axis = 0)

    #Get dimensions for generality.
    (iN, iK) = mXtilde.shape

    #Set starting dLambda value.
    dLambda_starting = 8

    #Calculate mOmega as specified.
    mOmega = (((mXtilde.T@ mXtilde)/mXtilde.shape[0]) * dLambda_starting) / (dLambda_starting - 2)

    #Set starting dBeta value.
    dBeta_starting = 0.96

    #Set starting mSigmat.
    mSigma_starting = (((mXtilde[0:50, :].T@ mXtilde[0:50, :])
    / mXtilde[0:50, :].shape[0]) * dLambda_starting ) / (dLambda_starting - 2)

    ##First model specification.
    dA_starting = np.sqrt(0.02)

    Model1(dBeta_starting, dLambda_starting, dA_starting)

    ##Second model specification.
    vA_starting = np.sqrt(np.array([0.02, 0.02, 0.02]))

    Model2(dBeta_starting, dLambda_starting, vA_starting)

    ##Third model specification.
    vA_starting = np.sqrt(np.array([0.02, 0, 0.02, 0, 0, 0.02]))

    Model3(dBeta_starting, dLambda_starting, vA_starting)
    
    return


###########################################################
### main
def main():
    # magic numbers
    path = r"data_ass_2.csv"
    _, df_real = loadin_data(path)

    # now call the functions that print all of the output for all questions
    output_Q1(df_real)
    output_Q2()
    estimates = output_Q3(df_real)
    output_Q4(df_real, estimates)
    output_Q5(df_real, estimates)
    output_Q7(df_real)
    

###########################################################
### start main
if __name__ == "__main__":
    main()

# import pachages
import enum
from numpy.ma.core import concatenate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import datetime as dt
from scipy.special import loggamma
import scipy.optimize as opt

#Change runtime warnings to errors to catch in debugging.
np.seterr(all='raise')

################################################################################
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

################################################################################
### loadin_data
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
###Function Log-likelihood.
def Multivariate_t_log_likelihood(dLambda, iK, mSigma_t, vX_t):
    #Force column vector.
    vX_t.shape = (3,1)

    # #Break up into terms for debugging.
    dTerm1 = loggamma(1/2 * (dLambda + iK)) - loggamma(dLambda/2) 
    dTerm2 = - 1/2 * np.log(np.linalg.det(np.pi * dLambda * mSigma_t))
    dTerm3 = -1/2 * (dLambda + iK) * np.log(1 + dLambda**-1 * (vX_t.T @ np.linalg.inv(mSigma_t) @ vX_t))

    #Break up into terms for debugging.
    # dTerm1 = loggamma(1/2 * (dLambda + iK))
    # dTerm2 = -loggamma(dLambda/2)
    # dTerm3 = -iK/2 * np.log(dLambda * np.pi)
    # dTerm4 = -1/2 * np.log(np.linalg.det(mSigma_t))
    # dTerm5 = -1/2 * (dLambda + iK) * np.log(1 + dLambda**-1 * (vX_t.T @ np.linalg.inv(mSigma_t) @ vX_t))

    #Sum for final log-likelihood contribution.
    dLog_likelihood = dTerm1 + dTerm2 + dTerm3 #+ dTerm4 + dTerm5

    return dLog_likelihood

################################################################################
###Parametrize(vTheta, sFrom_to, dModel) = vTrue_parameters = [dBeta, dLambda, vA]
def Parametrize(vTheta, iModel):

    #Save as array for easy combining later.
    dBeta = (1 + np.exp(-vTheta[0]))**-1
    dLambda = np.exp(vTheta[1])

    #Model specification 1.
    if iModel == 1:
        #Save as array for easy combining later.
        vA_parametrized = [np.array((1 + np.exp(-vTheta[2]))**-1)]

    #Model specification 2.
    if iModel == 2:
        #Save as array for easy combining later.
        vA_parametrized = np.array((1 + np.exp(-vTheta[2:]))**-1)
    
    #Model specification 3.
    if iModel == 3:
        #Create array of zeros to hold falttened A matrix.
        vA_parametrized = np.zeros(9)
        
        #Mask for which indices of which values in flattened A matrix to be 
        #replaced with values in vTheta.
        vMatrix_mask = np.array([0, 3, 4, 6, 7, 8])

        #Set values at indices of vMatrix_mask equal to the vTheta[2:] values,
        #so we are left with an upper triangular matrix.
        vA_parametrized[vMatrix_mask] = vTheta[2:]

        #Loop through flattened matrix to parametrize values.
        for iCount, dMatrix_value in enumerate(vA_parametrized):
            #For diagonal elements.
            if (iCount == 0 or iCount == 4 or iCount == 8):
                vA_parametrized[iCount] = (1 + np.exp(-dMatrix_value))**-1

            #For off-diagonal elements.
            else:
                vA_parametrized[iCount] = 1/3 * (-1 + 2 / (1 + np.exp(-dMatrix_value)))
    
    # #Put everything back together in the same format it was input.
    # vTrue_parameters = np.concatenate(([dBeta], [dLambda], vA_parametrized))
    
    return dBeta, dLambda, vA_parametrized

################################################################################
###Log_likelihood_function(vTheta, mXtilde, iK, iN, mOmega, mSigma_starting, iModel)
def Log_likelihood_function(vTheta, mXtilde, iK, iN, mOmega, mSigma_starting, iModel):

    if iModel == 1:
        #Re-parametrized  model parameters prior to optimisation.
        (dBeta, dLambda, dA11) = Parametrize(vTheta, iModel = 1)

        #Pre-specified A-matrix A11 * I.
        mA = dA11 * np.identity(3)
    
    if iModel == 2:
        #Re-parametrized  model parameters prior to optimisation.
        (dBeta, dLambda, vA_flat) = Parametrize(vTheta, iModel = 2)
        
        #Pre-specified diagonal A-matrix.
        mA = np.diagflat(vA_flat)

    if iModel == 3:
        #Re-parametrized  model parameters prior to optimisation.
        (dBeta, dLambda, vA_flat) = Parametrize(vTheta, iModel = 3)

        mA = vA_flat.reshape(3,3)
        #print(mA)

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
###def Model1(dBeta_starting, dLambda_starting, dA_starting, iModel)
def Model1(dBeta_starting, dLambda_starting, dA_starting, iModel):

    print("\nOptimising model specification 1: A11 * I")

    #Define objective function.
    dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(vTheta, mXtilde, iK, iN, mOmega, mSigma_starting, iModel))

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

    print("\ndLambda: " + str(dLambda_result))
    print("\ndBeta: " + str(dBeta_result))
    print("\nmA: " + str(dA11_result))

    print("\nEnd of model specification 1.")

    return

################################################################################
###def Model2(dBeta_starting, dLambda_starting, dA_starting, iModel)
def Model2(dBeta_starting, dLambda_starting, vA_starting, iModel):

    print("\nOptimising model specification 2: diag(A11, A22, A33)")

    #Define objective function.
    dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(
        vTheta,
        mXtilde,
        iK,
        iN,
        mOmega,
        mSigma_starting,
        iModel))

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

    print("\ndLambda: " + str(dLambda_result))
    print("\ndBeta: " + str(dBeta_result))
    print("\nmA: \n" + str(mA_result))

    print("\nEnd of model specification 2.")

    return

################################################################################
###def Model3(dBeta_starting, dLambda_starting, dA_starting, iModel)
def Model3(dBeta_starting, dLambda_starting, vA_starting, iModel):

    print("\nOptimising model specification 3: lower triangular(A11, A21, A22, A31, A32, A33)")

    #Define objective function.
    dAve_log_likelihood = lambda vTheta: -np.mean(Log_likelihood_function(vTheta, mXtilde, iK, iN, mOmega, mSigma_starting, iModel))

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
    dBeta_test = (1 + np.exp(-res.x[0]))**-1
    dLambda_test = np.exp(res.x[1])

    (dBeta_result,
    dLambda_result,
    vA_flattened) = Parametrize(res.x, iModel)

    mA_result = vA_flattened.reshape(3,3)

    print("\ndLambda: " + str(dLambda_result))
    print("\ndBeta: " + str(dBeta_result))
    print("\nmA: \n" + str(mA_result))

    print("\nEnd of model specification 3.")

    return

################################################################################
#Magic numbers.
path = r"data_ass_2.csv"
df_test, df_real = loadin_data(path)

#Full dataset for calculating mOmega.
mFull = np.array(df_test) * 100
mFull_de_mean = mFull - np.mean(mFull, axis = 0)

#Use first 2500 observations.
mSample = np.array(df_test.iloc[0: 2500, :], dtype=np.float64) * 100

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
iModel = 1

Model1(dBeta_starting, dLambda_starting, dA_starting, iModel)

##Second model specification.
vA_starting = np.sqrt(np.array([0.02, 0.02, 0.02]))
iModel = 2

Model2(dBeta_starting, dLambda_starting, vA_starting, iModel)

##Third model specification.
vA_starting = np.sqrt(np.array([0.02, 0, 0.02, 0, 0, 0.02]))
iModel = 3

Model3(dBeta_starting, dLambda_starting, vA_starting, iModel)
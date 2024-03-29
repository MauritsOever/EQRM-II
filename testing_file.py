# import pachages
from numpy.ma.core import concatenate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc
import datetime as dt
from scipy.special import loggamma
import scipy.optimize as opt
import pandas as pd
from scipy.stats import t
pd.options.display.float_format = '{:.6f}'.format
np.set_printoptions(suppress=True)
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
path = r"data_ass_2.csv"
df_test, df_real = loadin_data(path)

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
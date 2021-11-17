# -*- coding: utf-8 -*-
"""
store old vers of 4b temporarily

@author: gebruiker
"""
    # Q4 b
    # estimating the VAR(1) and VAR(2) model by ML
    print('Question 4 b: ')
    print('')
    # start with VAR(1) 
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
    
    
    params =  np.zeros(12)    # initialize parameters
    y = np.array(df[rets][1:]).T # define data used and put in right shape
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
    params = np.zeros(21) # initialize params
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ###########################################################################
        #question 4 c
    print('Question 4 c: ')
    print('')
    #initialize 0 matrix P
    # python implementation of the Cholesky-Banachiewicz algorithm
    P_hat = np.zeros((3,3))
    for i in range(3):
        for k in range(i+1):
            tmp_sum = sum(P_hat[i][j] * P_hat[k][j] for j in range(k))
            if (i == k): # Diagonal elements
                P_hat[i][k] = np.sqrt(sigma_hat_var2[i][i] - tmp_sum)
            else:
                P_hat[i][k] = (1.0 / P_hat[k][k] * (sigma_hat_var2[i][k] - tmp_sum))

    print('P_hat = ')
    print(P_hat)
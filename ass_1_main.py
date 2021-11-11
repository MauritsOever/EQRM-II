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
    
    df['DJIA.Ret'] = np.log(df.iloc[:,0]) - np.log(df.iloc[:,0].shift(1))
    df['N225.Ret'] = np.log(df.iloc[:,1]) - np.log(df.iloc[:,1].shift(1))
    df['SSMI.Ret'] = np.log(df.iloc[:,2]) - np.log(df.iloc[:,2].shift(1))
    
    return df

def output_Q1(df):
    # for this question, we only use data until 2010
    # we dont have date indexing or whatever, so we dont know when to split the data
    # but we assume:
    df = df.iloc[:5000,:]
    
    
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
    
    
    


###########################################################
### main
def main():
    # magic numbers
    path = r"C:\Users\gebruiker\Documents\GitHub\EQRM-II\triv_ts.txt"
    df = loadin_data(path)
    


###########################################################
### start main
if __name__ == "__main__":
    main()

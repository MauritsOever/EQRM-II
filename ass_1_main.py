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
    data = pd.read_csv(path, sep = "	")
    df = df.dropna(axis=0) # there's no dates? how do we know whats 2010 and what is not?
    
    df['DJIA.Ret'] = np.log(df.iloc[:,0]) - np.log(df.iloc[:,0].shift(1))
    df['N225.Ret'] = np.log(df.iloc[:,1]) - np.log(df.iloc[:,1].shift(1))
    df['SSMI.Ret'] = np.log(df.iloc[:,2]) - np.log(df.iloc[:,2].shift(1))
    
    return data

def output_Q1(df):
    
    # Q1 a
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

import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "..")

import BM

data = "../dati/"

def fai_cose():
    # ds = pd.read_excel(data+"iphone.xlsx")    
    # df = pd.to_numeric(ds.iphone[0:46], errors='coerce')

    # ds = pd.read_excel(data+"blackb.xlsx")    
    # df = pd.to_numeric(ds.blackberry, errors='coerce')
    
    ds = pd.read_excel(data+"denmarkwind.xlsx")    
    df = pd.to_numeric(ds.Denmark[7:31], errors='coerce')

    
 

    BM.BM(df)


if __name__== "__main__":
    fai_cose()


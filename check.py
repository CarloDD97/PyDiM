import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "..")

import BM, GBM

data = "../dati/"

if __name__== "__main__":
    ds = pd.read_excel(data+"iphone.xlsx")    
    df = pd.to_numeric(ds.iphone[0:46], errors='coerce')

    # ds = pd.read_excel(data+"blackb.xlsx")    
    # df = pd.to_numeric(ds.blackberry, errors='coerce')
    
    # ds = pd.read_excel(data+"denmarkwind.xlsx")    
    # df = pd.to_numeric(ds.Denmark[7:31], errors='coerce')
 
    # BM.BM(df, display=1)

    # ds = pd.read_excel(data+"apple.xlsx")    
    # df = pd.to_numeric(ds.iMac, errors='coerce')

    # BM.BM(df, display=1)

    GBM.GBM(df, shock="mixed", prelimestimates=[1823.7,0.0014,0.12587, 17, -0.1, 0.1, 25, 30, 0.1], nshock=2)
    # GBM.GBM(df, shock="rett", prelimestimates=[270.03, 0.0048, 0.0636, 20,30,0.1], nshock=1)


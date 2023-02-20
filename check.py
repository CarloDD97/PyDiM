import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, "..")

import BM, GBM, GGM, UCRCD

data = "../dati/"

if __name__== "__main__":
    ds = pd.read_excel(data+"iphone.xlsx")    
    df = pd.to_numeric(ds.iphone[:40])

    # ds = pd.read_excel(data+"blackb.xlsx")    
    # df = pd.to_numeric(ds.blackberry, errors='coerce')
    
    # ds = pd.read_excel(data+"denmarkwind.xlsx")    
    # df = pd.to_numeric(ds.Denmark[7:31], errors='coerce')
 
    # BM.BM(df, display=1)

    ds1 = pd.read_excel(data+"apple.xlsx")    
    df1 = pd.to_numeric(ds1.iMac)

    # BM.BM(df, display=1)

    # GBM.GBM(df, shock="exp", prelimestimates=[1823.7,0.0014,0.12587, 17, -0.1, 0.1], nshock=1)
    # GGM.GGM(df, prelim
    # estimates=[1823, 0.001, 0.1, 0.001, 0.1])
    UCRCD.UCRCD(df1, df, par='unique')

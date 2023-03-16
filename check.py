import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, "..")

import BM, GBM, GGM, UCRCD, plot

data = "../dati/"

if __name__== "__main__":
    ds = pd.read_excel(data+"iphone.xlsx")    
    df = pd.to_numeric(ds.iphone[:40])

    # ds = pd.read_excel(data+"blackb.xlsx")    
    # df = pd.to_numeric(ds.blackberry, errors='coerce')
    
    # ds1 = pd.read_excel(data+"denmarkwind.xlsx")    
    # df1 = pd.to_numeric(ds1.Denmark[6:31])
    # print(len(df1))
 
    # BM.BM(df, display=1)

    ds2 = pd.read_excel(data+"apple.xlsx")    
    df2 = pd.to_numeric(ds2.iMac)

    # bm = BM.BM(df, display=0)
    # plot.dimora_plot(bm, None, 40)

    # gbm = GBM.GBM(df, shock="exp", prelimestimates=[1823.7,0.0014,0.12587, 17, -0.1, 0.1], nshock=1, , display=0)
    # plot.dimora_plot(gbm, None, 5)

    # ggm = GGM.GGM(df, prelimestimates=[1823, 0.001, 0.1, 0.001, 0.1], display=0)
    # plot.dimora_plot(ggm, None, 5)

    ucrcd = UCRCD.UCRCD(df2, df, par='unique', display=0, stats = 1)
    plot.dimora_plot(ucrcd, oss=5)

    




    



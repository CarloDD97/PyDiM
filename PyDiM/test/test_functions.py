import pandas as pd
import os
from PyDiM import BM, GBM, GGM, UCRCD, plot, summary

stream = os.path.join(os.path.dirname(__file__), 'sample_data/')

ds = pd.read_excel(stream + "iphone.xlsx")    
df = pd.to_numeric(ds.iphone[:40])

ds0 = pd.read_excel(stream + "blackb.xlsx")    
df0 = pd.to_numeric(ds0.blackberry)

ds1 = pd.read_excel(stream + "denmarkwind.xlsx")    
df1 = pd.to_numeric(ds1.Denmark[6:31])

ds2 = pd.read_excel(stream + "apple.xlsx")    
df2 = pd.to_numeric(ds2.iMac)

ds3 = pd.read_excel(stream + 'australia1.xlsx')
df3 = pd.to_numeric(ds3.Caustralia)

def test_BM():
    assert BM.bm(df, display=0)
    assert BM.bm(df, method='optim', display=0)

def test_GBM():
    assert GBM.gbm(df, shock="exp", prelimestimates=[1823.7,0.0014,0.12587, 17, -0.1, 0.1], nshock=1, display=0)
    assert GBM.gbm(df2, shock="rett", prelimestimates=[270.03, 0.0048, 0.0636, 20,30,0.1], nshock=1, display=0)
    assert GBM.gbm(df3, shock="mixed", prelimestimates=[1.4574e+02, 4.2336e-03, 5.0649e-02, 47, -0.1, -0.1, 7, 45, -0.05], nshock=2, display=0)

def test_GGM():
    assert GGM.ggm(df, prelimestimates=[1823, 0.001, 0.1, 0.001, 0.1], display=0)

def test_UCRCD():
    assert UCRCD.ucrcd(df2, df)
    assert UCRCD.ucrcd(df2, df, par='unique')

def test_summary():
    bass = BM.bm(df, display=0)
    assert summary.print_summary(bass)
    genb = GBM.gbm(df3, shock="mixed", prelimestimates=[1.4574e+02, 4.2336e-03, 5.0649e-02, 47, -0.1, -0.1, 7, 45, -0.05], nshock=2, display=0)
    assert summary.print_summary(genb)
    ucrcd_a = UCRCD.ucrcd(df2, df, par='unique', display=0)
    assert summary.print_summary(ucrcd_a)

def test_plot():
    bass = BM.bm(df, display=0)
    assert plot.dimora_plot(bass, None, 40)
    ucrcd_a = UCRCD.ucrcd(df2, df, par='unique', display=0)
    assert plot.dimora_plot(ucrcd_a, None, 40)


    




    



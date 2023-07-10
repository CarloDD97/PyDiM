import pandas as pd
import os
from PyDiM import BM, GBM, GGM, UCRCD, plot, summary

stream = os.path.join(os.path.dirname(__file__), 'sample_data/')


# ds0 = pd.read_excel(stream + "blackb.xlsx")    
# df0 = pd.to_numeric(ds0.blackberry)

# ds1 = pd.read_excel(stream + "denmarkwind.xlsx")    
# df1 = pd.to_numeric(ds1.Denmark[6:31])

# ds2 = pd.read_excel(stream + "apple.xlsx")    
# df2 = pd.to_numeric(ds2.iMac)

# ds3 = pd.read_excel(stream + 'australia1.xlsx')
# df3 = pd.to_numeric(ds3.Caustralia)

cd = pd.read_csv(stream + "cd.csv").CD

jap = pd.read_csv(stream + "jap_br.csv").Birth

fb = pd.read_csv(stream + "fb_interest.csv").Int

iphone = pd.read_csv(stream + "iphone.csv").Int

covid = pd.read_csv(stream + 'covid_data.csv')

def test_BM():
    assert BM.bm(cd, display = 0)
    assert BM.bm(cd, method='optim', display=0)

def test_GBM():
    assert GBM.gbm(jap, shock='rett', nshock=1, prelimestimates=[3.3165e+03, 6.3688e-03, 2.7111e-02, 75, 110, -.1], display=0)
    assert GBM.gbm(fb, shock='mixed', nshock=2, prelimestimates=[8072.7, 0.00129, 0.041, 30, 0.3, .08, 75, 100, -.1], display=0)

def test_GGM():
    assert GGM.ggm(iphone, display=0)

def test_UCRCD():
    assert UCRCD.ucrcd(covid.new_cases, covid.new_vaccinations, display=0)
    assert UCRCD.ucrcd(covid.new_cases, covid.new_vaccinations, par='unique')

def test_summary():
    bass = BM.bm(cd, display=0)
    assert summary.print_summary(bass)
    genb =  GBM.gbm(fb, shock='mixed', nshock=2, prelimestimates=[8072.7, 0.00129, 0.041, 30, 0.3, .08, 75, 100, -.1], display=0)
    assert summary.print_summary(genb)
    ucrcd_a = UCRCD.ucrcd(covid.new_cases, covid.new_vaccinations, display=0)
    assert summary.print_summary(ucrcd_a)

def test_plot():
    bass = BM.bm(cd, display=0)
    assert plot.dimora_plot(bass, None, 40)
    ucrcd_a = UCRCD.ucrcd(covid.new_cases, covid.new_vaccinations, display=0)
    assert plot.dimora_plot(ucrcd_a, None, 40)




    




    



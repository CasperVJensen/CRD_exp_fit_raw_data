import pandas as pd
import os
import sys
import numpy as np
import re
from statistics import mean
from iminuit import Minuit
from collections import Counter
import time
import concurrent.futures
from scipy.stats import norm
#%%
tic = time.perf_counter()

dataname = str(sys.argv[1]) #takes first arg after script as input string
#%%
numbers = re.compile(r'(\d+)') 
def numericalSort(value):                       #function for ordering numbers
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def exp_mon(a,b,c):                             #normal exp fit
    return a * np.e**(-(b)*x) + c

def leastsq(a,b,c):                             #least square fit
    y_fit = exp_mon(a,b,c)
    ls = np.sum((y-y_fit)**2)
    return ls

def chi2_fit(a,b,c):                            #chi square fit
    y_fit = exp_mon(a,b,c)
    chi2 = np.sum(((y-y_fit)**2)/sigma_y_sq)
    return chi2

def fit_minuit(y1, a1, b1, c1, p=0):            #minuit fit function
    
    global y                                   #choose only where signal < 2.9V
    y = y1[y1<a1]
    y = y[y>c1].reset_index(drop=True)
    
    global x
    x = pd.Series(range(len(y)))*(1/sampling_rate)
    
    m = Minuit(leastsq, a=a1, b=b1, c=c1)       #first fit with least squares
    m.errordef = 1.0
    m.migrad()
    
    f = m.values[0] * np.e**(-m.values[1]*x) + m.values[2] #make function with found parameters
    
    global sigma_y_sq                                       #estimate sigma for all points from least sq
    sigma_y_sq = (y-f)**2
    
    m2 = Minuit(chi2_fit, a=m.values[0], b=m.values[1], c=m.values[2])  #chi square fit with found sigmas
    m2.errordef = 1.0
    m2.migrad()
    
    # chi2_val = m2.fval                     #USE FOR ANALYSIS
    # Ndof = len(x)-2
    # P = stats.chi2.sf(chi2_val, Ndof)
    
    # if p==1:
    #     print(m2.migrad())
    # else:
    #     pass
    
    if m2.valid == True and m2.accurate == True:
        pass
    else:
        #print("did not converge. Will try again starting with found tau")
        
        m = Minuit(leastsq, a=m2.values[0], b=m2.values[1], c=m2.values[2])
        m.errordef = 1.0
        m.migrad()
        
        f = m.values[0] * np.e**(-m.values[1]*x) + m.values[2]
        sigma_y_sq = (y-f)**2
        
        m2 = Minuit(chi2_fit, a=m.values[0], b=m.values[1], c=m.values[2])
        m2.errordef = 1.0
        m2.migrad()
        
        # chi2_val = m2.fval                     #USE FOR ANALYSIS
        # Ndof = len(x)-2
        # P = stats.chi2.sf(chi2_val, Ndof)
        
        # if p==1:
        #     print(m2.migrad())
        # else:
        #     pass
        
    if m2.valid == True and m2.accurate == True:
        return [m2.values[1], m2.errors[1]]# , m2.values[0], m2.errors[0], m2.values[2], m2.errors[2], m2.accurate]#, P]
        
    else:
        #print("did not converge AGAIN. Remove value")
        return [m2.values[1], 0]            #changes delta tau to 0 so it can be removed later

def decay_fit(row):                         #activating fitting in format for parallel processing
    a = pd.Series(row[1])
    
    m = fit_minuit(a,startfilter,0.02,deadV,0)
    
    if m[1] == 0:
        tau = float("NaN")                  #if delta = 0 makes NaN that can be removed later
        d_tau = float("NaN")
    else:    
        tau = 1/m[0]
        d_tau = (m[1]/m[0]) * (1/m[0])
    
    return [tau, d_tau]
#%%
cha_factor = (5/6)                          #slightly more harsh factor than normal 1/2 factor

def chauvenet():
    global j
    for j in range(1,40):                   #try max of 40 iterations

        cha_score = []
        mean_tau = []
        std_tau = []
        tau_i = []
        
        i = 0
        tau_list = []
        for val in zip(tau, nm_list):       #tau_list is a list of taus grouped by nm identity
            if val[1] == nm[i]:
                tau_list.append(val[0])
            
            else:    
                tau_i.append(tau_list)
                
                i+=1
                tau_list = []
                tau_list.append(val[0])
        
        tau_i.append(tau_list)              #tau_i is a list of all the tau_lists
        
        for idx, val in enumerate(tau_i):
            mean = np.mean(tau_i[idx])
            mean_tau.append(mean)
            
            std = np.std(tau_i[idx], ddof=1)
            std_tau.append(std)
            
        i = 0
        
        for val in zip(tau, nm_list):       #starts evaluation chauvenets criterion
            if val[1] == nm[i]:
                if numrings[-1][i] < (numrings[1][i]*0.8): #if number of ringdowns is above threshold of 80% of raw
                    cha_score.append(1)                     #appends 1 if too low --> counts as convergence
                else:
                    p = 1 - (0.5 * cha_factor * (1/numrings[-1][i]))
                    z_limit = norm.ppf(p)                               #find z_limit
                    
                    z_score = abs(val[0] - mean_tau[i]) / std_tau[i]    #evaluate z_score
                    check = z_limit - z_score
                    
                    if check > 0:                                       #if z-score is too high check is negative
                        cha_score.append(1)                             #if z_score okay --> 1
                    else:
                        cha_score.append(check)
            
            else:                                                       #just to keep correct nm grouping
                i+=1
                if numrings[-1][i] < (numrings[1][i]*0.8):
                    cha_score.append(1)
                else:
                    p = 1 - (0.5 * cha_factor * (1/numrings[-1][i]))
                    z_limit = norm.ppf(p)
                    
                    z_score = abs(val[0] - mean_tau[i]) / std_tau[i]
                    check = z_limit - z_score
                    
                    if check > 0:
                        cha_score.append(1)
                    else:
                        cha_score.append(check)
        
        if len(list(Counter(cha_score).values())) == 1:                 #if only ones then converged
            #print(str("I'm breaking free at ") + str(j) + str(" iterations"))
            break
        
        else:
            for idx, val in enumerate(zip(cha_score, tau, d_tau, nm_list)): #pop bad data points
                if val[0] <= 0 or np.isnan(val[0]) == True:
                    cha_score.pop(idx)
                    tau.pop(idx)
                    d_tau.pop(idx)
                    nm_list.pop(idx)
            
        numrings.append(list(Counter(nm_list).values()))                    #update number of ringdowns pr nm
#%%
raw = pd.read_csv(dataname,delimiter="\t", decimal=",",header=None) #import rawdata
#%%
nm = list(Counter(raw).keys()) #list of wavelengths

numrings = []
numrings.append(list(Counter(raw.iloc[:,0]).values())) #number of ringdonms for each wavelength

startfilter = 1.0 #filtering out dead signals below this value
sampling_rate = len(raw.iloc[0,:])/500 #we sample x number of points in 500 µs

raw = raw[raw.iloc[:,5:15].mean(axis=1)>=startfilter]; raw.head()  #only keep points where avg of first 10 values >2V (non dead signals)

nm_list = raw.iloc[:,0].reset_index(drop=True) #nm of each ringdonm

nm = list(Counter(nm_list).keys())
numrings.append(list(Counter(nm_list).values())) #number of ringdonms after first filter
nm_list = list(nm_list)
#%%
avg_n = np.mean(numrings[1]) #number of ringdowns pr wavelength
#%%
I = raw.iloc[:,5:].reset_index(drop=True) #signal not including rising edge
I.columns = range(I.columns.size) #reset column names(index)

upperfit = 2.9
deadV = mean(I.iloc[10,-100:])*1.3  #dead voltage+30%. deadV should be the value expfit finds for +c

#%%
'''fitting procedure'''

tau = []
d_tau = []

with concurrent.futures.ProcessPoolExecutor() as executor:
    
    results = executor.map(decay_fit,I.iterrows(), chunksize=1)

    for f in results:
        tau.append(f[0])
        d_tau.append(f[1])
#%%
for idx, val in enumerate(zip(tau, d_tau, nm_list)): #remove NaN values from fit
    if np.isnan(val[0])==True:
        tau.pop(idx)
        d_tau.pop(idx)
        nm_list.pop(idx)

numrings.append(list(Counter(nm_list).values())) #number of ringdowns after fitting
#%%
''' mean for first result'''
mean_tau = []
std_tau = []
tau_i = []

i = 0
tau_list = []
for val in zip(tau, nm_list):                       #groups tau by nm identity
    if val[1] == nm[i]:
        tau_list.append(val[0])
    
    else:    
        tau_i.append(tau_list)
        
        i+=1
        tau_list = []
        tau_list.append(val[0])

tau_i.append(tau_list)

for idx, val in enumerate(tau_i):
    mean = np.mean(tau_i[idx])
    mean_tau.append(mean)
    
    V = (tau_i[0]-mean_tau[0])**2                       #sample Variance
    std = np.sqrt((1/(len(tau_i[idx])-1)) * np.sum(V))  #sample standard deviation
    
    mean_std = std / np.sqrt(len(tau_i[idx]))           #standard uncertainty on mean tau  
    std_tau.append(mean_std)


first_tau = pd.Series(mean_tau)*1
first_std_tau = pd.Series(std_tau)*1
#%%
for idx, val in enumerate(first_tau):               #formats the number a little nicer
    first_tau[idx] = "{:.3e}".format(val)

for idx, val in enumerate(first_std_tau):
    first_std_tau[idx] = "{:.1e}".format(val)
#%%
chauvenet()                                         #runs chauvenet function. look in function for docs
#%%
'''mean for final result'''
mean_tau = []
std_tau = []
tau_i = []

i = 0
tau_list = []
for val in zip(tau, nm_list):
    if val[1] == nm[i]:
        tau_list.append(val[0])
    
    else:    
        tau_i.append(tau_list)
        
        i+=1
        tau_list = []
        tau_list.append(val[0])

tau_i.append(tau_list)

for idx, val in enumerate(tau_i):
    mean = np.mean(tau_i[idx])
    mean_tau.append(mean)
    
    V = (tau_i[0]-mean_tau[0])**2                       #sample Variance
    std = np.sqrt((1/(len(tau_i[idx])-1)) * np.sum(V))  #sample standard deviation
    
    mean_std = std / np.sqrt(len(tau_i[idx]))           #standard uncertainty on mean tau  
    std_tau.append(mean_std)

mean_tau = pd.Series(mean_tau)
std_tau = pd.Series(std_tau)

for idx, val in enumerate(mean_tau):
    mean_tau[idx] = "{:.3e}".format(val)

for idx, val in enumerate(std_tau):
    std_tau[idx] = "{:.1e}".format(val)

#%%
wn_laser = 10000000/(pd.Series(nm)*1.000293)                     #make wavenumber axis


for idx, val in enumerate(wn_laser):
    wn_laser[idx] = "{:.6e}".format(val)
#%%
out_file_name = dataname.replace('rawdata.dat', 'tau.txt')
out_file_name = r'{}'.format(out_file_name)
#%%
output1 = pd.DataFrame({"nm_laser_air" : nm,
                        "wn_laser_vac" : wn_laser,
                        "tau" : mean_tau,
                        "d_tau" : std_tau,
                        "first_tau" : first_tau,
                        "first_d_tau" : first_std_tau,
                        "n_final" : pd.Series(numrings[-1]),
                        "n_filter" : numrings[1],
                        "n_raw" : numrings[0]})

output1 = output1.iloc[::-1].reset_index(drop=True)

output1.to_csv(out_file_name, index=None, sep='\t', mode='w')
#%%
toc = time.perf_counter()

file = open(out_file_name, "a")
timer = (toc-tic)/60
timer = "{:.2f}".format(timer)
file.write("\n")
file.write("time to complete script = " + str(timer) + " mins")
file.write("\n")
file.write("chauvenet iterations = " + str(j))
file.close()
#%%
os.remove(dataname)

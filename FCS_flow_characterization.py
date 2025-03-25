# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 15:33:20 2024

@author: Alberto
"""

import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import pandas as pd
import os

from scipy.optimize import curve_fit
from matplotlib.ticker import (MultipleLocator)
from mpl_toolkits.axes_grid1.inset_locator import (InsetPosition, mark_inset)
from scipy.stats import linregress

os.chdir(r"XXXXXX") #Input working directory


sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper",font_scale=1.5)
sns.set_palette("bright")



def diffusion(tau,g0,td):
    
   
    y = g0/((1+tau/td)*(1+(w**2/z0**2)*(tau/td))**(1/2))
    
    return y



def flow(tau,g0,v):
    
    
    
    y = g0*np.exp(-(v*tau/w)**2)

    return y


def anomalous_D(tau,g0,td,a,):
    
    y = g0/((1+(tau/td)**a)*(1+(w**2/z0**2)*(tau/td)**a)**(1/2))
    
    return y
    


def combined_model(tau,g0,td,v):
    
    
    y = g0/((1+tau/td)*(1+(w**2/z0**2)*(tau/td))**(1/2))*np.exp(-(v*tau/w)**2/((1+tau/td)*(1+(w**2/z0**2)*(tau/td))**(1/2)))
    
    
    return y




def plot_flow_model(tau,g0,v):
    
    
    G_fit = flow(tau,g0,v)
    
    
    plt.plot(tau,G_fit,label = "Flow model")
    

    return


def plot_pure_diffusion_model(tau,g0,td):
    
    G_fit = diffusion(tau,g0,td)
    
    
    plt.plot(tau,G_fit,"orange",label = "Pure Diffusion model")
    

    return
    
def plot_anomalous_diffusion_model(tau,g0,td,a):
    
    G_fit = anomalous_D(tau, g0, td, a)
    
    plt.plot(tau,G_fit,"red",label = "Anomalous Diffusion model")
    
    return


def plot_combined_model(tau,g0,td,v):
    
    G_fit = combined_model(tau, g0, td, v)

    plt.plot(tau,G_fit,label = "Combined model")
    
    return

def fit_flow_model(t,G_exp,results,k,point,Q,rep):
    
    try:
        pars,cov = curve_fit(flow,t,G_exp,p0 = [0.1,10],bounds=((0,0),(np.inf,np.inf)))
        
        (g0,v) = pars
        
        
        
        plot_flow_model(t, *pars)
        
        
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"Flow velocity (mm/s)"] = np.abs(v)
        results.loc[k,"G$_0$ flow model"] = g0
        
        
      
        
        
    
    except:
        
        print(f"{Q} ul_min, rep {rep}, point {point} flow model not converged")
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"Flow velocity (mm/s)"] = np.nan
        results.loc[k,"G$_0$ flow model"] = np.nan

    return results


def fit_pure_diffusion_model(t,G_exp,results,k,point,Q,rep):
    
    try:
        pars,cov = curve_fit(diffusion,t,G_exp,bounds=((0,0),(np.inf,np.inf)))
        
        (g0,tau_d_pure) = pars
        
        D_pure = w**2/(4*tau_d_pure*1e-3)
        
        plot_pure_diffusion_model(t, *pars)
        
        
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"D$_{pure}$ ($\mu$m$^2$/s)"] = D_pure
        results.loc[k,"G$_0$ pure diffusion model"] = g0
        
        
        
        
        
    
    except:
        
        print(f"{Q} ul_min, rep {rep}, point {point} pure diffusion model not converged")
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"D$_{pure}$ ($\mu$m$^2$/s)"] = np.nan
        results.loc[k,"G$_0$ pure diffusion model"] = np.nan
 
    return results



def fit_anomalous_diffusion_model(t,G_exp,results,k,point,Q,rep):
    
    try:
        pars,cov = curve_fit(anomalous_D,t,G_exp)
        
        (g0,tau_d_anomalous,a) = pars
        
        D_anomalous = w**2/(4*tau_d_anomalous*1e-3)
        
        plot_anomalous_diffusion_model(t, *pars)
        
        
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"D$_{anomalous}$ ($\mu$m$^2$/s)"] = D_anomalous
        results.loc[k,r"Diffusion exponent ($\alpha$)"] = a
        results.loc[k,"G$_0$ anomalous diffusion model"] = g0
        
        
        
        
        
    
    except:
        
        print(f"{Q} ul_min, rep {rep}, point {point} anomalous diffusion model not converged")
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"D$_{anomalous}$ ($\mu$m$^2$/s)"] = np.nan
        results.loc[k,r"Diffusion exponent ($\alpha$)"] = np.nan
        results.loc[k,"G$_0$ anomalous diffusion model"] = np.nan
    return results



def fit_combined_model(t,G_exp,results,k,point,Q,rep):
    
    try:
        pars,cov = curve_fit(combined_model,t,G_exp)
        
        (g0,td,v) = pars
        
        D_combined = w**2/(4*td*1e-3)
        
        plot_combined_model(t, *pars)
        
        
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"Flow velocity combined model (mm/s)"] = np.abs(v)
        results.loc[k,"D$_{combined}$ ($\mu$m$^2$/s)"] = D_combined
        results.loc[k,"G$_0$ combined model"] = g0
        
        
        
        
        
    
    except:
        
        print(f"{Q} ul_min, rep {rep}, point {point} combined model not converged")
        results.loc[k,"Q ($\mu$l/min)"] = Q
        results.loc[k,"Repetition"] = rep
        results.loc[k,"Point"] = point
        results.loc[k,"Flow velocity combined model (mm/s)"] = np.nan
        results.loc[k,"D$_{combined}$ ($\mu$m$^2$/s)"] = np.nan
        results.loc[k,"G$_0$ combined model"] = np.nan
    
    return results










w = 0.23893  
z0 = 2.63658


results = pd.DataFrame()
k = 0
for rep in [1,2]:
    
    for Q in [0.1,1,10,20,50,100]:
        
        
        os.makedirs(f"Figures/Curves/Chip {rep} Q {Q} ul_min",exist_ok=True)
        
        try:
            data = pd.read_csv(f"Chip {rep}/Chip {rep} flow pofiles {Q} ul_min graphs.csv",skiprows=1,sep="\t",skipinitialspace=True)
        
            
            
        except:
            data = pd.read_csv(f"Chip {rep}/Chip {rep} flow pofiles {Q} ul_min repeated graphs.csv",skiprows=1,sep="\t",skipinitialspace=True)
            
        
        point = -35
        
       
        for col in data.columns:
            
            if "Correlation Channel" in col:
                
                G_exp = data.loc[:,col].dropna()
                
                t = data.loc[:len(G_exp)-1,"Time [ms]"]
                
                
                plt.plot(t,G_exp,".",label="Data")
                
            
                
                results = fit_flow_model(t, G_exp, results, k, round(point),Q,rep)
                
                #results = fit_pure_diffusion_model(t, G_exp, results, k, point,Q,rep)
                
                #results = fit_anomalous_diffusion_model(t, G_exp, results, k, point,Q,rep)
                
                results = fit_combined_model(t, G_exp, results, k, round(point),Q,rep)
                print(round(point))
                plt.ylabel(r"G($\tau$)")
                plt.xlabel("Time (ms)")    
                plt.xscale("log")
                plt.legend()
                plt.title(f"{Q} ul_min, rep {rep}, point {point}")
                #plt.savefig(f"Figures/Curves/Chip {rep} Q {Q} ul_min/Point {point}.tif",bbox_inches="tight")
                #plt.show()
                plt.close()
                
                point+=70/6
                k+=1


means = results.groupby(by=["Point","Q ($\mu$l/min)"],as_index = False).mean()


stds = results.groupby(by=["Point","Q ($\mu$l/min)"],as_index = False).std()


for Q in [0.1,1,10,20,50,100]:
    
    
    points = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Point"]
    vs = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity (mm/s)"]
    vs_err = stds.loc[stds.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity (mm/s)"]
    
    plt.errorbar(x = points, y = vs, yerr = vs_err,fmt = ".-",label=f"{Q}",capsize=2.5)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title = "Q ($\mu$l/min)")
plt.xlabel("Measurement Point ($\mu$m)")
plt.ylabel("Flow Velocity (mm/s)")
plt.xlim([-40,40])

plt.xticks(np.arange(-40,50,10))

plt.savefig("Figures/Flow profiles.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()

for Q in [0.1,1,10,20,50,100]:
    
    
    points = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Point"]
    vs = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"G$_0$ flow model"]
    vs_err = stds.loc[stds.loc[:,"Q ($\mu$l/min)"]==Q,"G$_0$ flow model"]
    
    plt.errorbar(x = points, y = vs, yerr = vs_err,fmt = ".-",label=f"{Q}",capsize=2.5)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title = "Q ($\mu$l/min)")
plt.xlabel("Measurement Point ($\mu$m)")
plt.ylabel("G$_0$")
plt.xlim([-40,40])

plt.xticks(np.arange(-40,50,10))

plt.savefig("Figures/Flow profiles G_0.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()

 


mean_v = means.groupby(by=["Q ($\mu$l/min)"],as_index = False).mean()

std_v = results.groupby(by=["Q ($\mu$l/min)"],as_index = False).sem()

fig, ax = plt.subplots()

ax.errorbar(mean_v.loc[:,"Q ($\mu$l/min)"],y = mean_v.loc[:,"Flow velocity (mm/s)"], yerr =  std_v.loc[:,"Flow velocity (mm/s)"],fmt = "k.",capsize=2.5)

Q_exp = np.array(mean_v.loc[:4,"Q ($\mu$l/min)"])

v_exp = np.array(mean_v.loc[:4,"Flow velocity (mm/s)"])

regression_result = linregress(Q_exp/60,v_exp)

m = regression_result.slope
b = regression_result.intercept
rvalue = regression_result.rvalue
pvalue = regression_result.pvalue
m_err = regression_result.stderr
b_err = regression_result.intercept_stderr

print(f"Measured Device area with pure flow model is A = {1/m*1e6} +- {1/m**2*m_err*1e6}")


Q_fit = np.arange(0,(100/60),0.001)
v_fit = m*Q_fit+b

ax.plot(Q_fit*60,v_fit,"-",color="#e8000b")


ax.set_xlabel("Flow Rate Q ($\mu$l/min)")
ax.set_ylabel("Average Flow Velocity (mm/s)")

ax2=plt.axes([0,0,1,1])


left, bottom, width, height = [0.6, 0.15, 0.35, 0.35]


ip = InsetPosition(ax, [left, bottom, width, height])
ax2.set_axes_locator(ip)
mark_inset(ax, ax2, loc1=2, loc2=3, fc="none", ec="k",alpha = 0.5)

ax2.errorbar(mean_v.loc[:1,"Q ($\mu$l/min)"],y = mean_v.loc[:1,"Flow velocity (mm/s)"], yerr =  std_v.loc[:1,"Flow velocity (mm/s)"],fmt = "k.",capsize=2.5)


Q_fit_inset = np.arange(0,1.1/60,0.0001)
v_fit_inset = m*Q_fit_inset+b

ax2.plot(Q_fit_inset*60,v_fit_inset,"-",color="#e8000b")

ax2.set_ylim([-0.1,2])
ax2.set_xlim([0,1.1])

ax2.set_yticks(np.arange(0,3,1))
ax2.set_xticks(np.arange(0,1.5,0.5))
ax2.set_xticklabels(np.arange(0,1.5,0.5))

ax2.xaxis.set_minor_locator(MultipleLocator(0.1))

ax2.yaxis.set_minor_locator(MultipleLocator(0.2))

plt.savefig("Figures/Avg flow velocity.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()




for Q in [0.1,1,10,20,50]:
    
    
    points = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Point"]
    vs = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity combined model (mm/s)"]
    vs_err = stds.loc[stds.loc[:,"Q ($\mu$l/min)"]==Q,"Flow velocity combined model (mm/s)"]
    
    plt.errorbar(x = points, y = vs, yerr = vs_err,fmt = ".-",label=f"{Q}",capsize=2.5)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title = "Q ($\mu$l / min)")
plt.xlabel("Measurement Point ($\mu$m)")
plt.ylabel("Flow Velocity (mm/s)")
plt.xlim([-40,40])

plt.xticks(np.arange(-40,50,10))

plt.savefig("Figures/Flow profiles combined.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


for Q in [0.1,1,10,20,50]:
    
    
    points = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"Point"]
    vs = means.loc[means.loc[:,"Q ($\mu$l/min)"]==Q,"G$_0$ combined model"]
    vs_err = stds.loc[stds.loc[:,"Q ($\mu$l/min)"]==Q,"G$_0$ combined model"]
    
    plt.errorbar(x = points, y = vs, yerr = vs_err,fmt = ".-",label=f"{Q}",capsize=2.5)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),title = "Q ($\mu$l/min)")
plt.xlabel("Measurement Point ($\mu$m)")
plt.ylabel("G$_0$")
plt.xlim([-40,40])

plt.xticks(np.arange(-40,50,10))

plt.savefig("Figures/Flow profiles G_0 combined model.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()




mean_v = means.groupby(by=["Q ($\mu$l/min)"],as_index = False).mean()

std_v = results.groupby(by=["Q ($\mu$l/min)"],as_index = False).sem()

fig, ax = plt.subplots()

ax.errorbar(mean_v.loc[:4,"Q ($\mu$l/min)"],y = mean_v.loc[:4,"Flow velocity combined model (mm/s)"], yerr =  std_v.loc[:4,"Flow velocity combined model (mm/s)"],fmt = "k.",capsize=2.5)

Q_exp = mean_v.loc[:4,"Q ($\mu$l/min)"]

v_exp = mean_v.loc[:4,"Flow velocity combined model (mm/s)"]

regression_result = linregress(Q_exp/60,v_exp)

m = regression_result.slope
b = regression_result.intercept
rvalue = regression_result.rvalue
pvalue = regression_result.pvalue
m_err = regression_result.stderr
b_err = regression_result.intercept_stderr

print(f"Measured Device area with combined model is A = {1/m*1e6} +- {1/m**2*m_err*1e6}")

Q_fit = np.arange(0,60/60,0.0001)
v_fit = m*Q_fit+b

ax.plot(Q_fit*60,v_fit,"-",color="#e8000b")


ax.set_xlabel("Flow Rate Q ($\mu$l/min)")
ax.set_ylabel("Average Flow Velocity (mm/s)")

ax2=plt.axes([0,0,1,1])


left, bottom, width, height = [0.6, 0.15, 0.35, 0.35]


ip = InsetPosition(ax, [left, bottom, width, height])
ax2.set_axes_locator(ip)
mark_inset(ax, ax2, loc1=2, loc2=3, fc="none", ec="k",alpha = 0.5)

ax2.errorbar(mean_v.loc[:1,"Q ($\mu$l/min)"],y = mean_v.loc[:1,"Flow velocity combined model (mm/s)"], yerr =  std_v.loc[:1,"Flow velocity combined model (mm/s)"],fmt = "k.",capsize=2.5)


Q_fit_inset = np.arange(0,1.1/60,0.0001)
v_fit_inset = m*Q_fit_inset+b

ax2.plot(Q_fit_inset*60,v_fit_inset,"r-")

ax2.set_ylim([-0.1,2])
ax2.set_xlim([0,1.1])

ax2.set_yticks(np.arange(0,3,1))
ax2.set_xticks(np.arange(0,1.5,0.5))
ax2.set_xticklabels(np.arange(0,1.5,0.5))

ax2.xaxis.set_minor_locator(MultipleLocator(0.1))

ax2.yaxis.set_minor_locator(MultipleLocator(0.2))

plt.savefig("Figures/Avg flow velocity combined model.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


fig, ax = plt.subplots()

ax.errorbar(mean_v.loc[:,"Q ($\mu$l/min)"],y = mean_v.loc[:,"Flow velocity (mm/s)"], yerr =  std_v.loc[:,"Flow velocity (mm/s)"],fmt = ".",capsize=2.5,label = "Pure Flow model")

ax.errorbar(mean_v.loc[:4,"Q ($\mu$l/min)"],y = mean_v.loc[:4,"Flow velocity combined model (mm/s)"], yerr =  std_v.loc[:4,"Flow velocity combined model (mm/s)"],fmt = ".",capsize=2.5,label = "Combined model")

ax.legend()


ax.set_xlabel("Flow Rate Q ($\mu$l/min)")
ax.set_ylabel("Average Flow Velocity (mm/s)")

ax2=plt.axes([0,0,1,1])


left, bottom, width, height = [0.6, 0.15, 0.35, 0.35]


ip = InsetPosition(ax, [left, bottom, width, height])
ax2.set_axes_locator(ip)
mark_inset(ax, ax2, loc1=2, loc2=3, fc="none", ec="k",alpha=0.5)


ax2.errorbar(mean_v.loc[:1,"Q ($\mu$l/min)"],y = mean_v.loc[:1,"Flow velocity (mm/s)"], yerr =  std_v.loc[:1,"Flow velocity combined model (mm/s)"],fmt = ".",capsize=2.5)


ax2.errorbar(mean_v.loc[:1,"Q ($\mu$l/min)"],y = mean_v.loc[:1,"Flow velocity combined model (mm/s)"], yerr =  std_v.loc[:1,"Flow velocity combined model (mm/s)"],fmt = ".",capsize=2.5)

ax2.set_ylim([-0.1,2])
ax2.set_xlim([0,1.1])

ax2.set_yticks(np.arange(0,3,1))
ax2.set_xticks(np.arange(0,1.5,0.5))
ax2.set_xticklabels(np.arange(0,1.5,0.5))

ax2.xaxis.set_minor_locator(MultipleLocator(0.1))

ax2.yaxis.set_minor_locator(MultipleLocator(0.2))


plt.savefig(r"Figures/Flow velocity comparison.tif",dpi=300,bbox_inches="tight")
plt.show()
plt.close()


fig, ax = plt.subplots()

plt.errorbar(mean_v.loc[:,"Q ($\mu$l/min)"],y = mean_v.loc[:,"D$_{combined}$ ($\mu$m$^2$/s)"], yerr =  std_v.loc[:,"D$_{combined}$ ($\mu$m$^2$/s)"],fmt = "k.-",capsize=2.5)

plt.xlabel("Q ($\mu$l/min)")

plt.ylabel("Diffusion Coefficient ($\mu$m$^2$/s)")

plt.yscale("log")
plt.xscale("log")

plt.savefig("Figures/D combined model",bbox_inches="tight",dpi=300)

plt.show()
plt.close()



fig = plt.figure()
plt.errorbar(mean_v.loc[:,"Q ($\mu$l/min)"],y = mean_v.loc[:,"G$_0$ flow model"], yerr =  std_v.loc[:,"G$_0$ flow model"],fmt = "o-",color="#023eff",capsize=2.5,label = "G$_0$")


plt.xlabel("Q ($\mu$l/min)")

plt.ylabel("G$_0$")

ax=plt.gca()

ax2 = ax.twinx()

ax2.errorbar(mean_v.loc[:,"Q ($\mu$l/min)"],y = mean_v.loc[:,"Flow velocity (mm/s)"], yerr =  std_v.loc[:,"Flow velocity (mm/s)"],fmt = "s-",color="#e8000b",capsize=2.5,label = "Flow velocity (mm/s)")

ax2.set_ylabel("Flow Velocity (mm/s)",color="#e8000b")
ax2.spines['right'].set_color('#e8000b')
ax2.tick_params(axis='y', colors='#e8000b')
ax.errorbar(np.nan,np.nan,np.nan,fmt="s",color="#e8000b",label = "Flow velocity",capsize=2.5)

ax.legend()

plt.savefig("Figures/G0 pure flow model.tif",bbox_inches="tight",dpi=300)


plt.show()
plt.close()



fig = plt.figure()
plt.errorbar(mean_v.loc[:4,"Q ($\mu$l/min)"],y = mean_v.loc[:4,"G$_0$ combined model"], yerr =  std_v.loc[:4,"G$_0$ combined model"],fmt = "o-",color="#023eff",capsize=2.5,label = "G$_0$")


plt.xlabel("Q ($\mu$l/min)")

plt.ylabel("G$_0$")

ax=plt.gca()

ax2 = ax.twinx()

ax2.errorbar(mean_v.loc[:4,"Q ($\mu$l/min)"],y = mean_v.loc[:4,"Flow velocity combined model (mm/s)"], yerr =  std_v.loc[:4,"Flow velocity combined model (mm/s)"],fmt = "s-",color="#e8000b",capsize=2.5,label = "Flow velocity (mm/s)")

ax2.set_ylabel("Flow Velocity (mm/s)",color="#e8000b")
ax2.spines['right'].set_color('#e8000b')
ax2.tick_params(axis='y', colors='#e8000b')
ax.errorbar(np.nan,np.nan,np.nan,fmt="s",color="#e8000b",label = "Flow velocity",capsize=2.5)

ax.legend()

plt.savefig("Figures/G0 combined model.tif",bbox_inches="tight",dpi=300)


plt.show()
plt.close()


mean_v.to_csv("Results means.txt",decimal=",",sep="\t",index=None)

std_v.to_csv("Results stds.txt",decimal=",",sep="\t",index=None)

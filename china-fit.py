import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import io
import requests
from utility import *

df_Ch = covid_country('China', url_total)
df_Ch = df_Ch.sum(numeric_only=True)
cases_china = df_Ch[2:].values

df_Ch_dead = covid_country('China', url_deaths)
df_Ch_dead = df_Ch_dead.sum(numeric_only=True)
deaths_china = df_Ch_dead[2:].values

df_Ch_reco = covid_country('China', url_deaths)
df_Ch_reco = df_Ch_reco.sum(numeric_only=True)
recovs_china = df_Ch_reco[2:].values

time = range(len(cases_china))


def f(x, M, k, x0):
  return M / (1 + np.exp(-k * (x - x0)))


def g(x, M, k, x0):
  return M * np.exp(k * (x - x0))


initial_guess = (80000, 0.1, 20)
upbound = [90000, 1, 50]
lowbound = [70000, 0, 0]

#sigma=np.array([np.sqrt(x) for x in n])
popt, pcov = curve_fit(
    f, time, cases_china, p0=initial_guess, bounds=[lowbound, upbound]
    )
#popt, pcov = optimize.curve_fit(Gaussian,bins20,binData20,sigma=sigma)

print('best fitted parameters (M,k,x0) : ')
print(popt)
print('covariance matrix : ')
print(pcov)

#To get parameters uncertainties we take the square root from diagonal elements of the covariance matrix
print('fitted M = %.4f +- %.4f ' % (popt[0], np.sqrt(np.diag(pcov)[0])))
print('fitted k = %.4f +- %.4f ' % (popt[1], np.sqrt(np.diag(pcov)[1])))
print('fitted x0 = %.4f +- %.4f ' % (popt[2], np.sqrt(np.diag(pcov)[2])))

fitted_M = [popt[0], np.sqrt(np.diag(pcov)[0])]
fitted_k = [popt[1], np.sqrt(np.diag(pcov)[1])]
fitted_x0 = [popt[2], np.sqrt(np.diag(pcov)[2])]

plt.figure(figsize=[12, 9])
plt.errorbar(time, cases_china, fmt='o', ecolor='g', c='g', label="data")
plt.plot(
    time,
    f(time, fitted_M[0], fitted_k[0], fitted_x0[0]),
    'r',
    label='Logistic fit'
    )
time_truncated = np.linspace(0, 17, 21)
plt.plot(
    time_truncated,
    g(time_truncated, fitted_M[0], fitted_k[0], fitted_x0[0]),
    'b',
    linestyle='--',
    label='Exponential growth'
    )
plt.hlines(
    fitted_M[0], 0, time[-1], colors='grey', linestyle='--', label='Plateau'
    )
plt.legend(loc="upper left")
plt.grid()
plt.xlabel('days from 22/01/20')
plt.ylabel('positive cases')
plt.title('COVID-19 positive cases in China')
#plt.text(1, 50000, r'1/(1 + exp(-k(x-x0)))' % (fitted_M[0], fitted_M[1]), bbox=dict(facecolor='yellow', alpha=0.7))
plt.text(
    1,
    65000,
    r'M = %1.0f $\pm$ %1.0f' % (fitted_M[0], fitted_M[1]),
    bbox=dict(facecolor='yellow', alpha=0.7)
    )
plt.text(
    1,
    60000,
    r'k = %1.4f $\pm$ %1.4f' % (fitted_k[0], fitted_k[1]),
    bbox=dict(facecolor='yellow', alpha=0.7)
    )
plt.text(
    1,
    55000,
    r'$x_0$ = %1.2f $\pm$ %1.2f' % (fitted_x0[0], fitted_x0[1]),
    bbox=dict(facecolor='yellow', alpha=0.7)
    )
plt.show()
plt.savefig("plot/china_fit.png")

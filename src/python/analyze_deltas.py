import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import datetime

from scipy.stats import sem, t
from scipy import mean

from scipy import stats

f = open('data/spin/46aa2776-91e9-11ea-8090-8a41a21ee60e/order.out', 'r')
f1 = f.readlines()

raw_deltas = []
prev = datetime.datetime.strptime(f1[0],"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
f1 = f1[1:]
for line in f1:
    now = datetime.datetime.strptime(line,"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
    if (now - prev > 5):
        continue
    raw_deltas.append(now - prev)
    prev = now

f = open('data/spin/af5b4516-91e9-11ea-a2d0-5862c1a46ee4/order.out', 'r')
f1 = f.readlines()

prev = datetime.datetime.strptime(f1[0],"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
f1 = f1[1:]
for line in f1:
    now = datetime.datetime.strptime(line,"%Y-%m-%d %H:%M:%S.%f UTC\n").timestamp()
    if (now - prev > 5):
        continue
    raw_deltas.append(now - prev)
    prev = now

deltas = np.array(raw_deltas)

low, high = np.percentile(deltas[deltas > 0.005], [2.5, 97.5])
mean = low + (high - low) / 2
delta = mean - low
print (low)
print (high)
print (np.min(deltas[deltas > 0.005]))
print (np.max(deltas[deltas > 0.005]))
print ("Signal TCP latency " + str(mean) + " ± " + str(delta))
print (len(deltas[(deltas >= low) & (deltas <= high)])/len(deltas[deltas > 0.005]))

low2, high2 = np.percentile(deltas[deltas <= 0.005], [2.5, 97.5])
mean2 = low2 + (high2 - low2) / 2
delta = mean2 - low2
print (low2)
print (high2)
print (np.min(deltas[deltas <= 0.005]))
print (np.max(deltas[deltas <= 0.005]))
print ("Signal TCP latency " + str(mean2) + " ± " + str(delta))
print (len(deltas[(deltas >= low2) & (deltas <= high2)])/len(deltas[deltas <= 0.005]))

# print (len(deltas[(deltas >= 0.000120) & (deltas <= 0.005)]))
print (len(deltas[deltas <= 0.005]))
print (len(deltas[deltas > 0.005]))

# fig, ax = plt.subplots()
# bins = np.arange(0.00005, 0.0002, 0.000005)
# print (bins)
# sns.distplot(deltas[deltas <= 0.005], bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', fit=stats.lognorm)
# sns.distplot(deltas[deltas > 0.005], kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', fit=stats.lognorm)
# print (stats.lognorm.fit(deltas[deltas <= 0.005]))
# print (stats.lognorm.fit(deltas[deltas > 0.005]))
plt.show()
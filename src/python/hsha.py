import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
from numpy.core import concatenate

from scipy import stats
from scipy import spatial
from matplotlib.pyplot import hist
from numpy.core._multiarray_umath import exp

# tbctf 6dba0952-8a1f-11ea-8d80-4fe9d50ff731-2/ -k route 'protocol_out_market_data:best_price,protocol_in_market_data:best_price' -o -k route 'protocol_out_order_entry:order,tr_fw:received_order_add' --no_protocol_translation -k layer trees -u us | grep duration | sed -En 's/^.*duration ([0-9]+) µs/\1/p' > raw_numbers

# example data
hsha1data = genfromtxt("data/hsha/sc_se_latency")
hsha2data = genfromtxt("data/hsha/se_sc_latency")

# print (np.quantile(hsha1data, [0.5, 0.8, 0.9, 0.95, 0.99]))
# print (np.quantile(hsha2data, [0.5, 0.8, 0.9, 0.95, 0.99]))

# low1, high1 = np.percentile(hsha1data, [2.5, 97.5])
# mean1 = low1 + (high1 - low1) / 2
# delta = mean1 - low1
# print (np.min(hsha1data))
# print (np.max(hsha1data))
# print ("Signal TCP latency " + str(mean1) + " ± " + str(delta))
# print (len(hsha1data[(hsha1data >= low1) & (hsha1data <= high1)])/len(hsha1data))

# low2, high2 = np.percentile(hsha2data, [2.5, 97.5])
# mean2 = low2 + (high2 - low2) / 2
# delta = mean2 - low2
# print (np.min(hsha2data))
# print (np.max(hsha2data))
# print ("Signal TCP latency " + str(mean2) + " ± " + str(delta))
# print (len(hsha2data[(hsha2data >= low2) & (hsha2data <= high2)])/len(hsha2data))

base_count = 25
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

sns.distplot(np.clip(hsha1data, bins[0], bins[-1]), bins=bins, kde=False, hist_kws={'edgecolor':'black'}, color='#3782CC', label='data')
# sns.distplot(np.clip(hsha2data, bins[0], bins[-1]), bins=bins, kde=False, hist_kws={'edgecolor':'black'}, color='#FF0000', label='Процесс-обработчик → Процесс-шлюз')

# Good fit
data = stats.norm.rvs(loc=10.78,scale=0.96,size=2*120000)
data = np.append(data, stats.norm.rvs(loc=13.77,scale=0.7,size=2*157000))
data = np.append(data, stats.norm.rvs(loc=16.4,scale=1.07,size=2*95000))
# data = stats.norm.rvs(loc=10.78,scale=1.3,size=2*120000)
# data = np.append(data, stats.norm.rvs(loc=13.77,scale=0.7,size=2*157000))
# data = np.append(data, stats.norm.rvs(loc=16.4,scale=1.07,size=2*95000))
sns.distplot(np.clip(data, bins[0], bins[-1]), bins=bins, kde=False, hist_kws={'edgecolor':'black'}, color='#FF0000', label='simulated')


xlabels = bins[0:-1].astype(str)
xlabels[-1] += "+"
plt.xlim([0, num_bins - 1])
N_labels = len(xlabels)
plt.xticks(np.arange(N_labels) + 0.5)
ax.set_xticklabels(xlabels)
ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=24)

plt.legend(prop={'size': 24}, title_fontsize = 24, title = 'Направление взаимодействия')
ax.set_xlabel('Временная задержка на передачу данных между процессами, мкс', fontsize=24)
ax.set_ylabel('Вероятность', fontsize=24)
# ax.set_title('Гистограмма временной задержки на передачу данных')

fig.tight_layout()
plt.show()

# cdata, cdatay, _ = hist(data, bins=bins)
# chsha1data, chshay, _ = hist(hsha1data, bins=bins)
# print (cdata, chsha1data)
# print (cdatay, chshay)
# minlen = min(len(hsha1data), len(data))
# print (1 - spatial.distance.cosine(chsha1data, cdata))

# def gauss(x,mu,sigma,A):
#     return A*stats.norm.cdf(x,mu, sigma)

# def trimodal_cdf(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
#     return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)

# data = stats.norm.rvs(loc=10.78,scale=0.96,size=2*120000)
# data = np.append(data, stats.norm.rvs(loc=13.77,scale=0.7,size=2*157000))
# data = np.append(data, stats.norm.rvs(loc=16.4,scale=1.07,size=2*95000))
# print (stats.kstest(hsha1data, cdf=trimodal_cdf, args=(10.78,0.96,240000/((240+157*2+2*95)*1000),13.77,0.7,2*157000/((240+157*2+2*95)*1000),16.4,1.07,2*95000/((240+157*2+2*95)*1000))))
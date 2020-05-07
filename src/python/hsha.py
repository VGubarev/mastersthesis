import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

# tbctf 6dba0952-8a1f-11ea-8d80-4fe9d50ff731-2/ -k route 'protocol_out_market_data:best_price,protocol_in_market_data:best_price' -o -k route 'protocol_out_order_entry:order,tr_fw:received_order_add' --no_protocol_translation -k layer trees -u us | grep duration | sed -En 's/^.*duration ([0-9]+) µs/\1/p' > raw_numbers

# example data
hsha1data = genfromtxt("data/hsha/sc_se_latency")
hsha2data = genfromtxt("data/hsha/se_sc_latency")

low1, high1 = np.percentile(hsha1data, [2.5, 97.5])
mean1 = low1 + (high1 - low1) / 2
delta = mean1 - low1
print (np.min(hsha1data))
print (np.max(hsha1data))
print ("Signal TCP latency " + str(mean1) + " ± " + str(delta))
print (len(hsha1data[(hsha1data >= low1) & (hsha1data <= high1)])/len(hsha1data))

low2, high2 = np.percentile(hsha2data, [2.5, 97.5])
mean2 = low2 + (high2 - low2) / 2
delta = mean2 - low2
print (np.min(hsha2data))
print (np.max(hsha2data))
print ("Signal TCP latency " + str(mean2) + " ± " + str(delta))
print (len(hsha2data[(hsha2data >= low2) & (hsha2data <= high2)])/len(hsha2data))

base_count = 25
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

sns.distplot(np.clip(hsha1data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', label='Процесс-шлюз → Процесс-обработчик')
sns.distplot(np.clip(hsha2data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#FF0000', label='Процесс-обработчик → Процесс-шлюз')

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
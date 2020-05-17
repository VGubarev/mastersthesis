import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

# tbctf 6dba0952-8a1f-11ea-8d80-4fe9d50ff731-2/ -k route 'protocol_out_market_data:best_price,protocol_in_market_data:best_price' -o -k route 'protocol_out_order_entry:order,tr_fw:received_order_add' --no_protocol_translation -k layer trees -u us | grep duration | sed -En 's/^.*duration ([0-9]+) µs/\1/p' > raw_numbers

# example data
lfdata = genfromtxt("data/lf/sc_se_latency")
lfdata = np.append(lfdata, genfromtxt("data/lf/se_sc_latency"))
hshadata = genfromtxt("data/hsha/sc_se_latency")
hshadata = np.append(hshadata, genfromtxt("data/hsha/se_sc_latency"))

base_count = 25
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

# pois = np.random.gamma(3.3, 0.8, x.size) + 7
sns.distplot(np.clip(lfdata, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', label='Ведущий/Ведомые')
sns.distplot(np.clip(hshadata, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#FF0000', label='Полусинхронный/Полуреактивный')
xlabels = bins[0:-1].astype(str)
xlabels[-1] += "+"
plt.xlim([0, num_bins - 1])
N_labels = len(xlabels)
plt.xticks(np.arange(N_labels) + 0.5)
ax.set_xticklabels(xlabels)

ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop={'size': 20}, title_fontsize = 24, title = 'Метод взаимодействия')
ax.set_xlabel('Временная задержка на передачу данных, мкс', fontsize=24)
ax.set_ylabel('Вероятность', fontsize=24)

fig.tight_layout()
plt.show()
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

from scipy import stats

# tbctf 6dba0952-8a1f-11ea-8d80-4fe9d50ff731-2/ -k route 'protocol_out_market_data:best_price,protocol_in_market_data:best_price' -o -k route 'protocol_out_order_entry:order,tr_fw:received_order_add' --no_protocol_translation -k layer trees -u us | grep duration | sed -En 's/^.*duration ([0-9]+) µs/\1/p' > raw_numbers

# example data
# enginedata = genfromtxt("data/lf/se_latency")
enginedata = genfromtxt("data/hsha/se_latency")
# enginedata = np.append(enginedata, genfromtxt("data/tcp/se_latency"))
# enginedata = np.append(enginedata, genfromtxt("data/spin/se_latency"))
# enginedata = np.append(enginedata, genfromtxt("data/pure_tcp/se_latency"))

print (np.quantile(enginedata, [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1]))

# base_count = 29
# bins = np.arange(0, base_count, 1)
# num_bins = bins.size

# fig, ax = plt.subplots()

# sns.distplot(np.clip(enginedata, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC')
# xlabels = bins[0:-1].astype(str)
# xlabels[-1] += "+"
# plt.xlim([0, num_bins - 1])
# N_labels = len(xlabels)
# plt.xticks(np.arange(N_labels) + 0.5)
# ax.set_xticklabels(xlabels)
# ax.tick_params(axis='both', which='major', labelsize=16)

# ax.set_xlabel('Время обслуживание обслуживания заявки, мкс', fontsize=24)
# ax.set_ylabel('Вероятность', fontsize=24)

# fig.tight_layout()
# plt.show()
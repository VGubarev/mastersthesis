import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

from scipy import stats

# example data
tcp1data = genfromtxt("data/tcp/sc_se_latency")
tcp2data = genfromtxt("data/tcp/se_sc_latency")

tcp1data = tcp1data[tcp1data <= 3 * np.median(tcp1data)]
mu, sigma = stats.norm.fit(tcp1data)
lo, hi = stats.norm.interval(0.95, mu, sigma)
print (len(tcp1data[(tcp1data >= lo) & (tcp1data <= hi)])/len(tcp1data))
print ("Signal TCP latency " + str(mu) + " ± " + str(hi - mu))


# means = [np.mean(np.random.choice(tcp1data,size=len(tcp1data),replace=True)) for i in range(5000)]
# low, high =  np.percentile(means, [2.5, 97.5])
# print (np.mean(means))
# print (low, high)
# print (len(tcp1data[(tcp1data >= low) & (tcp1data <= high)])/len(tcp1data))
# print (np.mean(means))
# print (np.mean(stddevs))

tcp2data = tcp2data[tcp2data <= 3 * np.median(tcp2data)]
mu, sigma = stats.norm.fit(tcp2data)
lo, hi = stats.norm.interval(0.95, mu, sigma)
print (len(tcp2data[(tcp2data >= lo) & (tcp2data <= hi)])/len(tcp2data))
print ("Signal TCP latency " + str(mu) + " ± " + str(hi - mu))

# print (np.quantile(tcp1data, [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1]))
# print (np.quantile(tcp2data, [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1]))

# means = []
# stddevs = []
# for i in range(5000):
#     samples = np.random.choice(tcp2data, size=len(tcp2data), replace=True)
#     means.append(stats.lognorm.mean(samples))
#     stddevs.append(stats.lognorm.std(samples))

# sns.distplot(means, color='r', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675))
# sns.distplot(stddevs, color='r', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675))

# print (np.mean(means))
# print (np.mean(stddevs))

# means = [np.mean(np.random.choice(tcp2data,size=len(tcp2data),replace=True)) for i in range(5000)]
# low, high =  np.percentile(means, [2.5, 97.5])
# print (np.mean(means))
# print (low, high)
# print (len(tcp2data[(tcp2data >= low) & (tcp2data <= high)])/len(tcp2data))
# sns.distplot(means, color='r', kde=True, hist_kws=dict(edgecolor="b", linewidth=.675))

base_count = 40
bins = np.arange(0, base_count, 1)
num_bins = bins.size

# fig, ax = plt.subplots()

sns.distplot(np.clip(tcp1data, bins[0], bins[-1]), bins=bins, kde=False, hist_kws={'edgecolor':'black'}, color='#3782CC', label='Шлюз → Обработчик, ' + str(mean1) + " ± " + str(delta) + ' мкс')
sns.distplot(np.clip(tcp2data, bins[0], bins[-1]), bins=bins, kde=False,  hist_kws={'edgecolor':'black'}, color='#23a4ad', label='Обработчик → Шлюз, ' + str(mean2) + " ± " + str(delta2) + ' мкс')
# sns.distplot(np.clip(stats.lognorm.rvs(np.mean(means), np.mean(stddevs), size=len(tcp2data)), bins[0], bins[-1]))
# # print (np.mean(means))
# # print (np.mean(stddevs))

# xlabels = bins[0:-1].astype(str)
# xlabels[-1] += "+"
# plt.xlim([0, num_bins - 1])
# N_labels = len(xlabels)
# plt.xticks(np.arange(N_labels) + 0.5)
# ax.set_xticklabels(xlabels)
# ax.tick_params(axis='x', which='major', labelsize=16)
# ax.tick_params(axis='y', which='major', labelsize=24)

# plt.legend(prop={'size': 24}, title_fontsize = 24, title = 'Направление взаимодействия')
# ax.set_xlabel('Временная задержка на передачу данных между процессами, мкс', fontsize=24)
# ax.set_ylabel('Вероятность', fontsize=24)
# # ax.set_title('Гистограмма временной задержки на передачу данных')

# fig.tight_layout()
# plt.show()
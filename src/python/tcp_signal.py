import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

# example data
tcp1data = genfromtxt("data/tcp/sc_se_latency")
tcp2data = genfromtxt("data/tcp/se_sc_latency")

low1, high1 = np.percentile(tcp1data, [2.5, 97.5])
mean1 = low1 + (high1 - low1) / 2
delta = mean1 - low1
print (np.min(tcp1data))
print (np.max(tcp1data))
print ("Signal TCP latency " + str(mean1) + " ± " + str(delta))
print (len(tcp1data[(tcp1data >= low1) & (tcp1data <= high1)])/len(tcp1data))

low2, high2 = np.percentile(tcp2data, [2.5, 97.5])
mean2 = low2 + (high2 - low2) / 2
delta = mean2 - low2
print (np.min(tcp2data))
print (np.max(tcp2data))
print ("Signal TCP latency " + str(mean2) + " ± " + str(delta))
print (len(tcp2data[(tcp2data >= low2) & (tcp2data <= high2)])/len(tcp2data))

base_count = 40
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

sns.distplot(np.clip(tcp1data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', label='Процесс-шлюз → Процесс-обработчик')
sns.distplot(np.clip(tcp2data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#FF0000', label='Процесс-обработчик → Процесс-шлюз')

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
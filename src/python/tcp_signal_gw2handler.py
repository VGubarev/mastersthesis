import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

# example data
tcp1data = genfromtxt("data/tcp/sc_se_latency")
tcp2data = genfromtxt("data/tcp/se_sc_latency")

tcp1median = np.median(tcp1data)
tcp1confidence = np.max([(tcp1median - np.percentile(tcp1data, 2.75)) , (np.percentile(tcp1data, 97.5) - tcp1median)])
print ("GW to Handler latency: " + str(tcp1median) + " ± " + str(tcp1confidence))

tcp2median = np.median(tcp2data)
tcp2confidence = np.max([(tcp2median - np.percentile(tcp2data, 2.75)) , (np.percentile(tcp2data, 97.5) - tcp2median)])
print ("Handler to GW latency: " + str(tcp2median) + " ± " + str(tcp2confidence))

base_count = 40
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

sns.distplot(np.clip(tcp1data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', label='85/96/104 байт')
sns.distplot(np.clip(tcp2data, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#FF0000', label='530 байт')
xlabels = bins[0:-1].astype(str)
xlabels[-1] += "+"
plt.xlim([0, num_bins - 1])
N_labels = len(xlabels)
plt.xticks(np.arange(N_labels) + 0.5)
ax.set_xticklabels(xlabels)

plt.legend(prop={'size': 16}, title = 'Размер передаваемых сообщений')
ax.set_xlabel('Временная задержка на передачу данных между процессами, мкс')
ax.set_ylabel('Вероятность временной задержки на передачу данных')
# ax.set_title('Гистограмма временной задержки на передачу данных')

fig.tight_layout()
plt.show()
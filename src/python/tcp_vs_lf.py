import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

np.random.seed(19680801)

# example data
lfdata = genfromtxt("data/lf/se_sc_latency")
np.append(lfdata, genfromtxt("data/lf/sc_se_latency"))
tcpdata = genfromtxt("data/tcp/se_sc_latency")
np.append(tcpdata, genfromtxt("data/tcp/sc_se_latency"))

base_count = 43
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

# pois = np.random.gamma(3.3, 0.8, x.size) + 7
sns.distplot(np.clip(lfdata, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#3782CC', label='Мультиплексор в разделяемой памяти + futex')
sns.distplot(np.clip(tcpdata, bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color='#FF0000', label='TCP + select (число соединений < 150)')
xlabels = bins[0:-1].astype(str)
xlabels[-1] += "+"
plt.xlim([0, num_bins - 1])
N_labels = len(xlabels)
plt.xticks(np.arange(N_labels) + 0.5)
ax.set_xticklabels(xlabels)

plt.legend(prop={'size': 16}, title = 'Методы оповещения о появлении данных в разделяемой памяти')
ax.set_xlabel('Временная задержка на передачу данных между процессами, мкс')
ax.set_ylabel('Вероятность временной задержки на передачу данных')
ax.set_title('Гистограмма временной задержки на передачу данных')

fig.tight_layout()
plt.show()
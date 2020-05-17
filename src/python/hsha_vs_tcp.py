import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns

from scipy import stats

# tbctf 6dba0952-8a1f-11ea-8d80-4fe9d50ff731-2/ -k route 'protocol_out_market_data:best_price,protocol_in_market_data:best_price' -o -k route 'protocol_out_order_entry:order,tr_fw:received_order_add' --no_protocol_translation -k layer trees -u us | grep duration | sed -En 's/^.*duration ([0-9]+) µs/\1/p' > raw_numbers

# example data
# enginedata = genfromtxt("data/lf/se_latency")
# enginedata = np.append(enginedata, genfromtxt("data/hsha/se_latency"))
# enginedata = np.append(enginedata, genfromtxt("data/tcp/se_latency"))
# enginedata = np.append(enginedata, genfromtxt("data/spin/se_latency"))
tests = [
# ('tcp/se_sc_latency','TCP, Шлюз → Обработчик', '#3782CC'),
# ('tcp/sc_se_latency','TCP, Обработчик → Шлюз', '#23a4ad'),
('hsha/se_sc_latency','HSHR, Шлюз → Обработчик', '#d1291d'),
('hsha/sc_se_latency','HSHR, Обработчик → Шлюз', '#781d13'),
]
data = []
for test in tests:
    data.append((test, genfromtxt("data/" + test[0])))
    low1, high1 = np.percentile(data[-1][1], [2.5, 97.5])
    mean1 = low1 + (high1 - low1) / 2
    delta = mean1 - low1
    print ("Test " + test[0])
    print (np.min(data[-1][1]))
    print (np.max(data[-1][1]))
    print ("latency " + str(mean1) + " ± " + str(delta))
    print (len(data[-1][1][(data[-1][1] >= low1) & (data[-1][1] <= high1)])/len(data[-1][1]))
    print ("---------------")



base_count = 38
bins = np.arange(0, base_count, 1)
num_bins = bins.size

fig, ax = plt.subplots()

i = 0
for d in data:
    low1, high1 = np.percentile(d[1], [2.5, 97.5])
    mean1 = low1 + (high1 - low1) / 2
    delta = mean1 - low1
    print ("latency " + str(mean1) + " ± " + str(delta))
    sns.distplot(np.clip(d[1], bins[0], bins[-1]), bins=bins, kde=False, norm_hist=True, hist_kws={'edgecolor':'black'}, color=d[0][2], label=d[0][1] + ', ' + str(mean1) + " ± " + str(delta) + ' мкс')

xlabels = bins[0:-1].astype(str)
xlabels[-1] += "+"
plt.xlim([0, num_bins - 1])
N_labels = len(xlabels)
plt.xticks(np.arange(N_labels) + 0.5)
ax.set_xticklabels(xlabels)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.legend(prop={'size': 24}, title_fontsize = 24, title = 'Метод и направление взаимодействия')
ax.set_xlabel('Временная задержка на передачу данных, мкс', fontsize=24)
ax.set_ylabel('Вероятность', fontsize=24)

fig.tight_layout()
plt.show()
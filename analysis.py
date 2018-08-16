# knn analysis. Compare k values.
from matplotlib import pyplot as plt
import pandas as pd

# data = pd.read_csv('att_knn.csv')
# p = data.nlargest(10, 'akurasi')
# print('YALE')
# print(p)
# p1 = data[data.k==1.0].sort_values('akurasi')['akurasi'].values
# p3 = data[data.k==3.0].sort_values('akurasi')['akurasi'].values
# p5 = data[data.k==5.0].sort_values('akurasi')['akurasi'].values
# p7 = data[data.k==7.0].sort_values('akurasi')['akurasi'].values
# p9 = data[data.k==9.0].sort_values('akurasi')['akurasi'].values
# plt.plot(p1, '-')
# plt.plot(p3, '--')
# plt.plot(p5, '-.')
# plt.plot(p7, ':')
# plt.plot(p9, ',')
# plt.legend(['k=1', 'k=3', 'k=5', 'k=7', 'k=9'], loc='lower right')
# plt.xlabel("Experiment#")
# plt.ylabel("% Accuracy (sorted)")
# plt.show()

data1 = pd.read_csv('results/att_knn.csv')
data2 = pd.read_csv('results/jaffe_knn.csv')
data3 = pd.read_csv('results/yale_knn.csv')
a1 = data1[(data1.radius==16.0) & (data1.k==1.0)]['akurasi'].values
p1 = data1[(data1.radius==16.0) & (data1.k==1.0)]['points'].values
a2 = data2[(data2.radius==16.0) & (data2.k==1.0)]['akurasi'].values
p2 = data2[(data2.radius==16.0) & (data2.k==1.0)]['points'].values
a3 = data3[(data3.radius==16.0) & (data3.k==1.0)]['akurasi'].values
p3 = data3[(data3.radius==16.0) & (data3.k==1.0)]['points'].values
plt.plot(p1, a1, '-')
plt.plot(p2, a2, '--')
plt.plot(p3, a3, '-.')

plt.legend(['AT&T', 'JAFFE', 'YALE'], loc='lower right')
plt.xlabel("Point# P")
plt.ylabel("% Accuracy")
plt.show()
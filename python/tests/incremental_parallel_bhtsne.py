import gzip
import pickle
import numpy as np
import matplotlib
from cycler import cycler
import urllib
import os
import sys
from MulticoreTSNE import MulticoreTSNE as TSNE
import time
import pylab as Plot


def n_nearest_neighbor_error(n, r, label):
    res = r
    length = len(res)
    i = 0
    result = []
    for i in range(0, length):
        temp = []
        for j in range(0, length):
            if i != j:
                dis = (res[i][0] - res[j][0]) * (res[i][0] - res[j][0]) + (res[i][1] - res[j][1]) * (
                    res[i][1] - res[j][1])
                temp.append(dis)
        temp_val = [100000000] * n
        temp_index = [0] * n
        for k in range(0, length - 1):
            for h in range(0, n):
                if temp[k] < temp_val[h]:
                    m = n - 1
                    while m > h:
                        temp_val[m] = temp_val[m - 1]
                        temp_index[m] = temp_index[m - 1]
                        m -= 1
                    temp_val[h] = temp[k]
                    if k >= i:
                        temp_index[h] = k + 1
                    else:
                        temp_index[h] = k
                    break
        result.append(temp_index)
    j = 0
    for i in range(0, length):
        count = 0
        for y in result[i]:
            if label[i] == label[y]:
                count += 1
        if count * 2 < n:
            j += 1
    return float(j) / length

color = {
    0: (1, 0, 0),
    1: (0, 1, 0),
    2: (0, 0, 1),
    3: (1, 1, 0),
    4: (0, 1, 1),
    5: (1, 0, 1),
    6: (0, 0, 0),
    7: (0.1, 0.5, 0.9),
    8: (0.9, 0.5, 0.1),
    9: (0.1, 0.2, 0.7)
}

total_label_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_labels.txt')
total_data_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_X.txt')
total_label = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/label.txt')
total_data = []
for i in range(20):
    temp = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/data' + str(i) + '.txt')
    total_data.append(temp)
label_2500_1 = total_label_2500[0:2000]
data_2500_1 = total_data_2500[0:2000, :]
label_1 = total_label[0:1500]
data_1 = [x[0:1500, :] for x in total_data]

label_2500_2 = np.concatenate((total_label_2500[0:1500], total_label_2500[2000:2500]))
data_2500_2 = np.concatenate((total_data_2500[0:1500, :], total_data_2500[2000:2500, :]))
label_2 = np.concatenate((total_label[0:1000], total_label[1500:2000]))
data_2 = [np.concatenate((x[0:1000, :], x[1500:2000, :])) for x in total_data]



result = []
perp = 8
n_jobs = 4
max_iter = 600

# init run
tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
embedding_array = tsne.fit_transform(data_2500_1)
result_2500 = embedding_array
Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_1[0:1500]])
Plot.savefig('init1 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
for i in range(20):
    embedding_array = tsne.fit_transform(data_1[i])
    result.append(embedding_array)
    Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_1[0:1000]])
    Plot.savefig('init1 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()



tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
ti = time.time()
embedding_array = tsne.fit_transform(data_2500_2)
ti = time.time() - ti
nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500_2)
nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500_2)
nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500_2)
Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500_2])
Plot.savefig('init2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
Plot.savefig('init2 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
print('init2 data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1, 'nne3:', nne3, 'nne5:', nne5)
nne1 = 0
nne3 = 0
nne5 = 0
ti = 0
for i in range(20):
    t = time.time()
    embedding_array = tsne.fit_transform(data_2[i])
    ti += time.time() - t
    nne1 += n_nearest_neighbor_error(1, embedding_array, label_2)
    nne3 += n_nearest_neighbor_error(3, embedding_array, label_2)
    nne5 += n_nearest_neighbor_error(5, embedding_array, label_2)
    Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2])
    Plot.savefig('init2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()
    Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
    Plot.savefig('init2 old  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()
print('init2 data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:', nne1 / 20, 'nne3:',
      nne3 / 20, 'nne5:', nne5 / 20)

#保存降维结果到文件
#f = open('result_2500.txt', 'w')
#length = len(result_2500)
#i = 0
#while i < length:
#    f.write(str(result_2500[i, 0]) + '\t' + str(result_2500[i, 1]))
#    if i != length - 1:
#        f.write('\n')
#    i += 1
#for j in range(20):
#    f = open('result' + str(j) + '.txt', 'w')
#    length = len(result[j])
#    i = 0
#    while i < length:
#        f.write(str(result[j][i, 0]) + '\t' + str(result[j][i, 1]))
#        if i != length - 1:
#            f.write('\n')
#        i += 1




#result_2500 = np.loadtxt('result_2500.txt')
#for i in range(20):
#    result.append(np.loadtxt('result' + str(i) + '.txt'))

# incremental run old_num = 1500/2000 和 1000/1500
tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
ti = time.time()
embedding_array = tsne.fit_transform(data_2500_2, 1500, result_2500[0:1500, :])
ti = time.time() - ti
nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500_2)
nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500_2)
nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500_2)
Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500_2])
Plot.savefig('incremental  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
Plot.scatter(result_2500[0:1500, 0], result_2500[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
Plot.savefig('incremental true old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
Plot.scatter(embedding_array[0:1500, 0], embedding_array[0:1500, 1], 10, c=[color[x] for x in label_2500_2[0:1500]])
Plot.savefig('incremental old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
Plot.close()
print('incremental data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1, 'nne3:', nne3, 'nne5:', nne5)
nne1 = 0
nne3 = 0
nne5 = 0
ti = 0
for i in range(20):
    t = time.time()
    embedding_array = tsne.fit_transform(data_2[i], 1000, result[i][0:1000, :])
    ti += time.time() - t
    nne1 += n_nearest_neighbor_error(1, embedding_array, label_2)
    nne3 += n_nearest_neighbor_error(3, embedding_array, label_2)
    nne5 += n_nearest_neighbor_error(5, embedding_array, label_2)
    Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2])
    Plot.savefig('incremental  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()
    Plot.scatter(result[i][0:1000, 0], result[i][0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
    Plot.savefig('incremental true old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()
    Plot.scatter(embedding_array[0:1000, 0], embedding_array[0:1000, 1], 10, c=[color[x] for x in label_2[0:1000]])
    Plot.savefig('incremental old perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
    Plot.close()
print('incremental data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:', nne1 / 20, 'nne3:',
      nne3 / 20, 'nne5:', nne5 / 20)

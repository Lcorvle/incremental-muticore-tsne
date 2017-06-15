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

label_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_labels.txt')
data_2500 = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/mnist2500_X.txt')
label = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/label.txt')
data = []
for i in range(20):
    temp = np.loadtxt('C:/Users/xsx19/laboratory/t-SNE/bhtsne/data' + str(i) + '.txt')
    data.append(temp)


def test(mode, perp, max_iter, n_jobs):
    if mode == 1:
        ti = 0
        while n_jobs < 9:
            tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
            ti = time.time()
            embedding_array = tsne.fit_transform(data_2500)
            ti = time.time() - ti
            nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500)
            nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500)
            nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500)
            Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500])
            Plot.savefig('test1  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
            Plot.close()
            print('data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1, 'nne3:', nne3, 'nne5:', nne5)
            nne1 = 0
            nne3 = 0
            nne5 = 0
            ti = 0
            for i in range(20):
                t = time.time()
                embedding_array = tsne.fit_transform(data[i])
                ti += time.time() - t
                nne1 += n_nearest_neighbor_error(1, embedding_array, label)
                nne3 += n_nearest_neighbor_error(3, embedding_array, label)
                nne5 += n_nearest_neighbor_error(5, embedding_array, label)
                Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label])
                Plot.savefig('test1  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(i) + '.png')
                Plot.close()
            print('data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:', nne1 / 20, 'nne3:',
                  nne3 / 20, 'nne5:', nne5 / 20)
            n_jobs *= 2
    elif mode == 2:
        ti = 0
        while perp < 25:
            tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
            ti = time.time()
            embedding_array = tsne.fit_transform(data_2500)
            ti = time.time() - ti
            nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500)
            nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500)
            nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500)
            Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500])
            Plot.savefig('test2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
            Plot.close()
            print('data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1,
                  'nne3:', nne3, 'nne5:', nne5)
            nne1 = 0
            nne3 = 0
            nne5 = 0
            ti = 0
            for i in range(20):
                t = time.time()
                embedding_array = tsne.fit_transform(data[i])
                ti += time.time() - t
                nne1 += n_nearest_neighbor_error(1, embedding_array, label)
                nne3 += n_nearest_neighbor_error(3, embedding_array, label)
                nne5 += n_nearest_neighbor_error(5, embedding_array, label)
                Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label])
                Plot.savefig('test2  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(
                    i) + '.png')
                Plot.close()
            print('data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:',
                  nne1 / 20, 'nne3:',
                  nne3 / 20, 'nne5:', nne5 / 20)
            perp += 5
    elif mode == 3:
        ti = 0
        while max_iter < 1001:
            tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
            ti = time.time()
            embedding_array = tsne.fit_transform(data_2500)
            ti = time.time() - ti
            nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500)
            nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500)
            nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500)
            Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500])
            Plot.savefig('test3  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
            Plot.close()
            print('data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1,
                  'nne3:', nne3, 'nne5:', nne5)
            nne1 = 0
            nne3 = 0
            nne5 = 0
            ti = 0
            for i in range(20):
                t = time.time()
                embedding_array = tsne.fit_transform(data[i])
                ti += time.time() - t
                nne1 += n_nearest_neighbor_error(1, embedding_array, label)
                nne3 += n_nearest_neighbor_error(3, embedding_array, label)
                nne5 += n_nearest_neighbor_error(5, embedding_array, label)
                Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label])
                Plot.savefig('test3  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(
                    i) + '.png')
                Plot.close()
            print('data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:',
                  nne1 / 20, 'nne3:',
                  nne3 / 20, 'nne5:', nne5 / 20)
            max_iter += 200
    elif mode == 4:
        ti = 0
        while n_jobs < 5:
            while max_iter < 1001:
                tsne = TSNE(n_jobs=n_jobs, perplexity=perp, n_iter=max_iter, angle=0.5)
                ti = time.time()
                embedding_array = tsne.fit_transform(data_2500)
                ti = time.time() - ti
                nne1 = n_nearest_neighbor_error(1, embedding_array, label_2500)
                nne3 = n_nearest_neighbor_error(3, embedding_array, label_2500)
                nne5 = n_nearest_neighbor_error(5, embedding_array, label_2500)
                Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label_2500])
                Plot.savefig(
                    'test4  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data2500.png')
                Plot.close()
                print('data2500 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti, 'nne1:', nne1,
                      'nne3:', nne3, 'nne5:', nne5)
                nne1 = 0
                nne3 = 0
                nne5 = 0
                ti = 0
                for i in range(20):
                    t = time.time()
                    embedding_array = tsne.fit_transform(data[i])
                    ti += time.time() - t
                    nne1 += n_nearest_neighbor_error(1, embedding_array, label)
                    nne3 += n_nearest_neighbor_error(3, embedding_array, label)
                    nne5 += n_nearest_neighbor_error(5, embedding_array, label)
                    Plot.scatter(embedding_array[:, 0], embedding_array[:, 1], 10, c=[color[x] for x in label])
                    Plot.savefig(
                        'test4  perp' + str(perp) + 'iter' + str(max_iter) + 'n_jobs' + str(n_jobs) + 'data' + str(
                            i) + '.png')
                    Plot.close()
                print('data0-19 perp:', perp, 'max_iteration:', max_iter, 'n_jobs:', n_jobs, 'time:', ti / 20, 'nne1:',
                      nne1 / 20, 'nne3:',
                      nne3 / 20, 'nne5:', nne5 / 20)
                max_iter += 200
            n_jobs *= 2
# test1 test for n_jobs
# test(mode=1, perp=8, max_iter=600, n_jobs=2)
# test2 test for perplexity
# test(mode=2, perp=5, max_iter=600, n_jobs=4)
# test3 test for max_iter
# test(mode=3, perp=10, max_iter=200, n_jobs=4)
# test4 test for speed
# test(mode=4, perp=8, max_iter=1000, n_jobs=4)

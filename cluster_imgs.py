from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import csv


def img_to_arr(img):
    img = img.resize((224, 224))
    return np.asarray(img)


def load_img(fpath):
    return Image.open(fpath)


def add_to_arr(arr1, arr2):
    return np.append(arr1, arr2)


def val_to_bucket(val, s, n):
    return min(val // s, n-1)


def rgb_to_bucket(r, g, b, size, num):
    return val_to_bucket(r, size, num), val_to_bucket(g, size, num), val_to_bucket(b, size, num)


def create_color_histogram(arr, bucket_size):
    # Create array to store pixel value histogram
    num_buckets = 256 // bucket_size
    bucketed_vals = np.zeros((num_buckets, num_buckets, num_buckets))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p = arr[i, j]

            # Get pixel bucket mapping
            x, y, z = rgb_to_bucket(p[0], p[1], p[2], bucket_size, num_buckets)

            bucketed_vals[x, y, z] += 1

    return bucketed_vals


def read_files(path, fname, bucket_size=1):
    file_names = []

    for subdir, dirs, files in os.walk(path):
        genres = ['disco']
        sd_arr = subdir.split('/')
        num_buckets = 256 // bucket_size

        genre_img_arr = None

        if len(sd_arr) > 1 and sd_arr[1] in genres:
            print(sd_arr[1])

            for file in tqdm(files):
                file_names.append(file.split('.')[0])

                if genre_img_arr is None:
                    img = load_img(os.path.join(subdir, file))
                    np_img = img_to_arr(img)

                    color_hist = create_color_histogram(np_img, bucket_size)

                    genre_img_arr = np.empty((1, num_buckets**3))
                    genre_img_arr[0] = color_hist.flatten()

                else:
                    img = load_img(os.path.join(subdir, file))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    np_img = img_to_arr(img)
                    color_hist = create_color_histogram(np_img, bucket_size)

                    color_hist = color_hist.flatten()
                    color_hist = color_hist.reshape((1, color_hist.shape[0]))

                    genre_img_arr = np.append(genre_img_arr, color_hist, axis=0)

                # print(os.path.join(subdir, file))

            with open('histograms/' + fname + '.npy', 'wb') as f:
                np.save(f, genre_img_arr)


def visualize_pca(data):

    scaled_data = preprocessing.StandardScaler().fit_transform(data)
    red = manifold.TSNE(n_components=2)

    data_reduced = red.fit_transform(data)

    clust_alg = cluster.AffinityPropagation()

    labels = clust_alg.fit_predict(data_reduced)

    unique_labels = np.unique(labels)

    label_dist = np.histogram(labels, bins=np.arange(len(unique_labels)))

    plt.figure()
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1])

    plt.show()
    plt.close()


def reduce_dims(data):
    # data = preprocessing.MinMaxScaler().fit_transform(data)
    # data = np.where(np.isnan(data), 0, data)
    # tSNE works better on this data than PCA (non-linear)
    data_reduced = manifold.TSNE(n_components=3).fit_transform(data)
    # data_reduced = decomposition.PCA(n_components=100).fit_transform(data)

    return data_reduced


def visualize_hist(data):
    fg = sns.clustermap(data, linewidths=0.001, metric='cosine').fig.suptitle('Clustering of tSNE embedded folk album art')
    plt.show()


def get_cube_mean(cube):
    mean = np.zeros(3)
    tot = 0

    for i in range(10):
        for j in range(10):
            for k in range(10):
                ct = cube[i, j, k]

                mean[0] += i*ct
                mean[1] += j*ct
                mean[2] += k*ct

                tot += ct

    # Some bug in here, if tot=0 just ignoring the image
    if tot > 0:
        mean /= tot

    return mean


def summarize_imgs(data, bucket_size):
    n = 256 // bucket_size
    means = np.zeros((data.shape[0], 3))

    for i, row in enumerate(data):
        cube = row.reshape((n, n, n))

        # process cube to get weighted RGB ave
        means[i] = get_cube_mean(cube)

    return means


def cluster_imgs(data, n_clusters=2):
    kmeans = cluster.KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)

    label_cts = {i: np.sum(np.where(labels == i, 1, 0)) for i in range(n_clusters)}
    sil_score = metrics.silhouette_score(data, labels)

    return labels, sil_score


def get_labels_by_cluster(clusters, labels, imgs):
    retain = []
    retain_imgs = np.empty((1, 224, 224, 3))
    for c, l, i in zip(clusters, labels, imgs):
        if c == 0:
            retain.append(l)

            if len(retain) == 1:
                retain_imgs[0] = i
            else:
                retain_imgs = np.append(retain_imgs, i.reshape(1, 224, 224, 3), axis=0)

    return retain, retain_imgs


# read_files('images_by_genre', 'disco_5', bucket_size=5)

data = np.load('featurized_data/folk.npy')
data = reduce_dims(data)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('tSNE embedding of Folk album art')
ax1.scatter(data[:, 0], data[:, 1], s=0.1)
ax1.set(xlabel='tSNE dimension 1', ylabel='tSNE dimension 2')

ax2.scatter(data[:, 1], data[:, 2], s=0.1)
ax2.set(xlabel='tSNE dimension 2', ylabel='tSNE dimension 3')
plt.show()
plt.close()

# data_summ = summarize_imgs(data, 10)

# visualize_hist(data)

# clusters, score = cluster_imgs(data, n_clusters=2)

# Evaluating clustering performance
# sil_scores = []
# for i in range(3, 100):
#     labels, score = cluster_imgs(data, n_clusters=i)
#     sil_scores.append(score)
#
#
# plt.figure()
# plt.plot(range(3, 100), sil_scores)
# plt.show()
# plt.close()

# labels = []
# with open('processed_data/folk_labels.csv') as f:
#     reader = csv.reader(f, delimiter=',')
#     for row in reader:
#         labels.append(''.join(row))
#
# img_data = np.load('processed_data/folk.npy')
# retained_labels, retained_imgs = get_labels_by_cluster(clusters, labels, img_data)
#
# with open('cleaned_data/' + 'folk' + '_labels.csv', 'w') as f:
#     csv.writer(f).writerows(retained_labels)
#
# np.save('cleaned_data/' + 'folk' + '.npy', retained_imgs)

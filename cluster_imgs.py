from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import manifold
from sklearn import cluster
import matplotlib.pyplot as plt
import seaborn as sns


def img_to_arr(img):
    img = img.resize((224, 224))
    return np.asarray(img)


def load_img(fpath):
    return Image.open(fpath)


def add_to_arr(arr1, arr2):
    return np.append(arr1, arr2)


def val_to_bucket(val):
    return min(val // 25, 9)


def rgb_to_bucket(r, g, b):
    return val_to_bucket(r), val_to_bucket(g), val_to_bucket(b)


def create_color_histogram(arr):
    # Create array to store pixel value histogram
    bucketed_vals = np.zeros((10, 10, 10))

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p = arr[i, j]

            # Get pixel bucket mapping
            x, y, z = rgb_to_bucket(p[0], p[1], p[2])

            bucketed_vals[x, y, z] += 1

    return bucketed_vals


def read_files(path):
    for subdir, dirs, files in os.walk(path):
        genres = ['disco']
        sd_arr = subdir.split('/')

        genre_img_arr = None

        if len(sd_arr) > 1 and sd_arr[1] in genres:
            print(sd_arr[1])

            for file in tqdm(files):
                if genre_img_arr is None:
                    img = load_img(os.path.join(subdir, file))
                    np_img = img_to_arr(img)

                    color_hist = create_color_histogram(np_img)

                    genre_img_arr = np.empty((1, 10*10*10))
                    genre_img_arr[0] = color_hist.flatten()

                else:
                    img = load_img(os.path.join(subdir, file))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    np_img = img_to_arr(img)
                    color_hist = create_color_histogram(np_img)

                    color_hist = color_hist.flatten()
                    color_hist = color_hist.reshape((1, color_hist.shape[0]))

                    genre_img_arr = np.append(genre_img_arr, color_hist, axis=0)

                # print(os.path.join(subdir, file))

            with open('histograms/' + sd_arr[1] + '.npy', 'wb') as f:
                np.save(f, genre_img_arr)


def visualize_pca(data):

    scaled_data = preprocessing.StandardScaler().fit_transform(data)
    red = manifold.TSNE(n_components=3)

    data_reduced = red.fit_transform(data)

    clust_alg = cluster.AffinityPropagation()

    labels = clust_alg.fit_predict(data_reduced)

    unique_labels = np.unique(labels)

    label_dist = np.histogram(labels, bins=np.arange(len(unique_labels)))

    plt.figure()
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1])

    plt.show()
    plt.close()


# read_files('images_by_genre')
data = np.load('histograms/disco.npy')
# visualize_pca(data)

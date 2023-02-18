from PIL import Image
import numpy as np
import os
from tqdm import tqdm


def img_to_arr(img):
    img = img.resize((224, 224))
    return np.asarray(img)


def load_img(fpath):
    return Image.open(fpath)


def add_to_arr(arr1, arr2):
    return np.append(arr1, arr2)


def read_files(path):
    for subdir, dirs, files in os.walk(path):
        genres = ['disco', 'electro', 'folk', 'rap', 'rock']
        sd_arr = subdir.split('/')

        genre_img_arr = None

        if len(sd_arr) > 1 and sd_arr[1] in genres:
            print(sd_arr[1])

            for file in tqdm(files):
                if genre_img_arr is None:
                    img = load_img(os.path.join(subdir, file))
                    np_img = img_to_arr(img)

                    genre_img_arr = np.empty((1, np_img.shape[0], np_img.shape[1], np_img.shape[2]))
                    genre_img_arr[0] = np_img

                else:
                    img = load_img(os.path.join(subdir, file))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    np_img = img_to_arr(img)

                    np_img = np_img.reshape((1, np_img.shape[0], np_img.shape[1], np_img.shape[2]))
                    genre_img_arr = np.append(genre_img_arr, np_img, axis=0)

                # print(os.path.join(subdir, file))

            with open('processed_data/' + sd_arr[1] + '.npy', 'wb') as f:
                np.save(f, genre_img_arr)


read_files('images_by_genre')

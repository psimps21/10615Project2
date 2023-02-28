import os
import csv
import numpy as np
import PIL
from process_imgs import load_img, img_to_arr
from tqdm import tqdm


def read_and_filter_files(path):
    for subdir, dirs, files in os.walk(path):
        genres = ['disco', 'electro', 'rap', 'rock']
        sd_arr = subdir.split('/')

        genre_img_arr = None

        if len(sd_arr) > 1 and sd_arr[1] in genres:
            labels = []
            with open('cleaned_data/' + sd_arr[1] + '_labels.csv') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    labels.append(''.join(row))

            for file in tqdm(files):
                fname = file.split('.')[0]

                if fname in labels:
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

                with open('cleaned_data/' + sd_arr[1] + '.npy', 'wb') as f:
                    np.save(f, genre_img_arr)


read_and_filter_files('images_by_genre')

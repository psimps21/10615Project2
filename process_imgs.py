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


img = load_img('keaty_tom_2.jpeg')
width, height = img.size

img = img.crop((0, 0, width, width))

img_np = img_to_arr(img)

np.save('keaty_tom.npy', img_np)


# read_files('images_by_genre')

# disco = np.load('processed_data/disco.npy')
# electro = np.load('processed_data/electro.npy')
# folk = np.load('processed_data/folk.npy')
# rap = np.load('processed_data/rap.npy')
# rock = np.load('processed_data/rock.npy')
#
# sizes = [disco.shape[0]//5, electro.shape[0]//5, folk.shape[0]//5, rap.shape[0]//5, rock.shape[0]//5]
#
# labels = np.repeat(0, len(disco[sizes[0]*4:]))
# labels = np.concatenate((labels, np.repeat(1, len(electro[sizes[1]*4:]))))
# labels = np.concatenate((labels, np.repeat(2, len(folk[sizes[2]*4:]))))
# labels = np.concatenate((labels, np.repeat(3, len(rap[sizes[3]*4:]))))
# labels = np.concatenate((labels, np.repeat(4, len(rock[sizes[4]*4:]))))
#
# np.save('batched_data/b5_labels.npy', labels)

# batch_5 = np.concatenate((
#     disco[sizes[0]*4:],
#     electro[sizes[1]*4:],
#     folk[sizes[2]*4:],
#     rap[sizes[3]*4:],
#     rock[sizes[4]*4:]
# ), axis=0)
#
# np.save('batched_data/b5.npy', batch_5)

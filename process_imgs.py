from PIL import Image
import numpy as np
import os
import csv
from tqdm import tqdm


def img_to_arr(img):
    img = img.resize((224, 224))
    return np.asarray(img)


def load_img(fpath):
    return Image.open(fpath)


def add_to_arr(arr1, arr2):
    return np.append(arr1, arr2)


def resize_all(arr, newsize):
    new_arr = np.empty((1, newsize, newsize, 3))
    flag = True

    for row in arr:
        img = Image.fromarray(row, 'RGB')
        img = img.resize((newsize, newsize))

        if flag:
            new_arr[0] = np.asarray(img)
            flag = False
        else:
            new_arr = np.append(new_arr, np.asarray(img).reshape((1, newsize, newsize, 3)), axis=0)

    return new_arr


def read_files(path):
    for subdir, dirs, files in os.walk(path):
        genres = ['disco', 'electro', 'folk', 'rap', 'rock']
        sd_arr = subdir.split('/')

        genre_img_arr = None

        if len(sd_arr) > 1 and sd_arr[1] in genres:
            print(sd_arr[1])
            labels = []

            for file in tqdm(files):
                labels.append(file.split('.')[0])

            #     if genre_img_arr is None:
            #         img = load_img(os.path.join(subdir, file))
            #         np_img = img_to_arr(img)
            #
            #         genre_img_arr = np.empty((1, np_img.shape[0], np_img.shape[1], np_img.shape[2]))
            #         genre_img_arr[0] = np_img
            #
            #     else:
            #         img = load_img(os.path.join(subdir, file))
            #         if img.mode != 'RGB':
            #             img = img.convert('RGB')
            #
            #         np_img = img_to_arr(img)
            #
            #         np_img = np_img.reshape((1, np_img.shape[0], np_img.shape[1], np_img.shape[2]))
            #         genre_img_arr = np.append(genre_img_arr, np_img, axis=0)
            #
            #     # print(os.path.join(subdir, file))
            #
            # with open('processed_data/' + sd_arr[1] + '.npy', 'wb') as f:
            #     np.save(f, genre_img_arr)
            with open('processed_data/' + sd_arr[1] + '_labels.csv', 'w') as f:
                csv.writer(f).writerows(labels)


# disco = np.load('cleaned_data/disco.npy')
# electro = np.load('cleaned_data/electro.npy')
# folk = np.load('cleaned_data/folk.npy')
# rap = np.load('cleaned_data/rap.npy')
# rock = np.load('cleaned_data/rock.npy')
#
# disco_100 = resize_all(disco, 100)
# electro_100 = resize_all(electro, 100)
# folk_100 = resize_all(folk, 100)
# rap_100 = resize_all(rap, 100)
# rock_100 = resize_all(rock, 100)
#
# np.save('cleaned_data/disco_100.npy', disco_100)
# np.save('cleaned_data/electro_100.npy', electro_100)
# np.save('cleaned_data/folk_100.npy', folk_100)
# np.save('cleaned_data/rap_100.npy', rap_100)
# np.save('cleaned_data/rock_100.npy', rock_100)

img = load_img('parker_pic.jpg')
width, height = img.size

img = img.crop((0, 0, width, width))
img = img.resize((224, 224))

img.save('parker_pic_224.jpg')

img_np = img_to_arr(img)

np.save('parker_pic.npy', img_np)


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

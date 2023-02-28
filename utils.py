import os, shutil, random
from pathlib import Path

import numpy as np
import zipfile
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from fastdownload import FastDownload
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from google.cloud import storage

cifar_labels = "airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck".split(",")
alphabet_labels = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def untar_data(url, force_download=False, base='./datasets'):
    d = FastDownload(base=base)
    return d.get(url, force=force_download, extract_key='data')


def one_batch(dl):
    return next(iter(dl))
        

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_gcloud_data():
    storage_client = storage.Client('fiery-nimbus-379022')
    bucket = storage_client.bucket('tck-ml-art-proj-2')
    blobs = storage_client.list_blobs('tck-ml-art-proj-2')

    for blob in blobs:
        blob.download_to_filename('../' + blob.name)


def create_base_diffusion_demo_tensors(base_fpath):
    raw_data = np.load(base_fpath)
    data = np.empty((5, raw_data.shape[0], raw_data.shape[0], 3))

    for i in range(5):
        data[i] = raw_data

    data = np.swapaxes(data, 1, 3).astype('f4')
    labels = np.asarray([0, 1, 2, 3, 4]).astype(int)

    data_t = torch.Tensor(data)
    label_t = torch.Tensor(labels)

    return data_t, label_t


def get_data(args, mode=0):
    train_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size + int(.25*args.img_size)),  # args.img_size + 1/4 *args.img_size
        T.RandomResizedCrop(args.img_size, scale=(0.8, 1.0)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_transforms = torchvision.transforms.Compose([
        T.Resize(args.img_size),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    """
        Changing these lines to read our data correctly
    
    train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.train_folder), transform=train_transforms)
    val_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, args.val_folder), transform=val_transforms)
    
    if args.slice_size>1:
        train_dataset = torch.utils.data.Subset(train_dataset, indices=range(0, len(train_dataset), args.slice_size))
        val_dataset = torch.utils.data.Subset(val_dataset, indices=range(0, len(val_dataset), args.slice_size))
    """
    genres = ['rap', 'rock', 'folk', 'disco', 'electro']
    labels = None

    data = None
    for i, g in enumerate(genres):
        if mode == 0 and i < 2:
            if data is None:
                data = np.load('../cleaned_data/' + g + '.npy')
                labels = np.zeros(len(data))
            else:
                g_data = np.load('../cleaned_data/' + g + '.npy')
                data = np.append(data, g_data, axis=0)
                labels = np.append(labels, np.repeat(i, len(g_data)))
        elif mode == 1 and i > 1 and i < 4:
            if data is None:
                data = np.load('../cleaned_data/' + g + '.npy')
                labels = np.zeros(len(data))
            else:
                g_data = np.load('../cleaned_data/' + g + '.npy')
                data = np.append(data, g_data, axis=0)
                labels = np.append(labels, np.repeat(i, len(g_data)))
        elif mode == 2 and i == 4:
            if data is None:
                data = np.load('../cleaned_data/' + g + '.npy')
                labels = np.zeros(len(data))
            else:
                g_data = np.load('../cleaned_data/' + g + '.npy')
                data = np.append(data, g_data, axis=0)
                labels = np.append(labels, np.repeat(i, len(g_data)))

    data = np.swapaxes(data, 1, 3)

    data = data.astype('f4')
    labels = labels.astype(int)

    val_set_size = int(0.9*len(data))
    val_inds = np.random.choice(np.arange(len(data)), size=val_set_size, replace=False)

    train_dataset = []
    val_dataset = []

    for i, img, label in zip(range(len(data)), data, labels):
        if i in val_inds:
            val_dataset.append([img, label])
        else:
            train_dataset.append([img, label])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataset = DataLoader(val_dataset, batch_size=2*args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_dataloader, val_dataset


def mk_folders(run_name):
    os.makedirs("TESTING/Diffusion-Models-pytorch-main/models", exist_ok=True)
    os.makedirs("TESTING/Diffusion-Models-pytorch-main/results", exist_ok=True)
    os.makedirs(os.path.join("TESTING/Diffusion-Models-pytorch-main/models", run_name), exist_ok=True)
    os.makedirs(os.path.join("TESTING/Diffusion-Models-pytorch-main/results", run_name), exist_ok=True)


if __name__ == '__main__':
    get_gcloud_data()

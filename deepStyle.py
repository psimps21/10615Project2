import torch
import numpy as np
import pickle
from PIL import Image as im
import torchvision
from vgg_model import AgnosticModel
import numbers
import math
import torch.nn.functional as F
from tqdm import tqdm

"""
From this tutorial: https://github.com/gordicaleksa/pytorch-deepdream/blob/master/deepdream.py
"""


class CascadeGaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(3, 1, 1, 1)

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3


def deep_style(img, model, label=None, lr=0.001):
    # Convert image to pytorch tensor (with gradient)
    new_img = np.zeros((1, 224, 224, 3))
    new_img[0] = img / 255.0

    input = torch.Tensor(new_img)
    input = torch.swapaxes(input, 1, 3)
    input.requires_grad = True

    # Convert label to appropriate type
    label = torch.Tensor([label])
    label = label.type(torch.long)

    # Define loss fun
    loss = torch.nn.CrossEntropyLoss()

    config = {
        'num_gradient_ascent_iterations': 10,
        'smoothing_coefficient': 0.5,
        'lr': 0.09
    }

    # Pass image through model
    out = model(input)

    # Compute loss and gradient
    l = loss(out, label)
    l.backward()

    # Apply gradient to image
    grad = input.grad.data

    sigma = 0.1
    smooth_grad = CascadeGaussianSmoothing(kernel_size=9, sigma=sigma)(grad)  # "magic number" 9 just works well

    g_std = torch.std(smooth_grad)
    g_mean = torch.mean(smooth_grad)
    smooth_grad = grad - g_mean
    smooth_grad = smooth_grad / g_std

    input.data += lr * smooth_grad

    # Clear gradient
    input.grad.data.zero_()

    return input.detach().numpy()


with open('resnet_trained.obj', 'rb') as f:
    model = pickle.load(f)
img_arr = np.load('parker_pic.npy')

new_img_arr = np.copy(img_arr)

for i in tqdm(range(1000)):
    new_img_arr = deep_style(new_img_arr, model, label=0)
    new_img_arr = np.swapaxes(new_img_arr, 1, 3)[0] * 255.0

new_img_arr = np.minimum(new_img_arr, 255.0)
new_img_arr = np.maximum(new_img_arr, 0.0)
new_img_arr = np.floor(new_img_arr)
new_img_arr = new_img_arr.astype(np.uint8)

image = im.fromarray(new_img_arr)
image.show()

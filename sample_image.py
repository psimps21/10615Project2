import torch
import sys
from ddpm_conditional import *
from utils import save_images, create_base_diffusion_demo_tensors


def retrieve_model(ckpt_path):
    model = Diffusion(img_size=64, num_classes=5)
    model.load(ckpt_path)

    return model


def gen_image(model, run_label='test'):
    labels = torch.Tensor([1, 2, 3, 4, 5])
    samples = model.sample(use_ema=False, labels=labels, n=len(labels))
    save_images(samples, 'demo_samples/' + run_label + '.jpg')


def gen_image_from_base(model, image_fpath, run_label='test'):
    x, labels = create_base_diffusion_demo_tensors(image_fpath)
    samples = model.diffuse_from_base(x, labels, t_ratio=0.5)
    save_images(samples, 'demo_2_samples/' + run_label + '.jpg')


if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    run_label = sys.argv[2]
    base = sys.argv[3]
    from_base = False

    if base == 'y':
        base_path = sys.argv[4]
        from_base = True

    model = retrieve_model(ckpt_path)

    if from_base:
        gen_image_from_base(model, base_path, run_label)

    else:
        gen_image(model, run_label)

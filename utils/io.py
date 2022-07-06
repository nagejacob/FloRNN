import cv2
import imageio
import numpy as np
import os
import torch

def list_dir(dir, postfix=None, full_path=False):
    if full_path:
        if postfix is None:
            names = sorted([name for name in os.listdir(dir) if not name.startswith('.')])
            return sorted([os.path.join(dir, name) for name in names])
        else:
            names = sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])
            return sorted([os.path.join(dir, name) for name in names])
    else:
        if postfix is None:
            return sorted([name for name in os.listdir(dir) if not name.startswith('.')])
        else:
            return sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])

def open_images_uint8(image_files):
    image_list = []
    for image_file in image_files:
        image = imageio.imread(image_file).astype(np.uint8)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image_list.append(image)
    seq = np.stack(image_list, axis=0)
    return seq

def log(log_file, str, also_print=True):
    with open(log_file, 'a+') as F:
        F.write(str)
    if also_print:
        print(str, end='')

# return pytorch image in shape 1x3xHxW
def image2tensor(image_file):
    image = imageio.imread(image_file).astype(np.float32) / np.float32(255.0)
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.asarray(image, dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    return image

# save numpy image in shape 3xHxW
def np2image(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    imageio.imwrite(image_file, image)

def np2image_bgr(image, image_file):
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0., 1.)
    image = image * 255.
    image = image.astype(np.uint8)
    cv2.imwrite(image_file, image)

# save tensor image in shape 1x3xHxW
def tensor2image(image, image_file):
    image = image.detach().cpu().squeeze(0).numpy()
    np2image(image, image_file)


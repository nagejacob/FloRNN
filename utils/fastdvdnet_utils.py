import cv2
import glob
import numpy as np
import os
from random import choices
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import torch

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif') # Supported image types

def fastdvdnet_batch_psnr(img, imclean, data_range=1.):
    r"""
    Computes the PSNR along the batch dimension (not pixel-wise)

    Args:
        img: a `torch.Tensor` containing the restored image
        imclean: a `torch.Tensor` containing the reference image
        data_range: The data range of the input image (distance between
            minimum and maximum possible values). By default, this is estimated
            from the image data-type.
    """
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :],
                       data_range=data_range)
    return psnr/img_cpu.shape[0]

def get_imagenames(seq_dir, pattern=None):
    """ Get ordered list of filenames
    """
    files = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))

    # filter filenames
    if not pattern is None:
        ffiltered = [f for f in files if pattern in os.path.split(f)[-1]]
        files = ffiltered
        del ffiltered

    # sort filenames alphabetically
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files

def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=85):
    r""" Opens a sequence of images and expands it to even sizes if necesary
    Args:
        fpath: string, path to image sequence
        gray_mode: boolean, True indicating if images is to be open are in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
        max_num_fr: maximum number of frames to load
    Returns:
        seq: array of dims [num_frames, C, H, W], C=1 grayscale or C=3 RGB, H and W are even.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    # Get ordered list of filenames
    files = get_imagenames(seq_dir)

    seq_list = []
    # print("\tOpen sequence in folder: ", seq_dir)
    for fpath in files[0:max_num_fr]:

        img, expanded_h, expanded_w = open_image(fpath,\
                                                   gray_mode=gray_mode,\
                                                   expand_if_needed=expand_if_needed,\
                                                   expand_axis0=False)
        seq_list.append(img)
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w

def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    r""" Opens an image and expands it if necesary
    Args:
        fpath: string, path of image file
        gray_mode: boolean, True indicating if image is to be open
            in grayscale mode
        expand_if_needed: if True, the spatial dimensions will be expanded if
            size is odd
        expand_axis0: if True, output will have a fourth dimension
    Returns:
        img: image of dims NxCxHxW, N=1, C=1 grayscale or C=3 RGB, H and W are even.
            if expand_axis0=False, the output will have a shape CxHxW.
            The image gets normalized gets normalized to the range [0, 1].
        expanded_h: True if original dim H was odd and image got expanded in this dimension.
        expanded_w: True if original dim W was odd and image got expanded in this dimension.
    """
    if not gray_mode:
        # Open image as a CxHxW torch.Tensor
        img = cv2.imread(fpath)
        # from HxWxC to CxHxW, RGB image
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        # from HxWxC to  CxHxW grayscale image (C=1)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)

    if expand_axis0:
        img = np.expand_dims(img, 0)

    # Handle odd sizes
    expanded_h = False
    expanded_w = False
    sh_im = img.shape
    if expand_if_needed:
        if sh_im[-2]%2 == 1:
            expanded_h = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
            else:
                img = np.concatenate((img, \
                    img[:, -1, :][:, np.newaxis, :]), axis=1)


        if sh_im[-1]%2 == 1:
            expanded_w = True
            if expand_axis0:
                img = np.concatenate((img, \
                    img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
            else:
                img = np.concatenate((img, \
                    img[:, :, -1][:, :, np.newaxis]), axis=2)

    if normalize_data:
        img = normalize(img)
    return img, expanded_h, expanded_w

def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data/255.)

def normalize_augment(img_train):
    '''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
        [N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
        patch as a ground truth.
    '''
    def transform(sample):
        # define transformations
        do_nothing = lambda x: x
        do_nothing.__name__ = 'do_nothing'
        flipud = lambda x: torch.flip(x, dims=[2])
        flipud.__name__ = 'flipup'
        rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
        rot90.__name__ = 'rot90'
        rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        rot90_flipud.__name__ = 'rot90_flipud'
        rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
        rot180.__name__ = 'rot180'
        rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
        rot180_flipud.__name__ = 'rot180_flipud'
        rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        rot270.__name__ = 'rot270'
        rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        rot270_flipud.__name__ = 'rot270_flipud'
        add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
                                 std=(5/255.)).expand_as(x).to(x.device)
        add_csnt.__name__ = 'add_csnt'

        # define transformations and their frequency, then pick one.
        aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
                    rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
        w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12] # one fourth chances to do_nothing
        transf = choices(aug_list, w_aug)

        # transform all images in array
        return transf[0](sample)

    N, T, C, H, W = img_train.shape
    # convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
    img_train = img_train.type(torch.float32).view(N, -1, H, W) / 255.

    # augment
    img_train = transform(img_train)

    # view back
    img_train = img_train.view(N, T, C, H, W)

    return img_train

def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel models to a normal one by removing the "module."
    wrapper in the module dictionary


    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel
        new_state_dict[name] = v

    return new_state_dict

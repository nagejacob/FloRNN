import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# img: T, C, H, W, imclean: T, C, H, W
def batch_ssim(img, imclean, data_range):

    img = img.data.cpu().numpy().astype(np.float32)
    img = np.transpose(img, (0, 2, 3, 1))
    img_clean = imclean.data.cpu().numpy().astype(np.float32)
    img_clean = np.transpose(img_clean, (0, 2, 3, 1))

    ssim = 0
    for i in range(img.shape[0]):
        origin_i = img_clean[i, :, :, :]
        denoised_i = img[i, :, :, :]
        ssim += compare_ssim(origin_i.astype(float), denoised_i.astype(float), multichannel=True, win_size=11, K1=0.01,
                             K2=0.03, sigma=1.5, gaussian_weights=True, data_range=1)
    return ssim/img.shape[0]
import h5py
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
from fastmri.data.subsample import EquispacedMaskFunc
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
from PIL import ImageOps
import torch

file_name = 'file_brain_AXFLAIR_200_6002435.h5'
hf = h5py.File(file_name)
dirname = 'unsupervised_dataset/train/'

# 1. get fully sampled image
volume_kspace = hf['kspace'][()]
slice_kspace = volume_kspace[5]
slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
slice_image = fastmri.ifft2c(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
slice_image_abs = fastmri.complex_abs(slice_image)   # Compute absolute value to get a real image
slice_image_rss = fastmri.rss(slice_image_abs, dim=0)
numpy_img = np.abs(slice_image_rss.numpy())
print('fs', numpy_img.shape)
plt.imsave('fully_sampled.png', numpy_img, cmap='gray')
# plt.show()
# breakpoint()

# 2. get original 4x equispaced (base) undersampled image
mask_func = EquispacedMaskFunc(center_fractions=[0.04], accelerations=[4])  # Create the mask function object
masked_kspace, mask = T.apply_mask(slice_kspace2, mask_func)   # Apply the mask to k-space
sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
numpy_img = np.abs(sampled_image_rss.numpy())
# plt.imshow(numpy_img, cmap='gray')
# plt.show()
plt.imsave(dirname + '0_equispaced.png', numpy_img, cmap='gray')
print('equi', numpy_img.shape)
# 3. get sobel of base undersampled image
# breakpoint()
sbl = sobel(numpy_img)
plt.imsave(dirname + '0_sobel.png', sbl, cmap='gray')
print('sobel', numpy_img.shape)
# breakpoint()
# 4. get gaussians of base undersampled image
sigmas = np.arange(0.5, 3.5, 0.1)
for i in range(len(sigmas)):
    gaussian = gaussian_filter(numpy_img, sigma=sigmas[i])
    plt.imsave(dirname + str(i) + '_gaussian_' + str(round(sigmas[i], 1)).replace('.', '_') + '.png', gaussian, cmap='gray')
    # print('gaussian', gaussian.shape)
    # breakpoint()
# 5. get randomly masked images from base undersampled images
center_fractions = np.arange(0.2, 1.2, 0.035)
for i in range(len(center_fractions)):
    # print(center_fractions[i])
    mask_func = RandomMaskFunc(center_fractions=[center_fractions[i]], accelerations=[2])  # Create the mask function object
    second_mask, mask = T.apply_mask(masked_kspace, mask_func)
    sampled_image = fastmri.ifft2c(second_mask)
    sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
    sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
    random = np.abs(sampled_image_rss.numpy())
    plt.imsave(dirname + str(i) + '_random_' + str(round(center_fractions[i], 1)).replace('.', '_') + '.png', random, cmap='gray')
    if center_fractions[i] == 0.9000000000000001:
        print('here!')
        # kspace_abs = fastmri.complex_abs(second_mask)
        kspace_rss = fastmri.rss(second_mask, dim=0)
        kspace_np = kspace_rss.numpy()
        white_img = np.ones((16, 640, 320, 2))
        white_img = np.zeros([640,320,3],dtype=np.uint8)
        white_img.fill(255) # or img[:] = 255
        white_mask, mask = T.apply_mask(torch.from_numpy(white_img), mask_func)
        print(white_mask.shape)
        np_white = white_mask.detach().numpy()
        norm = (np_white - np.min(np_white)) / (np.max(np_white) - np.min(np_white))
        # inverted = ImageOps.invert(white_img[0,:,:,0])
        plt.imsave('mask.png', norm, cmap='gray')
        # np_kspace = kspace.numpy()
        plt.imsave('kspace2.png', kspace_np[:,:,0], cmap='gray')
    # rsbl = sobel(random)
    # plt.imsave(dirname + str(i) + '_rsobel_' + str(round(center_fractions[i], 1)).replace('.', '_') + '.png', rsbl, cmap='gray')
# DEBUG: make the same number of random masks as gaussians

# mask_func = RandomMaskFunc(center_fractions=[0.4], accelerations=[2])  # Create the mask function object
# second_mask, mask = T.apply_mask(masked_kspace, mask_func)
# sampled_image = fastmri.ifft2c(second_mask)   
# sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
# sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
# plt.imshow(np.abs(sampled_image_rss.numpy()), cmap='gray')
# plt.show()

# # gaussian filter
# print('here')
# # gaussian_filter(slice_kspace2, sigma=10)
# sampled_image = fastmri.ifft2c(masked_kspace)           # Apply Inverse Fourier Transform to get the complex image
# sampled_image_abs = fastmri.complex_abs(sampled_image)   # Compute absolute value to get a real image
# sampled_image_rss = fastmri.rss(sampled_image_abs, dim=0)
# result = gaussian_filter(np.abs(sampled_image_rss.numpy()), sigma=3.4)
# # result = sobel(np.abs(sampled_image_rss.numpy()))
# plt.imshow(result, cmap='gray')
# plt.show()

# IDEA: MAYBE COMPARE SOBELS INSTEAD OF THE IMAGES THEMSELVES???
# want to learn function that estimates mean/median/mode of undersampled images
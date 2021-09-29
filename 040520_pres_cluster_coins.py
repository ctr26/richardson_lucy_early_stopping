#  %%
#!/usr/bin/env python
# coding: utf-8


from scipy.sparse.linalg.interface import LinearOperator
from numpy.random import binomial
from sklearn.metrics import mean_squared_error,mean_absolute_error
from skimage.metrics import structural_similarity as ssim

import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
from IPython.display import Video
from tqdm import tqdm
import multiprocessing
import os
import skimage
from enum import Enum, auto

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error


# from sklearn import neural_network, metrics, gaussian_process, preprocessing, svm, neighbors
# from sklearn import pipeline, model_selection

# from keras import metrics
# from keras import backend as K
from scipy.stats import pearsonr
from sklearn import svm, linear_model
# import microscPSF.microscPSF as msPSF
from skimage.metrics import structural_similarity as ssim

from scipy import matrix
from scipy.sparse import coo_matrix
import time
from scipy import linalg
from skimage import color, data, restoration, exposure
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.signal import convolve2d as conv2
# import matlab.engine

# from sklearn.preprocessing import Imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# from sklearn.experimental import enable_iterative_imputer
from scipy import signal
from sklearn.impute import SimpleImputer


class psf_switch_enum(Enum):
    STATIC, VAR_PSF, VAR_ILL = auto(), auto(), auto()


SAVE_IMAGES = 0

# # Image formation

# Define constants: psf height width and image rescaling factor


# Define constants: psf height width and image rescaling factor
psf_w, psf_h, scale = 64, 64, 4
psf_window_w, psf_window_h = round(psf_w/scale), round(psf_h/scale)
sigma = 1


# Define approximate PSF function


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def psf_guass(w=psf_w, h=psf_h, sigma=3):
    # blank_psf = np.zeros((w,h))
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    psf = gaussian(xx, 0, sigma) * gaussian(yy, 0, sigma)
    return psf/psf.sum()  # Normalise PSF "energy"


static_psf = psf_guass(w=round(psf_window_h),
                       h=round(psf_window_w), sigma=2 / scale)
plt.imshow(static_psf)

# # Deconvolution


astro = rescale(color.rgb2gray(data.astronaut()), 1.0 / scale)
astro_blur = conv2(astro, static_psf, 'same')  # Blur image
astro_corrupt = astro_noisy = astro_blur + \
    (np.random.poisson(lam=25, size=astro_blur.shape) - 10) / \
    255.  # Add Noise to Image
deconvolved_RL = restoration.richardson_lucy(
    astro_blur, static_psf, iterations=10)  # RL deconvolution

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 7))
ax[0].imshow(astro)
ax[0].set_title('Truth')
ax[1].imshow(astro_blur)
ax[1].set_title('Blurred')
ax[2].imshow(astro_noisy)
ax[2].set_title('Blurred and noised')
ax[3].imshow(deconvolved_RL)
ax[3].set_title('Deconvolved')



# In[12]:


N_v = np.ma.size(astro)
N_p = np.ma.size(astro_blur)
measurement_matrix = matrix(np.zeros((N_p, N_v)))


zero_image = np.zeros_like(astro)
psf_window_volume = np.full((N_v, psf_window_w, psf_window_h), np.NaN)

x_astro, y_astro = astro_blur.shape
xx_astro, yy_astro = np.meshgrid(np.linspace(-1, 1, x_astro),
                                 np.linspace(-1, 1, y_astro))



# In[14]:

# Define a function that scales the PSF as a function of radial distance



def psf_vary(psf_window_h, psf_window_w, radius, scale):
    return psf_guass(w=round(psf_window_h), h=round(psf_window_w), sigma=(1 / scale)*abs((-0.4*radius))+0.1)


# Make the PSF vary across the image (as a function of radius)

# In[16]:


r_map = np.sqrt(xx_astro**2 + yy_astro**2)
radius_samples = np.linspace(-1, 1, 5)
fig, ax = plt.subplots(nrows=1, ncols=len(radius_samples), figsize=(16, 7))

for i, radius in enumerate(np.linspace(-1, 1, 5)):
    psf_current = psf_vary(psf_window_h, psf_window_w, radius, scale)
    ax[i].imshow(psf_current)
    ax[i].set_title("Radius: " + str(radius))
plt.show()




psf_switch = psf_switch_enum.STATIC
if(psf_switch == psf_switch_enum.STATIC):
    filename = "H_staticpsf"
if(psf_switch == psf_switch_enum.VAR_PSF):
    filename = "H_varpsf"


# Loop over each row of H and insert a flattened PSF that is the same shape as the input image

# In[18]:


for i in tqdm(np.arange(N_v)):
    # Get the xy coordinates of the ith pixel in the original image
    coords = np.unravel_index(i, np.array(astro.shape))
    r_dist = r_map[coords]                              # Convert to radius
    # Select mode for generating H, i.e. static/varying psf etc.
    if(psf_switch == psf_switch_enum.STATIC):
        psf_current = static_psf
    if(psf_switch == psf_switch_enum.VAR_PSF):
        psf_current = psf_vary(psf_window_h, psf_window_w, r_dist, scale)


    psf_window_volume[i, :, :] = psf_current
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i, astro.shape)] = 1
    # Convolve PSF with a image with a single 1 at coord
    delta_PSF = scipy.ndimage.convolve(delta_image, psf_current)

    measurement_matrix[i, :] = delta_PSF.flatten()
    if(SAVE_IMAGES):
        plt.imshow(psf_current)
        plt.imsave(
            f'./output/psfs_new/{str(i).zfill(6)}.png', psf_window_volume[i, :, :])



plt.figure(figsize=(18, 7))
plt.imshow(exposure.equalize_hist(measurement_matrix), cmap="gray_r")





H = scipy.sparse.linalg.aslinearoperator(measurement_matrix)
g_blurred = H.dot(astro.reshape(-1, 1))
f = np.matrix(g_blurred)


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 7))

ax[0].imshow(g_blurred.reshape(astro.shape))
ax[0].set_title("Blurred")
ax[1].imshow(astro)
ax[1].set_title("Original")
# ax[2].imshow(g.reshape(astro.shape));ax[2].set_title("RL")
plt.show()


# In[131]:


A = H
b = f
x0 = None
Rtol = 1e-6
NE_Rtol = 1e-6
max_iter = 100
sigmaSq = 0.0
beta = 0.0
integer_signal = np.rint(b*2**16).astype(int)
coin_flip_scale = binomial([2**16]*len(b), 0.5)/2**16


# In[132]:


b.shape


# In[143]:

from sklearn import preprocessing
T = np.matrix(np.multiply(coin_flip_scale, np.array(b).T).T)
V = b-T
gt = astro.reshape(V.shape)



plt.imshow(V.reshape(astro.shape))
plt.show()

plt.imshow(T.reshape(astro.shape))
plt.show()

# b = T

# The A operator represents a large, sparse matrix that has dimensions [ nrays x nvoxels ]

nrays = A.shape[0]
nvoxels = A.shape[1]

# Pre-compute some values for use in stopping criteria below
b_norm = np.linalg.norm(b)
trAb = A.rmatvec(b)
trAb_norm = np.linalg.norm(trAb)
trAb
# Start the optimization from the initial volume of a focal stack.
if x0 != None:
    x = x0
else:
    x = trAb

Rnrm_v = np.zeros(max_iter+1)
Xnrm= np.zeros(max_iter+1)
NE_Rnrm= np.zeros(max_iter+1)
gt_error_l1= np.zeros(max_iter+1)
gt_error_l2 = np.zeros(max_iter+1)
cross_correlation = np.zeros(max_iter+1)
gt_error_ssim = np.zeros(max_iter+1)
log_liklihood = np.zeros(max_iter+1)
log_liklihood_v = np.zeros(max_iter+1)

eps = np.spacing(1)
tau = np.sqrt(eps)
sigsq = tau
minx = x.min()

# If initial guess has negative values, compensate
if minx < 0:
    x = x - min(0, minx) + sigsq

normalization = A.rmatvec(np.ones(nrays)) + 1
print(normalization.min(), normalization.max())
c = A.matvec(x) + np.matrix(beta*np.ones(nrays)).T + \
    np.matrix(sigmaSq*np.ones(nrays)).T
b = b + np.matrix(sigmaSq*np.ones(nrays)).T


# i =0
SAVEFIG = 1
for i in range(max_iter):
    tic = time.time()
    x_prev = x

    # STEP 1: RL Update step
    # b / c
    # (c+1e-12).min()
    v = A.rmatvec(b / c)
    x = (np.multiply(x_prev, v)) / matrix(normalization).T
    Ax = A.matvec(x)
    # b-Ax
    residual = b - Ax
    residual_V = v - Ax
    c = A.matvec(x) + np.matrix(beta*np.ones(nrays)).T + \
        np.matrix(sigmaSq*np.ones(nrays)).T
    # c = Ax + beta*np.ones(nrays) + sigmaSq*np.ones(nrays);

    # STEP 2: Compute residuals and check stopping criteria
    Rnrm[i] = np.linalg.norm(residual) / b_norm
    Xnrm[i] = np.linalg.norm(x - x_prev) / nvoxels

    Rnrm_v[i] = np.linalg.norm(residual_V) / b_norm
    # NE_Rnrm[i] = np.linalg.norm(trAb - A.rmatvec(Ax)) / trAb_norm # disabled for now to save on extra rmatvec

    toc = time.time()
    print('\t--> [ RL Iteration %d   (%0.2f seconds) ] ' % (i, toc-tic))
    print('\t      Residual Norm: %0.10g               (tol = %0.12e)  ' %
          (Rnrm[i], Rtol))
    print('\t      Residual V Norm: %0.10g               (tol = %0.12e)  ' %
          (Rnrm_v[i], Rtol))
    # print '\t         Error Norm: %0.4g               (tol = %0.2e)  ' % (NE_Rnrm[i], NE_Rtol)
    print('\t        Update Norm: %0.4g                              ' %
          (Xnrm[i]))

    # stop because residual satisfies ||b-A*x|| / ||b||<= Rtol
    if Rnrm[i] <= Rtol:
        break

    # stop because normal equations residual satisfies ||A'*b-A'*A*x|| / ||A'b||<= NE_Rtol
    # if NE_Rnrm[i] <= NE_Rtol:
    #    break
    gt_error_l2[i] = mean_squared_error(x,gt)
    gt_error_l1[i] = mean_absolute_error(x,gt)
    
    x_flat= np.array(x.copy()).flatten()
    gt_flat = np.array(gt.copy()).flatten()

    gt_error_ssim[i] = ssim(x_flat,gt_flat)

    x_flat_scaled = preprocessing.scale(x_flat)
    gt_flat_scaled = preprocessing.scale(gt_flat)

    # a_flat_scaled = (x_flat - np.mean(x_flat)) / (np.std(x_flat) * len(x_flat))
    # b_flat_scaled = (gt_flat - np.mean(gt_flat)) / (np.std(gt_flat))
    # cross_correlation[i] = np.sum(np.correlate(a_flat_scaled, b_flat_scaled, 'full'))
    cross_correlation[i] = np.sum(np.correlate(x_flat_scaled,gt_flat_scaled, 'full'))
    # a = np.dot(abs(x_flat_scaled),abs(gt_flat_scaled),'full')
    log_liklihood[i] = np.sum(np.multiply(np.log(Ax),(b))-Ax-np.log(scipy.special.factorial(b)))
    log_liklihood_v[i] = np.sum(np.multiply(np.log(Ax),(v))-Ax-np.log(scipy.special.factorial(v)))

plt.plot(Rnrm_v[2:-1])
plt.title("V residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()


plt.plot(Rnrm[2:-1])
plt.title("Residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()

plt.plot(gt_error_l2[2:-1])
plt.title("GT v x ~ L1")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(gt_error_l1[2:-1])
plt.title("GT v x ~ L2")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(cross_correlation[2:-1])
plt.title("GT v x ~ cross_correlation")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()



plt.plot(gt_error_ssim[2:-1])
plt.title("GT v x ~ SSIM")
plt.xlabel("Iterations")
plt.ylabel("SSIM")
plt.show()

plt.plot(log_liklihood[2:-1])
plt.title("GT v x ~ log_liklihood")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()

plt.plot(log_liklihood_v[2:-1])
plt.title("x vs validation ~ log_liklihood_v")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()




# %%

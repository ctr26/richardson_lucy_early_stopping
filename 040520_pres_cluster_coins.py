#  %%
#!/usr/bin/env python
# coding: utf-8

from sklearn import preprocessing
from scipy.sparse.linalg.interface import LinearOperator
from numpy.random import binomial
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
from IPython.display import Video
from tqdm import tqdm
# from tqdm.notebook import tqdm

import multiprocessing
import os
import skimage
from enum import Enum, auto

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import mean_squared_error


# from sklearn import neural_network, metrics, gaussian_process, preprocessing,
#  svm, neighbors
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


signal_strength = 1
coin_flip_bias = 0.5
savefig = 1
save_images = 0
image_scale = 4
psf_scale = 0.75
psf_size = 64
psf_type = "static"
max_iter = 100
thinning_type = "poisson"
out_dir = "out"
# Define constants: psf height width and image rescaling factor
# %%

# if __name__ == "__main__":
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--signal_strength", default=signal_strength, type=float)
parser.add_argument("--coin_flip_bias", default=coin_flip_bias, type=float)
parser.add_argument("--savefig", default=savefig, type=int)
parser.add_argument("--save_images", default=save_images, type=int)
parser.add_argument("--image_scale", default=image_scale, type=int)
parser.add_argument("--psf_scale", default=psf_scale, type=float)
parser.add_argument("--psf_type", default=psf_type, type=str)
parser.add_argument("--max_iter", default=max_iter, type=int)
parser.add_argument("--thinning_type", default=thinning_type, type=str)
parser.add_argument("--out_dir", default=out_dir, type=str)
args = parser.parse_args()
try:
    args = parser.parse_args()
# globals().update(vars(args))
except:
    args = parser.parse_args([])
globals().update(vars(args))
print(vars(args))
# if __name__ == "__main__":
#     args = parser.parse_args()
# %%
if psf_type == "static":
    psf_switch = psf_switch_enum.STATIC
elif psf_type == "variable":
    psf_switch = psf_switch_enum.VAR_PSF
else:
    raise Exception("I don't recongise this psf type")

import pathlib

print(f"dir: {out_dir}")
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


# # Image formation

psf_w, psf_h, scale = psf_size, psf_size, image_scale
psf_window_w, psf_window_h = round(psf_w / scale), round(psf_h / scale)
# Define approximate PSF function


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))


def psf_guass(w=psf_w, h=psf_h, sigma=3):
    # blank_psf = np.zeros((w,h))
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    psf = gaussian(xx, 0, sigma) * gaussian(yy, 0, sigma)
    return psf / psf.sum()  # Normalise PSF "energy"


# Define a function that scales the PSF as a function of radial distance


def psf_vary(psf_window_h, psf_window_w, radius, scale):
    return psf_guass(
        w=round(psf_window_h),
        h=round(psf_window_w),
        sigma=(0.5 / scale) * abs((-0.4 * radius)) + 0.1,
    )


static_psf = psf_guass(
    w=round(psf_window_h), h=round(psf_window_w), sigma=psf_scale / scale
)
plt.imshow(static_psf)
plt.title("PSF")
plt.show()
# %% Deconvolution
from skimage.util import random_noise

astro = rescale(color.rgb2gray(data.human_mitosis()), 1.0 / scale)/signal_strength
astro_blur = conv2(astro, static_psf, "same")  # Blur image
# astro_noise = (
#     np.random.poisson(lam=25, size=astro_blur.shape)
# ) / 255.0  # Add Noise to Image


# astro_corrupt = astro_noisy = astro_blur + astro_noise  # Add Noise to Image

# astro_corrupt = (
#     np.random.poisson(lam=astro_blur * signal_strength, size=astro_blur.shape)
# ) / signal_strength

astro_corrupt = random_noise(astro_blur,"poisson")

deconvolved_RL = restoration.richardson_lucy(
    astro_blur, static_psf, iterations=10
)  # RL deconvolution

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 7))
ax[0].imshow(astro)
ax[0].set_title("Truth")
ax[1].imshow(astro_blur)
ax[1].set_title("Blurred")
ax[2].imshow(astro_corrupt)
ax[2].set_title("Blurred and noised")
ax[3].imshow(deconvolved_RL)
ax[3].set_title("Deconvolved")
plt.show()

# %%

N_v = np.ma.size(astro)
N_p = np.ma.size(astro_blur)
# measurement_matrix = matrix(np.zeros((N_p, N_v)))


zero_image = np.zeros_like(astro)
psf_window_volume = np.full((N_v, psf_window_w, psf_window_h), np.NaN)

x_astro, y_astro = astro_blur.shape
xx_astro, yy_astro = np.meshgrid(
    np.linspace(-1, 1, x_astro), np.linspace(-1, 1, y_astro)
)


# Make the PSF vary across the image (as a function of radius)

r_map = np.sqrt(xx_astro ** 2 + yy_astro ** 2)
radius_samples = np.linspace(-1, 1, 5)
fig, ax = plt.subplots(nrows=1, ncols=len(radius_samples), figsize=(16, 7))

for i, radius in enumerate(np.linspace(-1, 1, 5)):
    psf_current = psf_vary(psf_window_h, psf_window_w, radius, scale)
    ax[i].imshow(psf_current)
    ax[i].set_title("Radius: " + str(radius))
plt.show()


if psf_switch == psf_switch_enum.STATIC:
    filename = "H_staticpsf"
if psf_switch == psf_switch_enum.VAR_PSF:
    filename = "H_varpsf"

# %% Loop over each row of H and insert a flattened PSF
# that is the same shape as the input image
import dask
import dask.array as da


def make_flat_psf(i, psf_switch, static_psf, astro):
    coords = np.unravel_index(i, np.array(astro.shape))
    r_dist = r_map[coords]  # Convert to radius
    # Select mode for generating H, i.e. static/varying psf etc.
    if psf_switch == psf_switch_enum.STATIC:
        psf_current = static_psf
    if psf_switch == psf_switch_enum.VAR_PSF:
        psf_current = psf_vary(psf_window_h, psf_window_w, r_dist, scale)

    psf_window_volume[i, :, :] = psf_current
    delta_image = np.zeros_like(astro)
    delta_image[np.unravel_index(i, astro.shape)] = 1
    # Convolve PSF with a image with a single 1 at coord
    delta_PSF = scipy.ndimage.convolve(delta_image, psf_current)
    return delta_PSF.flatten()


delayed_list = []
for i in tqdm(np.arange(N_v)):
    # Get the xy coordinates of the ith pixel in the original image
    # delayed_list.append(
    delayed_obj = dask.delayed(make_flat_psf)(i, psf_switch, static_psf, astro)
    delayed_list.append(
        da.from_delayed(
            delayed_obj, shape=(np.multiply(*astro.shape),), dtype=np.float32
        )
    )
    # )
    # measurement_matrix[i, :] = delta_PSF.flatten()
# array = da.from_delayed(
#     delayed_list, (len(delayed_list),), dtype=float
# )
from dask.diagnostics import ProgressBar

with ProgressBar():
    stack = da.stack(delayed_list, axis=0)
    measurement_matrix = np.array(stack)
#  stack.compute()
# %%
# measurement_matrix = stack
plt.figure(figsize=(18, 7))
plt.imshow(exposure.equalize_hist(measurement_matrix), cmap="gray_r")
# %%

H = scipy.sparse.linalg.aslinearoperator(measurement_matrix)
g_blurred = H.dot(astro.reshape(-1, 1))
# f = np.matrix(g_blurred) + astro_noise.reshape(g_blurred.shape)
# g_blurred
f = random_noise(np.matrix(g_blurred),"poisson")

fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(16, 7))

ax[0].imshow(f.reshape(astro.shape))
ax[0].set_title("Blurred")
ax[1].imshow(astro)
ax[1].set_title("Original")
# ax[2].imshow(g.reshape(astro.shape));ax[2].set_title("RL")
plt.show()

#  %% Deconvolve matrices
A = H
b = f
x0 = None
Rtol = 1e-6
NE_Rtol = 1e-6
# max_iter = 100
sigmaSq = 0.0
beta = 0.0
# integer_signal = np.rint(b*2**16).astype(int)
coin_flip_scale = (
    np.random.binomial([2**8] * len(b), coin_flip_bias) / [2**8] 
)
# coin_flip_scale = np.random.binomial([2 ** 8] * len(b), 0.5) / 2 ** 8

from skimage.transform import rescale, resize, downscale_local_mean

T_thinned = b.copy().reshape(astro.shape)
V_thinned = b.copy().reshape(astro.shape)

# thinning_type = "spatial_interlaced"

if thinning_type == "poisson":
    # %% FIsh thinning
    T_scaled = np.matrix(np.multiply(coin_flip_scale, np.array(b.copy()).T).T)
    V_scaled = b - T_scaled

if thinning_type == "spatial_interlaced":
    T_thinned = T_thinned.copy()
    V_thinned = V_thinned.copy()
    T_thinned[:, 0::2] = 0
    V_thinned[:, 1::2] = 0
    T_scaled = T_thinned
    V_scaled = V_thinned

if thinning_type == "spatial_interpolated":

    T_thinned_squished = T_thinned.copy()[:, 0::2]
    V_thinned_squished = V_thinned.copy()[:, 1::2]

    T_scaled = skimage.transform.rescale(
        T_thinned_squished, (1.0, 2.0), anti_aliasing=True
    )
    V_scaled = skimage.transform.rescale(
        V_thinned_squished, (1.0, 2.0), anti_aliasing=True
    )
    # T = T_scaled
    # V = V_scaled

if thinning_type == "spatial_repeated":
    T_thinned_squished = T_thinned.copy()
    V_thinned_squished = V_thinned.copy()

    T_thinned_squished[:, 0::2] = T_thinned_squished[:, 1::2]
    V_thinned_squished[:, 1::2] = V_thinned_squished[:, 0::2]

    T_scaled = T_thinned_squished
    V_scaled = V_thinned_squished

T = T_scaled.reshape(b.shape)
V = V_scaled.reshape(b.shape)

# FIsh thinning
# T = np.matrix(np.multiply(coin_flip_scale, np.array(b).T).T)
# V = b - T

# T = np.matrix(np.multiply(coin_flip_scale, np.array(b).T).T)
# V = b - T

# Spatial thinning

# T_thinned = b.copy().reshape(astro.shape)
# V_thinned = b.copy().reshape(astro.shape)

# T_thinned_squished = T_thinned.copy()[:,0::2]
# V_thinned_squished = V_thinned.copy()[:,1::2]

# T_thinned_squished = T_thinned.copy()
# V_thinned_squished = V_thinned.copy()

# T_thinned_squished[:,0::2] = T_thinned_squished[:,1::2]
# V_thinned_squished[:,1::2] = V_thinned_squished[:,0::2]

# T_thinned[:,0::2] = 0
# V_thinned[:,1::2] = 0

# T_scaled = T_thinned_squished
# V_scaled = V_thinned_squished

b = np.matrix(T).reshape(f.shape)
gt = astro.reshape(V.shape)

plt.imshow(V.reshape(astro.shape))
plt.show()

plt.imshow(T.reshape(astro.shape))
plt.show()

plt.imshow(T.reshape(astro.shape))
plt.show()
# b = T

# The A operator represents a large, sparse matrix that
# has dimensions [ nrays x nvoxels ]

nrays = A.shape[0]
nvoxels = A.shape[1]

# Pre-compute some values for use in stopping criteria below
b_norm = np.linalg.norm(b)
trAb = A.rmatvec(b)
trAb_norm = np.linalg.norm(trAb)

# Start the optimization from the initial volume of a focal stack.
if x0 != None:
    x = x0
else:
    x = trAb

Rnrm = np.zeros(max_iter)
Rnrm_v = np.zeros(max_iter)
Rnrm_T = np.zeros(max_iter)
Rnrm_V = np.zeros(max_iter)

residual = np.zeros(max_iter)
residual_v = np.zeros(max_iter)
residual_V = np.zeros(max_iter)
residual_T = np.zeros(max_iter)

Xnrm = np.zeros(max_iter)
NE_Rnrm = np.zeros(max_iter)
gt_error_l1 = np.zeros(max_iter)
gt_error_l2 = np.zeros(max_iter)

V_error_l1 = np.zeros(max_iter)
T_error_l1 = np.zeros(max_iter)
cross_correlation = np.zeros(max_iter)
gt_error_ssim = np.zeros(max_iter)
log_liklihood = np.zeros(max_iter)
log_liklihood_v = np.zeros(max_iter)
log_liklihood_V = np.zeros(max_iter)
log_liklihood_T = np.zeros(max_iter)
norm_v = np.zeros(max_iter)

eps = np.spacing(1)
tau = np.sqrt(eps)
sigsq = tau
minx = x.min()

# If initial guess has negative values, compensate
if minx < 0:
    x = x - min(0, minx) + sigsq

normalization = A.rmatvec(np.ones(nrays)) + 1
print(normalization.min(), normalization.max())
c = (
    A.matvec(x)
    + np.matrix(beta * np.ones(nrays)).T
    + np.matrix(sigmaSq * np.ones(nrays)).T
)
b = b + np.matrix(sigmaSq * np.ones(nrays)).T

# i =0

iterations = range(max_iter)
range_tqdm = tqdm(iterations)

def log_liklihood_x_given_Ax(x,Ax):
    return np.sum(
        np.multiply(np.log(Ax), (x)) - Ax - np.log(scipy.special.factorial(x))
    )

for i in range_tqdm:
    tic = time.time()
    x_prev = x

    # STEP 1: RL Update step
    # b / c
    # (c+1e-12).min()
    v = A.rmatvec(b / c)
    x = (np.multiply(x_prev, v)) / matrix(normalization).T
    if x.min() < 0:
        x = x - min(0, x.min()) + sigsq
    Ax = A.matvec(x)
    # b-Ax
    residual = np.linalg.norm(b - Ax)
    residual_v[i] = np.linalg.norm(v - Ax)
    residual_V[i] = np.linalg.norm(V - Ax)
    residual_T[i] = np.linalg.norm(T - Ax)
    norm_v[i] = np.linalg.norm(v)
    # residual_V = V - Ax

    c = (
        A.matvec(x)
        + np.matrix(beta * np.ones(nrays)).T
        + np.matrix(sigmaSq * np.ones(nrays)).T
    )
    # c = Ax + beta*np.ones(nrays) + sigmaSq*np.ones(nrays);

    # STEP 2: Compute residuals and check stopping criteria
    Rnrm[i] = residual / b_norm
    Xnrm[i] = np.linalg.norm(x - x_prev) / nvoxels
    Rnrm_V[i] = residual_V[i] / np.linalg.norm(V)
    Rnrm_T[i] = residual_T[i] / np.linalg.norm(T)
    Rnrm_v[i] = residual_v[i] / np.linalg.norm(v)

    # NE_Rnrm[i] = np.linalg.norm(trAb - A.rmatvec(Ax)) / trAb_norm
    # # disabled for now to save on extra rmatvec

    toc = time.time()
    range_tqdm.set_description_str(
        f"RL Iteration {i:0} ({(toc-tic):.4f} seconds)"
        f"Residual Norm: {Rnrm[i]:.4f}  (tol = {Rtol:.2f}"
        f"Residual V Norm: {Rnrm_V[i]:.5f} (tol = {Rtol:.2f}"
        f"Residual T Norm: {Rnrm_T[i]:.5f} (tol = {Rtol:.2f}"
        f"v Norm: {norm_v[i]:.4f} (tol = {Rtol:.2f}"
        # print '\tError Norm: %0.4g(tol = %0.2e)  ' % (NE_Rnrm[i], NE_Rtol)
        f"Update Norm:{Xnrm[i]:.4f}",
        refresh=False,
    )
    # range_tqdm.set_postfix({
    #     "RL Iteration":toc-tic
    # })
    # stop because residual satisfies ||b-A*x|| / ||b||<= Rtol
    if Rnrm[i] <= Rtol:
        break

    # stop because normal equations residual satisfies
    #  ||A'*b-A'*A*x|| / ||A'b||<= NE_Rtol
    # if NE_Rnrm[i] <= NE_Rtol:
    #    break
    gt_error_l2[i] = mean_squared_error(x, gt)
    gt_error_l1[i] = mean_absolute_error(x, gt)

    V_error_l1[i] = mean_absolute_error(Ax, V)
    T_error_l1[i] = mean_absolute_error(Ax, T)

    x_flat = np.array(x.copy()).flatten()
    gt_flat = np.array(gt.copy()).flatten()

    gt_error_ssim[i] = ssim(x_flat, gt_flat)

    x_flat_scaled = preprocessing.scale(x_flat)
    gt_flat_scaled = preprocessing.scale(gt_flat)

    # a_flat_scaled = (x_flat - np.mean(x_flat)) / (np.std(x_flat) * len(x_flat))
    # b_flat_scaled = (gt_flat - np.mean(gt_flat)) / (np.std(gt_flat))
    # cross_correlation[i] = np.sum(np.correlate(a_flat_scaled, b_flat_scaled, 'full'))
    cross_correlation[i] = np.sum(np.correlate(x_flat_scaled, gt_flat_scaled, "full"))
    # a = np.dot(abs(x_flat_scaled),abs(gt_flat_scaled),'full')

    log_liklihood[i] = log_liklihood_x_given_Ax(b,Ax)
    log_liklihood_v[i] = log_liklihood_x_given_Ax(v,Ax)
    log_liklihood_V[i] = log_liklihood_x_given_Ax(V,Ax)
    log_liklihood_T[i] = log_liklihood_x_given_Ax(T,Ax)

# data = pd.DataFrame()
# metadata = pd.DataFrame(vars(args),index=[0])

vars_dict = dict(
    {
        i: eval(i)
        for i in [
            "iterations",
            "gt_error_l2",
            "gt_error_l1",
            "V_error_l1",
            "T_error_l1",
            "gt_error_ssim",
            "cross_correlation",
            "log_liklihood",
            "log_liklihood_v",
            "log_liklihood_V",
            "log_liklihood_T",
            "residual_v",
            "residual_V",
            "residual_T",
            "norm_v",
            "Rnrm",
            "Xnrm",
            "Rnrm_V",
            "Rnrm_T",
            "Rnrm_v",
        ]
    }
)
vars_dict.update(vars(args))
data = pd.DataFrame.from_dict(vars_dict)
data


data.to_csv(os.path.join(out_dir, "data.csv"))

from PIL import Image, ImageOps
from skimage.exposure import rescale_intensity

Image.fromarray(rescale_intensity(T.reshape(astro.shape), out_range=np.uint8)).convert("L").save(os.path.join(out_dir, "T.png"))
Image.fromarray(rescale_intensity(f.reshape(astro.shape), out_range=np.uint8)).convert("L").save(os.path.join(out_dir, "f.png"))
Image.fromarray(rescale_intensity(V.reshape(astro.shape), out_range=np.uint8)).convert("L").save(os.path.join(out_dir, "V.png"))
Image.fromarray(rescale_intensity(g_blurred.reshape(astro.shape), out_range=np.uint8)).convert("L").save(os.path.join(out_dir, "g_blurred.png"))
Image.fromarray(rescale_intensity(astro, out_range=np.uint8)).convert("L").save(os.path.join(out_dir, "raw.png"))


#  %%
plt.plot(Rnrm_v[1:-1])
plt.yscale("log")
plt.title("v residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()

plt.plot(Rnrm_V[1:-1])
plt.yscale("log")
plt.title("V residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()

plt.plot(Rnrm_T[1:-1])
plt.yscale("log")
plt.title("T residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()

plt.plot(residual_v[1:-1])
plt.title("v residuals")
plt.xlabel("Iterations")
plt.ylabel(" residuals")
plt.show()

plt.plot(residual_V[1:-1])
plt.title("V residuals")
plt.xlabel("Iterations")
plt.ylabel(" residuals")
plt.show()

plt.plot(residual_T[1:-1])
plt.title("T residuals")
plt.xlabel("Iterations")
plt.ylabel(" residuals")
plt.show()

plt.plot(Rnrm[1:-1])
plt.title("Residuals")
plt.xlabel("Iterations")
plt.ylabel("Normalised residuals")
plt.show()

plt.plot(V_error_l1[1:-1])
plt.title("Ax v V ~ L2")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(V_error_l1[1:-1])
plt.title("Ax v T ~ L1")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(gt_error_l2[1:-1])
plt.title("GT v x ~ L1")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(gt_error_l1[1:-1])
plt.title("GT v x ~ L2")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()

plt.plot(cross_correlation[1:-1])
plt.title("GT v x ~ cross_correlation")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.show()


plt.plot(gt_error_ssim[1:-1])
plt.title("GT v x ~ SSIM")
plt.xlabel("Iterations")
plt.ylabel("SSIM")
plt.show()

plt.plot(log_liklihood[1:-1])
plt.title("GT v x ~ log_liklihood")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()

plt.plot(log_liklihood_v[1:-1])
plt.title("x vs validation ~ log_liklihood_v")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()


# plt.plot(Rnrm_t[2:-1])
# plt.title("T residuals")
# plt.xlabel("Iterations")
# plt.ylabel("Normalised residuals")
# plt.show()


plt.plot(log_liklihood_T[1:-1])
plt.title("x vs validation ~ log_liklihood_T")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()

plt.plot(log_liklihood_V[1:-1])
plt.title("x vs validation ~ log_liklihood_V")
plt.xlabel("Iterations")
plt.ylabel("log_liklihood")
plt.show()


# # %%
# # compute second derivative
# from scipy.ndimage import gaussian_filter1d

# smooth = gaussian_filter1d(Rnrm_v, 10)
# smooth = Rnrm_v[20:-1]
# plt.plot(smooth)
# plt.show()
# #  %%
# diff = np.diff(np.diff(smooth,axis=0),axis=0)[1:-2];
# plt.plot(diff)
# plt.show()
# #  %%
# inf_point = np.where ( Rnrm_v == Rnrm_v.max())[0][0]
# inf_point
# plt.plot(smooth)
# plt.vlines(x=inf_point,ymin=smooth.min(),ymax=smooth.max())
# plt.show()
# # find switching points
# # infls = np.where(np.diff(np.sign(diff)))[0]

# # %%

# %%

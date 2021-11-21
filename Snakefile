# configfile: "config.yaml"
import numpy as np
import os
# envvars:
#     "SHELL"


# PSF_TYPE = ["static","variable"]
# THINNING_TYPE = ["poisson","spatial_interlaced",
                # "spatial_interpolated", "spatial_repeated","none"]
THINNING_TYPE = "poisson"
# PSF_TYPE = "static"
# PSF_SCALE = 1
MAX_ITER = 200

SIGNAL_STRENGTH = 2**8
SIGNAL_STRENGTH = np.round(np.linspace(1,0.1,10),2)

COIN_FLIP_BIAS = 0.5
COIN_FLIP_BIAS = np.round(np.linspace(0.5,1,11),2)
SAVEFIG = 1
SAVE_IMAGES = 0
IMAGE_SCALE = 4

PSF_SCALE = 0.5
PSF_SCALE = np.round(np.linspace(0.1,1,10),2)
PSF_GRADIENT = np.round(np.linspace(0,1,11),2)

BACKGROUND_L = 1
BACKGROUND_K = 0

PSF_SIZE = 64

results = "results/{psf_scale}-{psf_gradient}-{background_l}-{background_k}-{signal_strength}-{coin_flip_bias}-{max_iter}-{thinning_type}"

all_results = expand(results,
            base_dir = workflow.basedir,
            psf_scale=PSF_SCALE,
            psf_gradient=PSF_GRADIENT,
            signal_strength=SIGNAL_STRENGTH,
            thinning_type=THINNING_TYPE,
            coin_flip_bias=COIN_FLIP_BIAS,
            max_iter=MAX_ITER,
            background_l=BACKGROUND_L,
            background_k=BACKGROUND_K)

# rule all:
#     input:
#         "out/{psf_type}_{psf_width}_{signal_strength}.csv"
# expand("out/{psf_type}_{psf_width}_{signal_strength}.csv",psf_type=PSF_TYPE,PSF_WIDTH,SIGNAL_STRENGTH)

base_dir = workflow.current_basedir
script = os.path.join(workflow.basedir,"simulate.py")

rule all:
    input:
        all_results
        # "out/{psf_type}_{psf_width}_{signal_strength}.csv"


rule simulate:
    # input:
    #     "{basedir}/040520_pres_cluster_coins.py"
    conda:
        "environment.yml"
    params:
        # shell=os.environ["SHELL"]
        # psf_scale="{psf_scale}"
        # psf_type = "{psf_type}",
        # psf_scale = "{psf_scale}",
        # signal_strength = "{signal_strength}",
        # thinning_type = "{thinning_type}"
    resources:
        mem_mb=12000
    output:
        directory(results)
    shell:
        """
	    python {script} \
        --out_dir {output} \
        --psf_scale {wildcards.psf_scale} \
        --psf_gradient {wildcards.psf_gradient} \
        --signal_strength {wildcards.signal_strength} \
        --thinning_type {wildcards.thinning_type} \
        --coin_flip_bias {wildcards.coin_flip_bias} \
        --max_iter {wildcards.max_iter} \
        --background_l {wildcards.background_l} \
        --background_k {wildcards.background_k} \
        """


# parser.add_argument("--signal_strength", default=signal_strength, type=float)
# parser.add_argument("--coin_flip_bias", default=coin_flip_bias, type=float)
# parser.add_argument("--savefig", default=savefig, type=int)
# parser.add_argument("--save_images", default=save_images, type=int)
# parser.add_argument("--image_scale", default=image_scale, type=int)
# parser.add_argument("--psf_scale", default=psf_scale, type=float)
# parser.add_argument("--psf_gradient", default=psf_gradient, type=float)
# parser.add_argument("--psf_type", default=psf_type, type=str)
# parser.add_argument("--max_iter", default=max_iter, type=int)
# parser.add_argument("--thinning_type", default=thinning_type, type=str)
# parser.add_argument("--out_dir", default=out_dir, type=str)
# parser.add_argument("--background_L", default=background_L, type=float)
# parser.add_argument("--background_k", default=background_k, type=float)
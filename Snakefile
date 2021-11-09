# configfile: "config.yaml"
import numpy as np
import os
# envvars:
#     "SHELL"


PSF_TYPE = ["static","variable"]
THINNING_TYPE = ["poisson","spatial_interlaced",
                "spatial_interpolated", "spatial_repeated","none"]
# THINNING_TYPE = "poisson"

PSF_TYPE = "static"
# PSF_SCALE = 1
MAX_ITER = 200

SIGNAL_STRENGTH = 2**8
SIGNAL_STRENGTH = np.round(np.linspace(1,0,11),2)

COIN_FLIP_BIAS = 0.5
COIN_FLIP_BIAS = np.round(np.linspace(0.5,1,11),2)
SAVEFIG = 1
SAVE_IMAGES = 0
IMAGE_SCALE = 4

PSF_SCALE = 0.5
PSF_SCALE = np.round(np.linspace(0,1,11),2)
PSF_SIZE = 64

results = "results/{psf_scale}-{signal_strength}-{coin_flip_bias}-{max_iter}-{thinning_type}-{psf_type}"

all_results = expand(results,
            base_dir = workflow.basedir,
            psf_scale=PSF_SCALE,
            signal_strength=SIGNAL_STRENGTH,
            thinning_type=THINNING_TYPE,
            psf_type=PSF_TYPE,
            coin_flip_bias=COIN_FLIP_BIAS,
            max_iter=MAX_ITER)
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
        --psf_type {wildcards.psf_type} \
        --psf_scale {wildcards.psf_scale} \
        --signal_strength {wildcards.signal_strength} \
        --thinning_type {wildcards.thinning_type} \
        --coin_flip_bias {wildcards.coin_flip_bias} \
        --max_iter {wildcards.max_iter}
        """

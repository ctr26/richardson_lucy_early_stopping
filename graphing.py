# %%
import pandas as pd
import os
import glob as glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd

# configfile: "config.yaml"
import numpy as np

WORKING_DIR = "snakeout"
PSF_TYPE = ["static", "variable"]
BEST_METRICS = ["psnr_V", "Rnrm_V", "log_liklihood_V",
                "gt_error_ssim", "gt_error_l2"]
THINNING_TYPES = [
    "poisson",
    "spatial_interlaced",
    "spatial_interpolated",
    "spatial_repeated",
]
THINNING_TYPE = "poisson"

# PSF_SCALE = 1
PSF_TYPE = ["static", "variable"]
# THINNING_TYPE = "poisson"

PSF_TYPE = "static"
# PSF_SCALE = 1
MAX_ITER = 200
# %%
from dask.diagnostics import ProgressBar

folder = "results_full"
file = f"{folder}.csv"
# %%

df = pd.read_csv(file)

# %%
THINNING_TYPE = "poisson"
SIGNAL_STRENGTH = 1.0
COIN_FLIP_BIAS = 0.5
SAVEFIG = 1
SAVE_IMAGES = 0
IMAGE_SCALE = 4
PSF_SCALE = 0.5
PSF_SIZE = 64

index = ["psf_type", "psf_scale", "thinning_type", "coin_flip_bias", "signal_strength"]
metadata_index = ["savefig", "save_images", "out_dir", "max_iter", "image_scale"]
x_axis = ["iterations"]
# df_dropped = df.drop(metadata_index, axis=1)
df_clean = df.iloc[:, 2:][BEST_METRICS+index+x_axis]
df_indexed = df_clean.set_index(index)
# df_full = df_indexed.drop(metadata_index, axis=1)
variables = df_indexed.columns

df_melt = pd.melt(df_indexed, id_vars=["iterations"],
        var_name="metric",
        ignore_index=False).set_index(["metric"],
        append=True)

df_melt_slim = df_melt

# df_melt_slim = df_melt.xs(
#     (PSF_SCALE,COIN_FLIP_BIAS),
#     level=("psf_scale","coin_flip_bias"),
# )
# df_melt_slim

# df_melt_slim.xs("log_liklihood_V",level="metric").plot()
from sklearn.preprocessing import minmax_scale

# df_melt_slim["normalised_value"] =
vars_list = list(df_melt_slim.index.names)
# vars_list.remove("iterations");vars_list
# %%
ddf = dd.from_pandas(df_melt_slim.reset_index(), npartitions=32)
groups = (
    ddf.groupby(vars_list)
    # .apply(minmax_scale,axis=0)
)

# df_melt_slim["normalised_values"]

# group = groups.get_group(list(groups.groups)[9])
# group
# from IPython.display import display


def helper(df):
    # display(df)
    try:
        df["value"] = minmax_scale(df["value"])
    except :
        df["value"] = 0
    return df


# temp = group.apply(helper)
normalised_df = groups.apply(helper)
with ProgressBar():
    normalised_df = normalised_df.compute()
normalised_df = normalised_df.set_index(vars_list)
# temp_df
# groups.apply(lambda x: [1]*200)
# df_melt_slim
# df_for_plot = normalised_df[["iterations","values_normalised"]].loc[
#     pd.IndexSlice[:, :, :, :, ["log_liklihood_V", "log_liklihood_T"]]
# ]


# %%
# df_for_plot = normalised_df[["iterations", "value"]].xs(
#     "log_liklihood_V", level="metric"
# )
# pd.concat([df1, df2, df2])

normalised_df_clipped = pd.concat([
                normalised_df.reset_index(),
                normalised_df.xs(0.1,level="psf_scale",drop_level=False).reset_index(),
                normalised_df.xs(1.0,level="coin_flip_bias",drop_level=False).reset_index(),
                normalised_df.xs(0.1,level="signal_strength",drop_level=False).reset_index()
                ])\
                .drop_duplicates(keep=False).set_index(vars_list)

# %%
dict_of_vars = {
    "psf_scale": PSF_SCALE,
    "coin_flip_bias": COIN_FLIP_BIAS,
    "signal_strength": SIGNAL_STRENGTH
}

for leave_out in dict_of_vars:
    current_dict = dict_of_vars.copy()
    current_dict.pop(leave_out)
    # print(current_dict)

# df_for_plot = normalised_df[["iterations", "value"]]
    df_for_plot = normalised_df_clipped.xs(
        tuple(current_dict.values()),
        level=tuple(current_dict.keys()),
    )

    sns.lmplot(
        x="iterations",
        y="value",
        hue=leave_out,
        sharey=False,
        sharex=False,
        row="metric",
        col="thinning_type",
        col_order=THINNING_TYPES,
        row_order=BEST_METRICS,
        # fit_reg=False,
        ci=None,
        fit_reg=False,
        data=df_for_plot.reset_index(),
    ).set(xlim=(1,50))
    plt.show()
# # %%
# from bokeh.plotting import figure,show
# from bokeh.models import ColumnDataSource
# from bokeh.models import CDSView, ColumnDataSource, IndexFilter,GroupFilter


# groups = df_melt_slim.reset_index().groupby(vars_list)
# source = ColumnDataSource(groups)

# p = figure()
# p.circle(x='iterations', y='value', source=source)
# # %%

# # dict_df = df_melt_slim.reset_index().to_dict()

# source = ColumnDataSource(data=df_melt_slim.reset_index())
# view = CDSView(source=source, filters=[GroupFilter(
#         column_name = ['psf_scale'],
#         group = [str(PSF_SCALE)],
# )])

# # (df.groupby('Column1')['Column3'].apply(list))
# # %%


# df_melt_slim_super_slim = df_melt_slim.xs((PSF_TYPE,PSF_SCALE,COIN_FLIP_BIAS,SIGNAL_STRENGTH),
#                                     level=("psf_type","psf_scale","coin_flip_bias","signal_strength"))

# df_melt_slim_super_slim.to_csv("df_melt_slim_super_slim.csv")


# # %%

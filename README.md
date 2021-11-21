


# Nomenclature

- signal_strength
    - Between 1 and 255 scaling of image intensity before poisson noise
-  psf_scale
   -  gaussian psf width in sigma (I think)
- coin_flip_bias
    -  bionomial odds

-  thinning_type:
    - Split image into training (T) and validation (V)
    - poisson:
        - Bionomial splitting
    - spatial:
        -  All spatial thinning methods split the image into two
        -  spatially rather than per pixel.
        -  To make these images work with the rest of the code
        -  they need to be stretched in the thinned axis
    -  spatial_interlaced
        -  zero out signal every other pixel
    -  spatial_interpolated
        -  smooth linear interp
    -  spatial_repeated:
        -  take left pixel and copy to right pixel
-  metrics:
    -  gt_error_l[1/2]:
        -  l[1/2]_norm between x and ground_truth
    -  gt_error_ssim
        -  structural similiarity
    -  Rnrm_[VAR]:
        -  Sum of normalised residuals between T and V
    -  log_liklihood_[VAR]
        -  Liklihood between Ax and [VAR]

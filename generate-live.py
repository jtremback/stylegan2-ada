
import argparse
import sys
import os
import subprocess
import pickle
import re

import scipy
import numpy as np
from numpy import linalg
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# import moviepy.editor
# from opensimplex import OpenSimplex

import warnings # mostly numpy warnings for me
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def generate_image(Gs, seed, truncation_psi):
    """Generate an image

    Arguments:
    Gs: initialized neural network
    seed: seed, can be array of 32bit integers to get to desired precision
    truncation_psi: uniqueness, float

    Returns:
    PIL image
    """

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False,
        'truncation_psi': truncation_psi
    }

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    noise_rnd = np.random.RandomState(1) # fix noise
    tflib.set_vars({var: noise_rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    image = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    out_image = PIL.Image.fromarray(image[0], 'RGB')#.save(f'{outdir}/seed{seed:04d}.{image_format}', optimize=optimized, quality=jpg_quality)

    return out_image
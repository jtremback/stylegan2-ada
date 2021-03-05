import argparse
import sys
import os
import pickle

import dnnlib
import dnnlib.tflib as tflib
import dg_lib
import json
import numpy as np


def generate_image(network_pkl, seed, truncation_psi, outdir):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    seed = json.loads(seed)

    seed = np.array(seed)

    seed = seed.astype(np.int)

    print(seed)

    image = dg_lib.generate_image(Gs, seed, truncation_psi)

    image.save(f'{outdir}/seed-{seed}_psi-{truncation_psi}.jpg',
               optimize=True, quality=85)

# ----------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(description='Generate an image using pretrained network pickle.')
    
    parser.add_argument(
        '--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument(
        '--seed', help='Seed', dest='seed', required=True)
    parser.add_argument(
        '--psi', type=float, help='Psi', dest='truncation_psi', required=True)
    parser.add_argument(
        '--out', help='Output directory', dest='outdir', required=True)

    args = parser.parse_args()

    generate_image(args.network_pkl, args.seed, args.truncation_psi, args.outdir)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

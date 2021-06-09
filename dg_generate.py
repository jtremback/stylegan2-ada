import argparse
import sys
import os
import pickle

import dnnlib
import dnnlib.tflib as tflib
import dg_lib
import json
import numpy as np


def generate_image(network_pkl, tokenId, outdir):

    # Convert tokenId to hex and pad it out to 64 chars
    t_i = hex(int(tokenId, 10))[2:].zfill(64)

    seed = [
        int(t_i[0:8], 16),
        int(t_i[8:16], 16),
        int(t_i[16:24], 16),
        int(t_i[24:32], 16),
    ]
    seed = np.array(seed)
    seed = seed.astype(np.int)

    truncation_psi = (int(t_i[32:64], 16) / 100000000000000000000) ** (3/5)

    network_pkl = 'network-snapshot.pkl'

    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    image = dg_lib.generate_image(Gs, seed, truncation_psi)

    image.save(f'{outdir}/seed-{seed}_psi-{truncation_psi}.jpg',
               optimize=True, quality=85)

    return image

# ----------------------------------------------------------------------------


def main():

    parser = argparse.ArgumentParser(
        description='Generate an image using pretrained network pickle.')

    parser.add_argument(
        '--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument(
        '--tokenId', help='Token Id', dest='tokenId', required=True)
    parser.add_argument(
        '--out', help='Output directory', dest='outdir', required=True)

    args = parser.parse_args()

    generate_image(args.network_pkl, args.tokenId, args.outdir)
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

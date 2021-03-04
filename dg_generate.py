import argparse
import sys
import os
import pickle

import dnnlib
import dnnlib.tflib as tflib
import dg_lib


def generate_image(network_pkl, seed, truncation_psi, outdir):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    os.makedirs(outdir, exist_ok=True)

    image = lib.generate_image(Gs, seed, truncation_psi)

    image.save(f'{outdir}/seed{seed:04d}.jpg',
               optimize=True, quality=85)

# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Generate an image using pretrained network pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    generate_image = subparsers.add_parser(
        'generate-image', help='Generate an image')
    generate_image.add_argument(
        '--network', help='Network pickle filename', dest='network_pkl', required=True)
    generate_image.add_argument(
        '--seed', help='Seed', dest='seed', required=True)
    generate_image.add_argument(
        '--psi', help='Psi', dest='truncation_psi', required=True)
    generate_image.add_argument(
        '--out', help='Output directory', dest='outdir', required=True)

    generate_image.set_defaults(func=generate_image)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    func = kwargs.pop('func')
    func(**kwargs)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

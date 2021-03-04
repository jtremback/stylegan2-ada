import warnings  # mostly numpy warnings for me
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json

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
import dg_lib


def startServer(network_pkl):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    class GemServer(BaseHTTPRequestHandler):
        def do_GET(self):
            numbers = json.loads(self.path)
            image = lib.generate_image(Gs, numbers[0:6], numbers[6])
            image.save(f'./out.jpg', optimize=True, quality=85)
            content = open("./out.jpg", 'rb')

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(content.read())

    hostName = "localhost"
    hostPort = 9000

    gemServer = HTTPServer((hostName, hostPort), GemServer)
    print(time.asctime(), "Server Starts - %s:%s" % (hostName, hostPort))

    try:
        gemServer.serve_forever()
    except KeyboardInterrupt:
        pass

    gemServer.server_close()
    print(time.asctime(), "Server Stops - %s:%s" % (hostName, hostPort))


# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Set up a server to generate images using pretrained network pickle.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    start_server = subparsers.add_parser(
        'start-server', help='Start server')
    start_server.add_argument(
        '--network', help='Network pickle filename', dest='network_pkl', required=True)
    start_server.set_defaults(func=startServer)

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

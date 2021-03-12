from http.server import BaseHTTPRequestHandler, HTTPServer
import time
import json

import argparse
import pickle

import dnnlib
import dnnlib.tflib as tflib
import dg_lib


def start_server(network_pkl):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    class GemServer(BaseHTTPRequestHandler):
        def do_GET(self):
            print(self.path[1:])
            numbers = json.loads(self.path[1:])
            image = dg_lib.generate_image(Gs, numbers[0:4], numbers[4])
            image.save(f'/tmp/out.jpg', optimize=True, quality=85)
            content = open("/tmp/out.jpg", 'rb')

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
        description='Set up a server to generate images using pretrained network pickle.')

    parser.add_argument(
        '--network', help='Network pickle filename', dest='network_pkl', required=True)

    args = parser.parse_args()

    start_server(args.network_pkl)

# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------

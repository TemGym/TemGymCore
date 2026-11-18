"""Runs on the Merlin PC: answers a GET request with one Merlin frame, as a NumPy array, over the network.

    python merlin_server.py            listens on port 8765 and waits
    python merlin_server.py --fake     the same, with a made-up frame (to try the network before the Merlin is hooked up)
    add --port=N, --host=ADDRESS (default 0.0.0.0: reachable from other computers), --token=WORD

The microscope PC asks for a frame with GET http://<this PC>:8765/frame (merlin_camera.py does that).
    /frame   runs read_frame() below, once, and sends back what it returns
    /ping    answers "ok" and takes nothing: to check the connection

The ONE LINE you supply is in read_frame() below. Nothing a caller sends is ever run: a request can only start that line.

The array comes back in NumPy's own file format (np.save), so its shape and data type arrive with it. Requests are
answered one at a time, so two callers never read the Merlin at once.

With --token=WORD a caller has to send the same word (merlin_camera.py: MERLIN_TOKEN), or it is refused. Anyone on the
network who can reach this port can otherwise start a frame: keep it to the lab network, or use a token.
"""

import io
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def read_frame():
    """THE LINE OF CODE YOU SUPPLY: read one Merlin frame and return it as a 2-D NumPy array (counts)."""
    raise NotImplementedError("merlin_server.py: put in read_frame() the line that reads one Merlin frame, "
                              "for example:  return my_merlin_module.get_frame()")


def fake_frame():
    """A made-up 512 x 512 frame (uint16): a bright round spot on a dark background."""
    y, x = np.mgrid[:512, :512]
    spot = 4000 * np.exp(-((x - 256 + 5 * np.random.randn()) ** 2 + (y - 256) ** 2) / (2 * 24.0 ** 2))
    return (100 + spot + np.random.poisson(5, (512, 512))).astype(np.uint16)


def option(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


HOST, PORT, TOKEN = option("host", "0.0.0.0"), int(option("port", 8765)), option("token", "")
if "--fake" in sys.argv:
    read_frame = fake_frame                      # noqa: F811 (replaces the line above for a try-out)


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body=b"", kind="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if TOKEN and self.headers.get("X-Token") != TOKEN:
            return self.reply(403, b"wrong or missing token")
        if self.path == "/ping":
            return self.reply(200, b"ok")
        if self.path != "/frame":
            return self.reply(404, b"ask for /frame or /ping")
        try:
            started = time.time()
            frame = np.asarray(read_frame())
            file = io.BytesIO()
            np.save(file, frame, allow_pickle=False)
            self.reply(200, file.getvalue(), "application/octet-stream")
            print(f"{time.strftime('%H:%M:%S')}  sent a frame {frame.shape} {frame.dtype} to {self.client_address[0]} "
                  f"({time.time() - started:.2f} s)")
        except NotImplementedError as error:
            self.reply(501, str(error).encode())
            print("read_frame() is not filled in yet")
        except Exception:
            self.reply(500, traceback.format_exc().encode())
            traceback.print_exc()

    def log_message(self, *arguments):           # the lines above say what matters
        pass


if __name__ == "__main__":
    print(f"Merlin server on {HOST}:{PORT}" + (" (made-up frames)" if "--fake" in sys.argv else "")
          + (", token needed" if TOKEN else ", no token") + ". Stop it with Ctrl+C.")
    try:
        HTTPServer((HOST, PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("stopped")

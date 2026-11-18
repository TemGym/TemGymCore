"""The Merlin, for when PyJEM does not list it as one of its cameras: frames come over the network.

record_two_cameras.py first looks for the Merlin among PyJEM's cameras (the name given with --merlin=NAME). If it is
not there, it uses the class below. The class asks the Merlin PC for a frame with a GET request: on that PC run
merlin_server.py, which runs the one line that reads a Merlin frame and sends back the NumPy array.

    Merlin PC:        python merlin_server.py                 (first fill in read_frame() in it; --fake to try the network)
    microscope PC:    set MERLIN_URL below, then
                      python merlin_camera.py                 asks for one frame and prints its size and type

Which camera the beam reaches is set by record_two_cameras.py (screen position 2 for the Merlin, with a 4 s wait), so there is no insert() or retract() here.
"""

import io
import sys
import urllib.error
import urllib.request

import numpy as np

MERLIN_URL = "http://merlin-pc:8765"       # the Merlin PC's name or address, and the port merlin_server.py listens on
MERLIN_TOKEN = ""                          # the same word as merlin_server.py --token=WORD (empty: none)
FRAME_SHAPE = (512, 512)                   # what a frame should be: record_two_cameras.py warns if it is not
TIMEOUT_S = 60                             # a frame may take a while (the exposure, then the read)


class Merlin:
    """One Merlin frame at a time, as a 2-D array of counts, from the Merlin PC."""

    def __init__(self, url=MERLIN_URL, token=MERLIN_TOKEN):
        self.url, self.token = url.rstrip("/"), token

    def ask(self, path):
        """GET ``path`` from the Merlin PC. Returns the answer's bytes; says plainly what went wrong if it fails."""
        request = urllib.request.Request(self.url + path, headers={"X-Token": self.token} if self.token else {})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as answer:
                return answer.read()
        except urllib.error.HTTPError as error:               # the server answered, with a complaint
            raise RuntimeError(f"the Merlin PC said {error.code}: {error.read().decode(errors='replace')[:2000]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise RuntimeError(f"cannot reach the Merlin PC at {self.url}: {error}. Is merlin_server.py running there, "
                               "and is MERLIN_URL in merlin_camera.py right?") from None

    def ping(self):
        """True if the Merlin PC answers (takes no frame)."""
        return self.ask("/ping") == b"ok"

    def take_frame(self):
        """GET /frame: one frame as a NumPy array (its shape and data type arrive with it)."""
        return np.load(io.BytesIO(self.ask("/frame")), allow_pickle=False)     # allow_pickle=False: data only, no code


if __name__ == "__main__":
    merlin = Merlin(*(sys.argv[1:2] or [MERLIN_URL]))
    print("asking", merlin.url, "->", "answers" if merlin.ping() else "no good answer")
    frame = merlin.take_frame()
    print(f"one frame: {frame.shape[1]} x {frame.shape[0]} px, {frame.dtype}, values {frame.min()} to {frame.max()}")
    if frame.shape != FRAME_SHAPE:
        print(f"   NOTE: record_two_cameras.py expects {FRAME_SHAPE}: change FRAME_SHAPE here and MERLIN_FRAME there")

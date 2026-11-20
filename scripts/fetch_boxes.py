"""Fetch MegaDetector v4 animal boxes for iWildCam (training labels for the crop classifier).

    python scripts/fetch_boxes.py --out $EYES_DATA/raw/iwildcam_mdv4

The boxes ship inside the 169 GB iWildCam 2022 archive on LILA. A zip keeps its table of
contents at the end, so HTTP range requests read the table and then only the 82 MB
detections file -- nothing else is downloaded.
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://storage.googleapis.com/public-datasets-lila/iwildcam-2022/iwildcam-2022.zip"
MEMBER = "iwildcam-2022/metadata/iwildcam2022_mdv4_detections.json"


class HTTPFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests."""

    def __init__(self, url):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default=URL)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(io.BufferedReader(HTTPFile(args.url), buffer_size=1 << 20))
    dest = out / Path(MEMBER).name
    with z.open(MEMBER) as src, open(dest, "wb") as dst:
        while chunk := src.read(1 << 22):
            dst.write(chunk)
    print("wrote", dest)


if __name__ == "__main__":
    main()

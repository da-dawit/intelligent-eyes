"""Resumable parallel download for large dataset files over a bad connection.

    python scripts/download.py URL DEST [--sha256 HEX] [--workers 8] [--piece-mb 32]

The file is split into pieces. Several threads fetch them with HTTP Range requests and
write them straight to disk. Finished pieces are recorded in DEST.parts, so a dropped
connection only retries the pieces in flight and a killed run resumes where it stopped.
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "python-urllib dataset-download"}


def total_size(url: str) -> int:
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")
        if cr is None:
            raise RuntimeError("server does not support range requests")
        return int(cr.split("/")[-1])


def fetch_piece(url, fd, start, end, stop):
    """Download bytes [start, end] into fd, retrying until done."""
    delay = 5
    while not stop.is_set():
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                pos = start
                while pos <= end:
                    block = r.read(min(1 << 20, end - pos + 1))
                    if not block:
                        raise IOError("connection closed early")
                    os.pwrite(fd, block, pos)
                    pos += len(block)
            return end - start + 1
        except Exception as e:  # network drop, timeout, 5xx: wait and retry this piece
            print(f"  piece {start}: {type(e).__name__}: {e}; retry in {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return 0


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--sha256", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--piece-mb", type=int, default=32)
    args = ap.parse_args()

    parts_file = args.dest + ".parts"
    if os.path.exists(args.dest) and not os.path.exists(parts_file):
        print(f"{args.dest} already complete")
        return
    size = total_size(args.url)
    piece = args.piece_mb << 20
    pieces = [(s, min(s + piece, size) - 1) for s in range(0, size, piece)]
    done = set(json.load(open(parts_file))) if os.path.exists(parts_file) else set()
    fd = os.open(args.dest, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    todo = [p for p in pieces if p[0] not in done]
    print(f"{args.dest}: {size / 2**30:.2f} GiB, {len(pieces)} pieces, {len(todo)} to fetch", flush=True)

    lock, stop = threading.Lock(), threading.Event()
    got = [sum(e - s + 1 for s, e in pieces if s in done)]
    t0, last = time.time(), [time.time()]
    start_bytes = got[0]

    def work(p):
        n = fetch_piece(args.url, fd, p[0], p[1], stop)
        if n:
            with lock:
                done.add(p[0])
                got[0] += n
                with open(parts_file + ".tmp", "w") as f:
                    json.dump(sorted(done), f)
                os.replace(parts_file + ".tmp", parts_file)
                if time.time() - last[0] > 30:
                    rate = (got[0] - start_bytes) / (time.time() - t0)
                    eta = (size - got[0]) / max(rate, 1)
                    print(f"  {got[0] / 2**30:.2f}/{size / 2**30:.2f} GiB  {rate / 2**20:.2f} MiB/s  eta {eta / 60:.0f} min",
                          flush=True)
                    last[0] = time.time()

    try:
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(work, todo))
    except KeyboardInterrupt:
        stop.set()
        sys.exit(1)
    finally:
        os.close(fd)

    if len(done) != len(pieces):
        sys.exit("incomplete; run again to resume")
    if args.sha256:
        digest = sha256_of(args.dest)
        if digest != args.sha256:
            sys.exit(f"SHA-256 mismatch: {digest}")
        print("SHA-256 ok")
    os.remove(parts_file)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

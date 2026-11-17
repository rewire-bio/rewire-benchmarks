"""Extract selected members of a remote zip archive using HTTP range requests (stdlib only).

Usage:
  rangezip.py URL --list [--grep TEXT]
  rangezip.py URL --dest DIR --member-glob PATTERN [--member-glob PATTERN ...]

Used to take a few assays from a multi-gigabyte archive without downloading all of it. Member
paths are kept, so the result unpacks exactly as a full `unzip` would for those files.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import sys
import urllib.request
import zipfile
from pathlib import Path


class RangeFile(io.RawIOBase):
    def __init__(self, url: str) -> None:
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as resp:
            self.size = int(resp.headers["Content-Length"])
            if resp.headers.get("Accept-Ranges") != "bytes":
                raise SystemExit(f"{url} does not advertise byte ranges")
        self.requests = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(buffer)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        self.requests += 1
        n = len(data)
        buffer[:n] = data
        self.pos += n
        return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--grep", default="")
    ap.add_argument("--dest")
    ap.add_argument("--member-glob", action="append", default=[])
    args = ap.parse_args()

    raw = RangeFile(args.url)
    with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20)) as zf:
        names = zf.namelist()
        if args.list:
            for info in zf.infolist():
                if args.grep in info.filename:
                    print(info.file_size, info.filename)
            return 0
        wanted = [n for n in names if any(fnmatch.fnmatch(n, g) for g in args.member_glob)]
        if not wanted:
            print("no members match", args.member_glob, file=sys.stderr)
            return 1
        dest = Path(args.dest)
        for name in wanted:
            zf.extract(name, dest)
        total = sum(zf.getinfo(n).file_size for n in wanted)
    print(f"extracted {len(wanted)} members, {total} bytes, {raw.requests} range requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())

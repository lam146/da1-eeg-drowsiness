"""Fetch only MindWave Mobile 2 EDF files from the public Ahn–Lee archive."""

import argparse
import io
import json
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from urllib.error import URLError


ARCHIVE = 'https://ndownloader.figshare.com/files/58088461'
SIZE = 400410665
CHUNK = 128 * 1024
DOI = 'https://doi.org/10.6084/m9.figshare.30162868'


def request_range(start, end):
    for attempt in range(3):
        # Request a fresh redirect. Signed storage URLs may expire mid-run.
        url = ARCHIVE if attempt == 0 else f'{ARCHIVE}?refresh={time.time_ns()}'
        request = Request(url, headers={'Range': f'bytes={start}-{end}'})
        try:
            with urlopen(request, timeout=20) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}':
                    raise OSError('Server did not honor the expected bounded byte range')
                data = response.read(end-start+2)
            if len(data) != end-start+1:
                raise OSError('Incomplete archive byte range')
            return data
        except (URLError, OSError):
            if attempt == 2:
                raise
            time.sleep(.5 * (attempt + 1))


class HttpRangeFile(io.RawIOBase):
    def __init__(self):
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: SIZE}[whence]
        if base + offset < 0:
            raise OSError('Negative archive offset')
        self.position = base + offset
        return self.position

    @lru_cache(maxsize=24)
    def chunk(self, index):
        start = index * CHUNK
        end = min(start + CHUNK, SIZE) - 1
        return request_range(start, end)

    def read(self, size=-1):
        end = min(SIZE, self.position + size) if size >= 0 else SIZE
        parts = []
        while self.position < end:
            block, offset = divmod(self.position, CHUNK)
            take = min(end - self.position, CHUNK - offset)
            parts.append(self.chunk(block)[offset:offset+take])
            self.position += take
        return b''.join(parts)


def fetch_entry(info, output):
    name = PurePosixPath(info.filename).name
    if not re.fullmatch(r'sub-\d+_task-(EB|BT|MVC|MVO)_acq-MW2_eeg\.edf', name):
        raise ValueError(f'Unexpected MindWave entry name: {name!r}')
    path = output / name
    if path.exists():
        data = path.read_bytes()
        if len(data) == info.file_size and zlib.crc32(data) == info.CRC:
            return dict(file=name, archive_path=info.filename, bytes=len(data), crc32=info.CRC)
    if info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED):
        raise ValueError(f'Unsupported archive entry: {name}')
    start = info.header_offset
    data = request_range(start, min(SIZE-1, start+30+4096+info.compress_size-1))
    if data[:4] != b'PK\x03\x04':
        raise ValueError(f'Invalid ZIP local header: {name}')
    header_size = 30 + sum(struct.unpack_from('<HH', data, 26))
    if len(data) < header_size + info.compress_size:
        data = request_range(start, start+header_size+info.compress_size-1)
    compressed = data[header_size:header_size+info.compress_size]
    plain = zlib.decompress(compressed, -15) if info.compress_type == zipfile.ZIP_DEFLATED else compressed
    if len(plain) != info.file_size or zlib.crc32(plain) != info.CRC:
        raise ValueError(f'ZIP CRC mismatch: {name}')
    path.write_bytes(plain)
    return dict(file=name, archive_path=info.filename, bytes=len(plain), crc32=info.CRC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    with HttpRangeFile() as source, zipfile.ZipFile(source) as archive:
        edfs = [info for info in archive.infolist() if info.filename.lower().endswith('.edf')]
        candidates = [info for info in edfs if re.search(r'_acq-(?:MW2|Mindwave2|MW)_', info.filename, re.I)]
        if not candidates or args.list:
            print('Acquisition codes:', sorted({name.split('_acq-')[-1].split('_')[0] for name in
                                                  (info.filename for info in edfs) if '_acq-' in name}))
            print('MindWave candidates:', len(candidates), 'compressed bytes:', sum(i.compress_size for i in candidates))
            print('\n'.join(info.filename for info in (candidates or edfs)[:12]))
            return
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries = pool.map(lambda info: fetch_entry(info, args.output), candidates)
        for index, entry in enumerate(entries, 1):
            manifest.append(entry)
            if index % 10 == 0 or index == len(candidates):
                print(f'{index}/{len(candidates)} files', flush=True)
    (args.output / 'source_manifest.json').write_text(json.dumps({'source': DOI, 'files': manifest}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

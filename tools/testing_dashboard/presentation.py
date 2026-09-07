"""Immutable PNG presentation assets, separate from live dashboard history. Stdlib only."""
import hashlib
from pathlib import Path
import re
import struct
import tempfile
import os
import zlib

MAX_FILE=8*1024*1024
MAX_STORE=512*1024*1024

def png_rgb(width,height,pixels):
    if type(width) is not int or type(height) is not int or not 1<=width<=1920 or not 1<=height<=1080:
        raise ValueError('PNG dimensions exceed presentation budget')
    if len(pixels)!=width*height*3:raise ValueError('RGB byte count mismatch')
    def chunk(kind,data):return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data)&0xffffffff)
    rows=b''.join(b'\0'+pixels[y*width*3:(y+1)*width*3] for y in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!2I5B',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows,6))+chunk(b'IEND',b'')

class AssetWriter:
    def __init__(self,root,*,budget=256*1024*1024):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.budget=budget;self.bytes=0;self.seen=set()

    def put(self,data):
        if not data.startswith(b'\x89PNG\r\n\x1a\n') or len(data)>MAX_FILE:raise ValueError('Invalid or oversized PNG')
        digest=hashlib.sha256(data).hexdigest();path=self.root/(digest+'.png')
        if digest not in self.seen:
            if self.bytes+len(data)>self.budget:raise ValueError('Presentation bundle budget exceeded')
            if path.exists():
                if path.is_symlink() or path.resolve().parent!=self.root or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                    raise ValueError('Existing presentation asset mismatch')
            else:
                if sum(p.stat().st_size for p in self.root.glob('*.png'))+len(data)>MAX_STORE:
                    raise ValueError('Presentation store budget exceeded; request reviewed cleanup')
                handle,name=tempfile.mkstemp(prefix='pending-',dir=self.root)
                try:
                    with os.fdopen(handle,'wb') as stream:stream.write(data)
                    os.replace(name,path)
                finally:
                    if os.path.exists(name):os.unlink(name)
            self.bytes+=len(data);self.seen.add(digest)
        return {'url':'/testing/media/'+digest+'.png','sha256':digest,'bytes':len(data)}

def read_asset(root,name):
    if not re.fullmatch(r'[0-9a-f]{64}\.png',name):raise FileNotFoundError('Unknown presentation asset')
    root=Path(root).resolve();path=root/name
    if path.is_symlink() or path.resolve().parent!=root:raise FileNotFoundError('Unknown presentation asset')
    if path.stat().st_size>MAX_FILE:raise FileNotFoundError('Oversized presentation asset')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=name[:-4]:raise FileNotFoundError('Presentation asset digest mismatch')
    return data

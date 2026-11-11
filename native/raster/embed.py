import sys
from pathlib import Path
data=Path(sys.argv[1]).read_bytes()+b'\0'
Path(sys.argv[2]).write_text('static const unsigned char gffx_raster_ptx[]={'+','.join(map(str,data))+'};\n',encoding='ascii')

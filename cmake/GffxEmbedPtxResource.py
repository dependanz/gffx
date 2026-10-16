"""Embed exact PTX bytes as a Windows DLL resource without a huge C initializer."""
from pathlib import Path
import argparse,json

def generate(source,payload,rc,header,resource_id=101):
    source,payload,rc,header=map(Path,(source,payload,rc,header))
    size=source.stat().st_size
    if not 0<size<4294967295 or type(resource_id)!=int or not 0<resource_id<65536:raise ValueError('nonempty bounded PTX and16-bit resource id required')
    raw=source.read_bytes()
    for p in (payload,rc,header):p.parent.mkdir(parents=True,exist_ok=True)
    payload.write_bytes(raw+b'\0')
    rc.write_text('#pragma code_page(65001)\nLANGUAGE 0,0\n'+str(resource_id)+' RCDATA '+json.dumps(payload.resolve().as_posix(),ensure_ascii=False)+'\n',encoding='utf8')
    header.write_text('/* Exact PTX payload resides in the same DLL resource. */\n#define GFFX_PTX_RESOURCE_ID '+str(resource_id)+'\nstatic const unsigned long gffx_cuda_embedded_ptx_size = '+str(size)+'UL;\n',encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--payload',required=True);parser.add_argument('--rc',required=True);parser.add_argument('--header',required=True);args=parser.parse_args();generate(args.input,args.payload,args.rc,args.header)

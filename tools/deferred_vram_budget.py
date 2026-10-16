"""Pure host-side admission policy; render() never samples ambient free memory."""
MIB=1024**2
GIB=1024**3


def choose_budget(*,free_bytes,total_bytes,allocated_bytes,ceiling_bytes=None):
    for name,value in (('free',free_bytes),('total',total_bytes),('allocated',allocated_bytes)):
        if type(value) is not int:raise TypeError(name+' bytes must be integer')
        if value<0:raise ValueError(name+' bytes must be nonnegative')
    if total_bytes<=0 or free_bytes+allocated_bytes>total_bytes:raise ValueError('inconsistent device memory snapshot')
    if ceiling_bytes is not None and (type(ceiling_bytes) is not int or ceiling_bytes<=0):raise ValueError('caller ceiling must be positive integer bytes')
    headroom=max(GIB,(free_bytes+4)//5)
    if free_bytes<=headroom:raise ValueError('insufficient free VRAM after headroom')
    limit=allocated_bytes+free_bytes-headroom
    if ceiling_bytes is not None:limit=min(limit,ceiling_bytes)
    limit=limit//MIB*MIB
    if limit<=allocated_bytes:raise ValueError('no additional tensor budget')
    return dict(free_bytes=free_bytes,total_bytes=total_bytes,allocated_bytes=allocated_bytes,
                headroom_bytes=headroom,limit_bytes=limit,caller_ceiling_bytes=ceiling_bytes,
                policy='max(1GiB,ceil(free/5)) reserve; ignore reusable allocator cache; floor limit to MiB',
                disposition='complete-workload-preflight-required')

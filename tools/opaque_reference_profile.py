"""Explicit offline checker limits; never production rendering configuration."""
from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class ReferenceProfile:
    name:str
    cells:int
    depth:int
    reference_seconds:int
    worker_seconds:int
    watchdog_seconds:int
    host_bytes:int=1073741824
    threads:int=1
    def record(self):return asdict(self)

REMAINING33=ReferenceProfile('remaining33-offline-two-hour-v1',8192,14,7200,7230,7260)
REMAINING28=ReferenceProfile('remaining28-offline-two-hour-v1',8192,14,7200,7230,7260)
STANDARD=ReferenceProfile('standard',256,10,90,120,180)
STREAMED_TWO_HOUR=ReferenceProfile('streamed22-offline-two-hour-v1',8192,14,7200,7230,7260)
STREAMED_HOUR=ReferenceProfile('streamed22-offline-hour-v1',4096,14,3600,3630,3660)
STREAMED_EXTENDED=ReferenceProfile('streamed22-offline-extended-v1',4096,14,900,930,960)
STREAMED=ReferenceProfile('streamed22-offline-v1',256,10,90,120,180)
EXTENDED=ReferenceProfile('original13-offline-extended-v1',4096,14,900,930,960)

def profile(name='standard'):
    if type(name)!=str:raise ValueError('explicit known reference profile required')
    for p in (STANDARD,EXTENDED,STREAMED,STREAMED_EXTENDED,STREAMED_HOUR,STREAMED_TWO_HOUR,REMAINING33,REMAINING28):
        if name==p.name:return p
    raise ValueError('unknown reference profile')

def worker_profile(mode,name='standard'):
    p=profile(name)
    if (mode=='remaining-piece') != (p in (REMAINING33,REMAINING28)):raise ValueError('remaining profile requires explicit remaining-piece worker')
    if (mode=='streamed-two-hour-piece') != (p is STREAMED_TWO_HOUR):raise ValueError('two-hour profile requires explicit streamed-two-hour-piece worker')
    if (mode=='streamed-hour-piece') != (p is STREAMED_HOUR):raise ValueError('one-hour profile requires explicit streamed-hour-piece worker')
    if (mode=='streamed-extended-piece') != (p is STREAMED_EXTENDED):raise ValueError('extended streamed profile requires explicit streamed-extended-piece worker')
    if (mode=='streamed-piece') != (p is STREAMED):raise ValueError('streamed profile requires explicit streamed-piece worker')
    if (mode in ('offline-preflight','offline-late-preflight','offline-tail-preflight','offline-centered-tail','offline-value-tail','offline-direction-tail','extended-campaign-piece')) != (p is EXTENDED):
        raise ValueError('extended reference profile is restricted to offline preflight modes')
    return p


def stronger_target(value):
    """An explicit private target can only strengthen the original5e-13 gate."""
    from decimal import Decimal,InvalidOperation
    if not isinstance(value,str):raise ValueError('explicit stronger reference target required')
    try:q=Decimal(value)
    except InvalidOperation as e:raise ValueError('finite positive stronger reference target required') from e
    if not q.is_finite() or not 0<q<=Decimal('5e-13'):raise ValueError('stronger reference target must be positive and at most5e-13')
    return value

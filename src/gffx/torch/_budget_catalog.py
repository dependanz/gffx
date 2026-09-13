"""Shipped finite-profile identity, not user-provided acceptance metadata.

No file access at import or query time. See docs/PRINCIPLED_BUDGET_V1.md for
physical evidence, source hashes and limitations. No latest-ID aliases.
"""
import hashlib
from importlib.resources import files

RESOURCE_ID = 'ggx-smith-loss-cpu-f64-v1'
PROFILE_ID = 'public-budget-v0.1'
PAYLOAD_SHA256 = 'eb9393c7afd6e1a4964019643ee827dff72696896bd05786e8a2ec908ca538e5'
ENVELOPE_SHA256 = '5e095d1aa417dcba460f7cb2a5f72b1cf8c857d793cba1da1b6e77b9fc769cb7'
RECORD_SHA256 = 'df424fcb92630ccd2bc70f8cf26997bcc3b0929ad1d56d3ca22e6ad1f4b98131'
CUDA_RESOURCE_ID = 'ggx-smith-loss-cuda-f64-v1'
CUDA_PROFILE_ID = 'public-budget-cuda-v0.1'
CUDA_RECORD_SHA256 = 'e34977b2cc9641664210f0307a881fcad996e4d8e27b9c28824b9f761fd8d96e'


def acceptance(value, *, device, dtype):
    """Called only after full envelope validation; evidence_id is not authority."""
    if (device == 'cpu' and dtype == 'float64'
            and value['format_version'] == 1
            and value['model'] == 'ggx-correlated-smith-schlick-reciprocal-matte-v1'
            and value['representation'] == 'bounded-cubic-loss-v1'
            and value['dtype'] == 'float64'
            and value['domain'] == {'roughness': [.001, 1.], 'cosine': [0, 1], 'f0': [0, 1]}
            and value['generator'] == {
                'source_sha256': '62df9a500bca26e9eeb79fef05275e42567de45eef301a369ba7d6fa838839fd',
                'policy': 'conditional-allocation-v1'}
            and value['payload_sha256'] == PAYLOAD_SHA256):
        return RESOURCE_ID, PROFILE_ID, RECORD_SHA256
    return None


def cuda_acceptance(value, *, device, dtype):
    """Named-bundle admission only, after full validation; never a cast grant."""
    if (device == 'cuda' and dtype == 'float64'
            and acceptance(value, device='cpu', dtype='float64') is not None):
        return CUDA_RESOURCE_ID, CUDA_PROFILE_ID, CUDA_RECORD_SHA256
    return None


def read_bundle(resource_id, *, byte_limit):
    if resource_id not in (RESOURCE_ID, CUDA_RESOURCE_ID):
        raise ValueError('no admitted bundled resource for this exact resource_id')
    # Device profiles share the exact numerical payload, not acceptance metadata.
    path = files('gffx').joinpath('data', 'principled', RESOURCE_ID+'.json')
    with path.open('rb') as stream:
        blob = stream.read(byte_limit+1)
    if len(blob) > byte_limit:
        raise ValueError('bundled resource byte limit')
    if hashlib.sha256(blob).hexdigest() != ENVELOPE_SHA256:
        raise ValueError('bundled resource file hash mismatch')
    return blob

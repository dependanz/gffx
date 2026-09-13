"""Publish both continued images with separate original/additional timing."""
import argparse
from pathlib import Path

from tools.environment_continuation import PARENT_HASH, replay
from tools.environment_full_images import load_bundle, MANIFEST, MANIFEST_HASH
from .environment_full_demo import publish


def projection(result):
    parent,record=result['parent'],result['record']
    return dict(sha256=result['sha256'],images=result['images'],cohort=dict(
        status=record['status'],elapsed_seconds=record['cumulative_elapsed_seconds'],
        images={'classic':parent['images']['classic'],'principled':record['images']['principled']},
        tiles={'classic':parent['tiles']['classic'],'principled':parent['tiles']['principled']+record['tiles']},
        segments=dict(original_sha256=PARENT_HASH,original_status=parent['status'],
            original_elapsed_seconds=record['original_elapsed_seconds'],
            additional_elapsed_seconds=record['additional_elapsed_seconds'],
            additional_budget_seconds=900,cumulative_elapsed_seconds=record['cumulative_elapsed_seconds'],
            reused_tiles=record['reused_tiles'],new_tiles=len(record['tiles']))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--address',required=True);p.add_argument('--run',required=True)
    a=p.parse_args();bundle=load_bundle(a.corpus,path=MANIFEST,sha256=MANIFEST_HASH)
    result=replay(a.root,a.sha256,a.parent,bundle['buffers'])
    publish(a.address,a.run,projection(result))


if __name__=='__main__':main()

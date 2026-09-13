"""Finite replay of pinned environment evidence, including unresolved attempts."""
import argparse
import hashlib
import json
from pathlib import Path

HASHES = {
    'classic-0':'a4597235056db6b00ed90c9c490f0b86059227102b6b71b065a07ba4d36a240f',
    'split-principled-0':'992f98ce4c02988cb22501ca113f63a276ac5a89b6d75c2f60ab958214bc1358',
    'square-principled-0':'8a85b82cbc80963f5fa821f4fe08f51f8eac16c3ade4a069fc4b1b9798a0ea9d',
    'separated-classic-0':'f674c286e812d667f363f0ca8ceb3e91a42c4508b75e5f1907677e4f81224c1a',
    'separated-classic-1':'f7e65ab52f99accb44bac0349a1e8b94c5dbc420c5efc06971a3ff4266ad180f',
    'separated-classic-2':'945b821abec89094d413f8d02d21fdfd05b7210c6af51bb9425094c0874d8d4c',
    'separated-classic-3':'4d34aad393e20309134627a19b44c40b44338f6db24ef571045f3eecc2593428',
    'separated-classic-4':'b9360024b841e1edbb7ae75b6c4282d3081e5b5db95a6983c171cf4a027f2553',
    'separated-classic-5':'2fac43bff52b677b40342e9a0f5100ed09f140e6dba6cc356533a1eee235f32e',
    'separated-classic-6':'2a93191b78ee9b5b5af8273e57f3b9c7ba3ef7883f1eba6bb43f18368de2387b',
    'separated-classic-7':'32557f44ba1d6886a3a3aca31f2fb01a431844b989a2b5af9de859d86776bb4b',
    'separated-principled-0':'8a633eca7d278e037e9e4c8bce232455681c30a5d0f1e84ddfc97a6df5b94474',
    'separated-principled-1':'fc8cc2b56b84be884b25d3f41c194412662c18b92aea214e9ae9debdbb5e9fea',
    'separated-principled-2':'fca9fda790ad83253c76403901358f309357c473429bc5e254a6b65a36188464',
    'separated-principled-3':'99b05bac7666ab41b03e3e37fe6d252f2829670f3a1e7b4a04fcfeda38a1668a',
    'separated-principled-4':'9fa42dc139b75ab06f2e58890b05057931af104496887c6a54899cec2b10da8a',
    'separated-principled-5':'7db60131108373078a598cc0e3febea0473684be58fafbf785134184be4cc658',
    'separated-principled-6':'fcc2c61ee4ae28336740935c06e267ad2f150a960a3cedd435dcf30e4c10d3d8',
    'separated-principled-7':'f4fc8091dfed580b2390cb48d8323520869169d46bed935bba86c69172c9b27c',
    'public-principled-0':'1b35d8b12c2cf3d6d194199447001bc8195dd34b860e8d12e3399cbc68d857e0',
}


def read(root,name):
    if name not in HASHES:raise ValueError('unknown measured report')
    with (Path(root)/(name+'.json')).open('rb') as handle:blob=handle.read(262145)
    if len(blob)>262144 or hashlib.sha256(blob).hexdigest()!=HASHES[name]:
        raise ValueError('environment report identity mismatch')
    return json.loads(blob)


def provenance():
    return {'artifact_sha256':'6ab5ccb2335281b8c05e6d1466c791cc94dfb56fbd425b77af11546516c56285'}


def publish(root,address,run):
    from gffx.dashboard import connect
    from .reporter import Reporter
    data={name:read(root,name) for name in HASHES}
    reporter=Reporter(address,run)
    def send(event,**values):
        if not reporter.send(event,**values):raise RuntimeError(reporter.error)
    send('start',kind='demo',provenance=provenance())
    artifacts=[]
    with connect(address,run=run,wait_acks=True) as dash:
        dash.text('environment-images/read-me',
            'Environment IMAGE ACCEPTANCE IS NOT COMPLETE. All sixteen independent reference points now converge. '
            'The full public principled material matches the most-grazing reference, but costs 110.205 seconds '
            'for that one point at its required quadrature order. Full-image execution needs a bounded CPU '
            'integration/resource solution; no expensive full image was launched. These plots replay recorded '
            'measurements, not a running renderer. Failed uniform, angular-only and square-coordinate reference '
            'attempts remain visible. The successful rule integrates the reciprocal matte term exactly and '
            'numerically integrates the glossy term; it is not plain Lambert or a general SH shortcut.',step=0)
        artifacts.append(dict(kind='text',path='environment-images/read-me',label='Read first: image gate remains open'))
        for name in ('classic-0','split-principled-0','square-principled-0','separated-principled-0'):
            path='environment-images/convergence/'+name
            for level,row in enumerate(data[name]['reference']['history']):
                dash.scalar(path+'/partition-disagreement',row['offset_change'][0],step=level)
                dash.scalar(path+'/threshold',row['threshold'][0],step=level)
                if row['change'] is not None:dash.scalar(path+'/refinement-change',row['change'][0],step=level)
            artifacts.append(dict(kind='plot',path=path,label=name+' — convergence errors, red channel'))
        for model in ('classic','principled'):
            path='environment-images/reference/'+model
            for index in range(8):
                row=data[f'separated-{model}-{index}']
                for channel,value in zip(('red','green','blue'),row['reference']['radiance']):
                    dash.scalar(path+'/'+channel,value,step=index)
            artifacts.append(dict(kind='plot',path=path,label=model+' — reference light by view rank (grazing to frontal)'))
        summary={name:dict(sha256=HASHES[name],status=row['status'],point=row.get('point'),
                          supervisor=row['supervisor'],reference=row.get('reference'),
                          result=row.get('result'),errors=row.get('absolute_errors')) for name,row in data.items()}
        dash.record('environment-images/evidence',summary,step=0)
        artifacts.append(dict(kind='record',path='environment-images/evidence',label='Exact histories, costs and provenance'))
    send('demo',page='shading/environment-image-progress',title='Environment lighting: references pass; image execution remains open',artifacts=artifacts)
    send('finish',exit_code=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--address',required=True);parser.add_argument('--run',required=True)
    args=parser.parse_args();publish(args.root,args.address,args.run)
    print('Published finite environment progress; no numerical work repeated.')


if __name__=='__main__':main()

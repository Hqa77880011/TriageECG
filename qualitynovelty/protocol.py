"""Common label-schema merge and test-only SNR panels under a frozen source policy."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .perturb import corrupt


def merge(sources,output):
    arrays=[dict(np.load(source,allow_pickle=False)) for source in sources]
    for data in arrays[1:]:
        for key in ['fs','class_names','lead_names']:
            if not np.array_equal(arrays[0][key],data[key]):
                raise ValueError('Cross-corpus '+key+' differs; declare a common physical and semantic schema')
    rows=['x','y','patient','split','defect','readable','semantic_id','record','canonical_record']
    result={key:np.concatenate([data[key] for data in arrays],axis=0) for key in rows}
    for key in ['fs','class_names','lead_names']: result[key]=arrays[0][key]
    ids=result['canonical_record']
    _,keep=np.unique(ids,return_index=True)
    for identity in np.unique(ids):
        selected=ids==identity
        if len(np.unique(result['split'][selected]))>1 or len(np.unique(result['y'][selected]))>1:
            raise ValueError('Redistributed ECG copy has conflicting role or label: '+identity)
    for key in rows: result[key]=result[key][np.sort(keep)]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**result)
    return result


def stress(config,source,output,snr_values=(24,18,12,6,0)):
    from .__main__ import evaluate,run
    source=Path(source).resolve(); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    original=dict(np.load(source/'data.npz'))
    test=original['split']=='test'
    rows=[]
    for snr in snr_values:
        directory=output/('snr_'+str(snr))
        directory.mkdir(exist_ok=True)
        for filename in ['model.pt','novelty.npz','baseline.pt','baseline_calibration.npz','sqi.joblib','policy.json']:
            if (source/filename).exists(): shutil.copyfile(source/filename,directory/filename)
        torch.manual_seed(config['seed'])
        data=dict(original)
        value=torch.tensor(original['x'][test])
        damaged=corrupt(value,snr,0.)
        measured=10*torch.log10(value.square().mean((1,2))/(damaged-value).square().mean((1,2)))
        data['x']=original['x'].copy(); data['x'][test]=damaged.numpy()
        data['defect']=original['defect'].copy(); data['defect'][test]=1
        # Clinical readable labels remain the independent annotation; no noise-to-readability assertion.
        np.savez_compressed(directory/'data.npz',**data)
        run(config,directory); evaluate(config,directory)
        result=pd.read_csv(directory/'metrics.csv').iloc[0].to_dict()
        rows.append(dict(result,snr_db=snr,measured_mean_snr_db=float(measured.mean()),
                         policy='frozen_source',readability='independent_annotation'))
    pd.DataFrame(rows).to_csv(output/'snr_metrics.csv',index=False)


def coverage(config,source,output,targets=(.7,.8,.9)):
    from .__main__ import calibrate,evaluate,run
    source=Path(source); output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for target in targets:
        directory=output/str(target); directory.mkdir(exist_ok=True)
        for filename in ['model.pt','baseline.pt','sqi.joblib','data.npz']:
            if (source/filename).exists(): shutil.copyfile(source/filename,directory/filename)
        setting=dict(config,coverage=target)
        calibrate(setting,directory)
        policy=json.loads((directory/'policy.json').read_text())
        if not policy['feasible']:
            rows.append(dict(target_coverage=target,feasible=False)); continue
        run(setting,directory); evaluate(setting,directory)
        rows.append(dict(pd.read_csv(directory/'metrics.csv').iloc[0],target_coverage=target,feasible=True))
    pd.DataFrame(rows).to_csv(output/'coverage.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('merge'); p.add_argument('--inputs',nargs='+',required=True); p.add_argument('--output',required=True)
    p=sub.add_parser('stress'); p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--snr',nargs='+',type=float,default=[24,18,12,6,0]); p.add_argument('--config',default='config.json')
    p=sub.add_parser('coverage'); p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--targets',nargs='+',type=float,default=[.7,.8,.9]); p.add_argument('--config',default='config.json')
    args=parser.parse_args()
    if args.command=='merge': merge(args.inputs,args.output)
    elif args.command=='stress': stress(json.loads(Path(args.config).read_text()),args.source,args.output,args.snr)
    else: coverage(json.loads(Path(args.config).read_text()),args.source,args.output,args.targets)


if __name__=='__main__': main()

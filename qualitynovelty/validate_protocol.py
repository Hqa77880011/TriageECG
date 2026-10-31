"""Native waveform fixtures and actual comparator training with source-frozen policies."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

from .__main__ import calibrate,evaluate,prepare,run,train
from .baselines import BASELINES,ResNet18,fit_openmax,load,logits,openmax,sqi
from .convert import LEADS,convert
from .protocol import coverage,merge,stress


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/protocol-check')
    args=parser.parse_args(); out=Path(args.output)
    from .validate import main as core_main
    sys.argv=['core','--output',str(out)]; core_main()
    c=json.loads(Path('config.json').read_text())
    c.update(classes=2,width=8,epochs=1,batch_size=8,threshold_points=4,pretrain_epochs=1,
        ensemble_members=5,coverage_tolerance=.21,retention_min_support=1)
    reports=[]
    for variant in BASELINES:
        directory=out/variant; directory.mkdir(exist_ok=True)
        settings=dict(c,variant=variant)
        # This tiny source problem needs multiple updates before EVT has correct-class support.
        if variant=='openmax': settings['epochs']=8
        prepare(settings,directory,out/'input.npz')
        train(settings,directory)
        calibrate(settings,directory)
        assert json.loads((directory/'policy.json').read_text())['feasible']
        run(settings,directory); evaluate(settings,directory)
        p=np.load(directory/'predictions.npz')['probability'].copy()
        run(settings,directory)
        assert np.array_equal(p,np.load(directory/'predictions.npz')['probability'])
        states=torch.load(directory/'baseline.pt',weights_only=True)
        if variant!='sqi-svm':
            torch.manual_seed(c['seed'])
            initial=ResNet18(1,2,technical=variant=='unconstrained').state_dict()
            assert not torch.equal(initial['classifier.weight'],states['states'][0]['classifier.weight'])
            models=load(settings,directory)
            data=np.load(directory/'data.npz')
            a=logits(models[0],data['x'][:2]); b=logits(load(settings,directory)[0],data['x'][:2])
            assert all(np.array_equal(x,y) for x,y in zip(a,b))
        if variant=='temperature': assert float(np.load(directory/'baseline_calibration.npz')['temperature'])>0
        if variant=='ensemble': assert len(states['states'])==5
        if variant=='ssl': assert any(row['stage']=='simclr' for row in json.loads((directory/'learning.json').read_text()))
        reports.append(variant)
    # Hand EVT distances, genuine Weibull tail fitting and probability conservation.
    activation=np.array([[5.,1.],[4.5,1.],[6.,1.],[1.,5.],[1.,4.5],[1.,6.]])
    mean,parameters=fit_openmax(activation,np.repeat([0,1],3),2)
    known,unknown=openmax(np.array([[5.,1.],[50.,40.]]),mean,parameters)
    assert np.allclose(known.sum(1)+unknown,1.) and unknown[1]>unknown[0]
    assert np.isfinite(sqi(np.zeros((2,12,200)))).all()
    # Native PTB-XL SCP rhythm metadata and challenge physical WFDB files.
    native=out/'native'; native.mkdir(exist_ok=True)
    mapping=dict(class_names=['sinus','atrial_fibrillation'],known={'SR':0,'AFIB':1},
        development_unknown=['SVARR'],test_unknown=['AFLT'],
        unknown_semantics={'SVARR':'supraventricular_arrhythmia','AFLT':'atrial_flutter'})
    rows=[]; annotations=[]
    index=0
    for fold in [1,8,9,10]:
        for code in ['SR','AFIB',('AFLT' if fold==10 else 'SVARR')]:
            index+=1; name=f'ecg{index}'
            t=np.arange(200)/100
            signal=np.tile(np.sin(2*np.pi*(1+index%2)*t)[:,None],(1,12))
            wfdb.wrsamp(name,fs=100,units=['mV']*12,sig_name=LEADS,p_signal=signal,fmt=['16']*12,write_dir=str(native))
            rows.append(dict(ecg_id=index,patient_id=index,strat_fold=fold,filename_lr=name,filename_hr=name,scp_codes=str({code:100})))
            annotations.append(dict(record=str(index),defect=int(index%4==0),readable=1))
    pd.DataFrame(rows).to_csv(native/'ptbxl_database.csv',index=False)
    pd.DataFrame(dict(code=['SR','AFIB','SVARR','AFLT'],rhythm=[1]*4,diagnostic=[0]*4)).set_index('code').to_csv(native/'scp_statements.csv')
    pd.DataFrame(annotations).to_csv(native/'annotations.csv',index=False)
    converted=convert('ptbxl',native,mapping,native/'annotations.csv',native/'ptbxl.npz',100,2)
    assert converted['x'].shape==(11,12,200)
    external=native/'challenge'; external.mkdir(exist_ok=True)
    wfdb.wrsamp('A0001',fs=100,units=['uV']*12,sig_name=LEADS,p_signal=signal*1000,fmt=['16']*12,
        comments=['Dx: 164889003'],write_dir=str(external))
    pd.DataFrame([dict(record='A0001',patient='external:1',split='test',defect=0,readable=1)]).to_csv(external/'annotations.csv',index=False)
    challenge_map=dict(mapping,known={'426783006':0,'164889003':1})
    result=convert('challenge',external,challenge_map,external/'annotations.csv',native/'challenge.npz',100,2)
    assert np.allclose(result['x'][0],signal.T,atol=.001)
    combined=merge([native/'ptbxl.npz',native/'challenge.npz'],native/'combined.npz')
    assert len(combined['y'])==12
    native_run=native/'end-to-end'; native_run.mkdir(exist_ok=True)
    native_settings=dict(c,variant='full')
    prepare(native_settings,native_run,native/'combined.npz')
    train(native_settings,native_run); calibrate(native_settings,native_run)
    assert json.loads((native_run/'policy.json').read_text())['feasible']
    run(native_settings,native_run); evaluate(native_settings,native_run)
    # Fixed-source stress must never refit model/novelty/policy or infer clinical readability.
    stress(dict(c,variant='full'),out,out/'stress',[24,0])
    report=pd.read_csv(out/'stress/snr_metrics.csv')
    assert np.allclose(report.snr_db,report.measured_mean_snr_db,atol=1e-4)
    assert (out/'policy.json').read_bytes()==(out/'stress/snr_24/policy.json').read_bytes()
    # No development semantic outlier is a supported, explicit source-only setting.
    no_ood=out/'no-development-outliers'; no_ood.mkdir(exist_ok=True)
    data=dict(np.load(out/'data.npz'))
    keep=~(np.isin(data['split'],['dev','cal'])&(data['y']<0))
    for key in ['x','y','patient','split','defect','readable']: data[key]=data[key][keep]
    np.savez_compressed(no_ood/'data.npz',**data); shutil.copyfile(out/'model.pt',no_ood/'model.pt')
    calibrate(dict(c,variant='full'),no_ood)
    assert json.loads((no_ood/'policy.json').read_text())['feasible']
    coverage(dict(c,variant='full'),out,out/'coverage',[.7,.9])
    assert len(pd.read_csv(out/'coverage/coverage.csv'))==2
    from .__main__ import select_policy
    policy=select_policy(np.tile([.9,.1],(6,1)),np.zeros(6),np.zeros(6),np.array([0]*5+[1]),
        np.ones(6),dict(c,variant='full',retention_min_support=5,coverage=1.,coverage_tolerance=0.))
    assert policy['feasible'] and policy['retention_classes']==[0]
    broken=dict(np.load(out/'input.npz'))
    broken['semantic_id']=np.where(broken['y']<0,'same_held_out_rhythm','known')
    np.savez(out/'overlapping_unknowns.npz',**broken)
    try:
        prepare(c,out,out/'overlapping_unknowns.npz')
    except ValueError as error:
        assert 'unknown rhythms overlap' in str(error)
    else: raise AssertionError('Development/final semantic overlap was accepted')
    report=dict(synthetic_function_check=True,baselines=reports,native_wfdb=True,
        explicit_rhythm_mapping=True,source_only_novelty=True,snr_fixed_policy=True,openmax_weibull=True)
    (out/'protocol_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__': main()

"""PTB-XL rhythm codes and native challenge WFDB using explicit label/annotation tables."""

import argparse
import ast
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']


def rhythm(codes,mapping,role):
    known=mapping['known']; dev=set(mapping['development_unknown']); final=set(mapping['test_unknown'])
    if (set(known)&(dev|final)) or dev&final:
        raise ValueError('Known/development-unknown/test-unknown codes must be disjoint')
    semantics=mapping['unknown_semantics']
    if not (dev|final)<=set(semantics): raise ValueError('Every held-out code needs a canonical unknown rhythm name')
    if {semantics[code] for code in dev}&{semantics[code] for code in final}:
        raise ValueError('Development and final unknowns overlap in rhythm semantics')
    selected=set(codes)&(set(known)|dev|final)
    labels={known[code] for code in selected if code in known}
    unknown=selected&(dev|final)
    if (labels and unknown) or len(labels)>1 or len(unknown)>1:
        return None,'ambiguous_rhythm'
    if labels: return (labels.pop(),'known'),'included'
    if unknown:
        code=next(iter(unknown))
        if code in dev and role in ['dev','cal']: return (-1,semantics[code]),'included'
        if code in final and role=='test': return (-1,semantics[code]),'included'
        return None,'unknown_excluded_from_this_role'
    return None,'no_mapped_rhythm'


def waveform(path,fs=100.,seconds=10.):
    import wfdb
    record=wfdb.rdrecord(str(path))
    if set(LEADS)-set(record.sig_name):
        raise ValueError('Native record lacks the declared twelve-lead schema: '+str(path))
    signal=record.p_signal[:,[record.sig_name.index(lead) for lead in LEADS]].T
    # WFDB returns physical values. Convert their declared unit to mV before pooling sources.
    for index,lead in enumerate(LEADS):
        unit=record.units[record.sig_name.index(lead)]
        if unit.lower() in ['uv','μv','µv']: signal[index]/=1000
        elif unit.lower()=='v': signal[index]*=1000
        elif unit.lower()!='mv': raise ValueError('Declare an ECG unit conversion for '+unit)
    ratio=Fraction(float(fs)/record.fs).limit_denominator(10000)
    signal=resample_poly(signal,ratio.numerator,ratio.denominator,axis=-1)
    length=round(fs*seconds)
    if signal.shape[-1]<length:
        raise ValueError('Record is shorter than the declared window; no fabricated padding: '+str(path))
    return signal[:,:length].astype(np.float32)


def convert(kind,root,mapping,annotations,output,fs=100.,seconds=10.):
    root=Path(root)
    annotation=pd.read_csv(annotations,dtype={'record':str,'patient':str}).set_index('record',verify_integrity=True)
    required={'defect','readable'}
    if not required<=set(annotation): raise ValueError('Annotation CSV needs record,defect,readable (separate binary labels)')
    if not annotation[['defect','readable']].isin([0,1]).all().all():
        raise ValueError('Technical and readability annotations must be explicit binary values')
    names=mapping['class_names']
    if set(mapping['known'].values())!=set(range(len(names))):
        raise ValueError('Known code map must cover every declared class name contiguously')
    if kind=='ptbxl':
        frame=pd.read_csv(root/'ptbxl_database.csv',dtype={'patient_id':str})
        statements=pd.read_csv(root/'scp_statements.csv',index_col=0)
        codes=set(mapping['known'])|set(mapping['development_unknown'])|set(mapping['test_unknown'])
        if not codes<=set(statements[statements.rhythm==1].index):
            raise ValueError('PTB-XL mapping must use original SCP rhythm codes, not diagnostic superclasses')
        entries=[]
        for row in frame.itertuples():
            role={8:'dev',9:'cal',10:'test'}.get(int(row.strat_fold),'train')
            entries.append((str(row.ecg_id),'ptbxl:'+str(row.patient_id),role,
                list(ast.literal_eval(row.scp_codes)),root/(row.filename_lr if fs<=100 else row.filename_hr)))
    else:
        if not {'patient','split'}<=set(annotation):
            raise ValueError('Challenge identities require globally consistent patient and split columns')
        if not (annotation['split']=='test').all():
            raise ValueError('External challenge conversion is test-only; source calibration remains frozen')
        entries=[]
        for header in sorted(root.rglob('*.hea')):
            identity=header.stem
            if identity not in annotation.index: continue
            lines=header.read_text().splitlines()
            diagnosis=next((line.split(':',1)[1].strip() for line in lines
                            if line.startswith('#') and line[1:].strip().startswith('Dx:')),None)
            if diagnosis is None: raise ValueError('Missing native #Dx in '+str(header))
            entries.append((identity,str(annotation.loc[identity,'patient']),'test',diagnosis.replace(' ','').split(','),header.with_suffix('')))
    rows=[]; audit=[]
    for identity,patient,role,codes,path in entries:
        target,reason=rhythm(codes,mapping,role)
        if target is None:
            audit.append(dict(record=identity,reason=reason)); continue
        if identity not in annotation.index:
            raise ValueError('Missing independent technical/readability annotation for record '+identity)
        labels=annotation.loc[identity]
        if 'patient' in annotation and pd.notna(labels.patient): patient=str(labels.patient)
        y,semantic=target
        canonical=str(labels.get('canonical_record',kind+':'+identity))
        rows.append(dict(x=waveform(path,fs,seconds),y=y,patient=patient,split=role,
            defect=int(labels.defect),readable=int(labels.readable),semantic_id=semantic,
            record=kind+':'+identity,canonical_record=canonical))
        audit.append(dict(record=identity,reason='included',split=role,y=y,semantic_id=semantic))
    if not rows: raise ValueError('No eligible unambiguous mapped rhythm records')
    arrays={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
    arrays.update(fs=fs,class_names=np.asarray(names),lead_names=np.asarray(LEADS))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    pd.DataFrame(audit).to_csv(output.with_suffix('.eligibility.csv'),index=False)
    return arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['ptbxl','challenge'],required=True)
    parser.add_argument('--input',required=True)
    parser.add_argument('--mapping',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fs',type=float,default=100.)
    parser.add_argument('--seconds',type=float,default=10.)
    args=parser.parse_args()
    convert(args.kind,args.input,json.loads(Path(args.mapping).read_text()),args.annotations,args.output,args.fs,args.seconds)


if __name__=='__main__': main()

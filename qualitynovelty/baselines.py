"""Actual SQI/SVM, full ResNet18, temperature, ensemble, OpenMax and SimCLR controls."""

import json

import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from scipy.stats import kurtosis, skew, weibull_min
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F

from .perturb import corrupt

BASELINES=['sqi-svm','resnet-msp','temperature','ensemble','energy','openmax','cascade','ssl','unconstrained']


class BasicBlock(nn.Module):
    def __init__(self,source,target,stride=1):
        super().__init__()
        self.body=nn.Sequential(nn.Conv1d(source,target,3,stride,padding=1,bias=False),nn.BatchNorm1d(target),nn.ReLU(),
            nn.Conv1d(target,target,3,padding=1,bias=False),nn.BatchNorm1d(target))
        self.skip=(nn.Sequential(nn.Conv1d(source,target,1,stride,bias=False),nn.BatchNorm1d(target))
                   if source!=target or stride!=1 else nn.Identity())

    def forward(self,x): return F.relu(self.body(x)+self.skip(x))


class ResNet18(nn.Module):
    def __init__(self,channels,classes,technical=False):
        super().__init__()
        levels=[nn.Conv1d(channels,64,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(3,2,1)]
        source=64
        for index,target in enumerate([64,128,256,512]):
            levels.extend([BasicBlock(source,target,1 if index==0 else 2),BasicBlock(target,target)])
            source=target
        levels.extend([nn.AdaptiveAvgPool1d(1),nn.Flatten()])
        self.encoder=nn.Sequential(*levels)
        self.classifier=nn.Linear(512,classes)
        self.technical=nn.Sequential(nn.Linear(512+3*channels,32),nn.ReLU(),nn.Linear(32,1)) if technical else None

    def forward(self,x):
        z=self.encoder(x)
        summary=torch.cat([x.std(-1),(x.abs()>4).float().mean(-1),(x==0).float().mean(-1)],1)
        q=self.technical(torch.cat([z,summary],1)).squeeze(-1) if self.technical else torch.zeros(len(x),device=x.device)
        return self.classifier(z),z,q


def sqi(x):
    """Twelve per-lead morphology, flatness and spectral-quality features."""
    spectrum=abs(np.fft.rfft(x,axis=-1))**2
    frequency=np.fft.rfftfreq(x.shape[-1])
    power=spectrum.sum(-1).clip(1e-12)
    features=[x.mean(-1),x.std(-1),np.ptp(x,axis=-1),np.median(abs(x),axis=-1),
        (x==0).mean(-1),(abs(np.diff(x,axis=-1))<1e-5).mean(-1),
        skew(x,axis=-1),kurtosis(x,axis=-1),np.mean(abs(np.diff(x,axis=-1)),axis=-1),
        spectrum[...,frequency<.01].sum(-1)/power,spectrum[...,frequency>.3].sum(-1)/power,
        np.max(spectrum,axis=-1)/power]
    return np.nan_to_num(np.stack(features,-1).reshape(len(x),-1))


def augment(x):
    scale=.9+.2*torch.rand(len(x),x.shape[1],1,device=x.device)
    return x*scale+.01*torch.randn_like(x)


def simclr(model,x,c):
    projection=nn.Sequential(nn.Linear(512,512),nn.ReLU(),nn.Linear(512,128))
    optimizer=torch.optim.Adam([*model.encoder.parameters(),*projection.parameters()],lr=c['learning_rate'])
    history=[]
    for _ in range(c.get('pretrain_epochs',50)):
        for index in torch.randperm(len(x)).split(c['batch_size']):
            if len(index)<2: continue
            value=x[index]
            z=F.normalize(torch.cat([projection(model.encoder(augment(value))),projection(model.encoder(augment(value)))],0),dim=1)
            logits=z@z.T/.1
            logits=logits.masked_fill(torch.eye(len(z),dtype=torch.bool),-1e9)
            target=(torch.arange(len(z))+len(value))%len(z)
            loss=F.cross_entropy(logits,target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            history.append(float(loss.detach()))
    return history


def logits(model,values,batch_size=32):
    model.eval()
    rows=[]
    with torch.no_grad():
        for group in torch.as_tensor(values,dtype=torch.float32).split(batch_size):
            rows.append(tuple(value.numpy() for value in model(group)))
    return tuple(np.concatenate([row[index] for row in rows]) for index in range(3))


def train(c,out):
    d=np.load(out/'data.npz')
    source=d['split']=='train'; dev=(d['split']=='dev')&(d['y']>=0)
    x=torch.tensor(d['x'][source]); y=torch.tensor(d['y'][source],dtype=torch.long)
    history=[]
    states=[]
    members=0 if c['variant']=='sqi-svm' else c.get('ensemble_members',5) if c['variant']=='ensemble' else 1
    for member in range(members):
        torch.manual_seed(c['seed']+17*member)
        model=ResNet18(x.shape[1],c['classes'],technical=c['variant']=='unconstrained')
        if c['variant']=='ssl':
            history.extend(dict(stage='simclr',loss=value) for value in simclr(model,x,c))
        optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
        best=float('inf'); selected=None
        for epoch in range(c['epochs']):
            model.train()
            for index in torch.randperm(len(x)).split(c['batch_size']):
                prediction,z,q=model(x[index])
                loss=F.cross_entropy(prediction,y[index])
                if model.technical:
                    _,_,bad=model(corrupt(x[index],c.get('training_snr_db',12.),c['lead_dropout']))
                    loss+=c['technical_weight']*(F.binary_cross_entropy_with_logits(q,torch.tensor(d['defect'][source][index],dtype=torch.float32))+
                        F.binary_cross_entropy_with_logits(bad,torch.ones_like(bad)))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            prediction,_,_=logits(model,d['x'][dev],c['batch_size'])
            score=float(-np.log(softmax(prediction,axis=1)[np.arange(dev.sum()),d['y'][dev]].clip(1e-12)).mean())
            history.append(dict(stage='supervised',member=member,epoch=epoch,source_dev_nll=score))
            if score<best:
                best=score; selected={key:value.detach().clone() for key,value in model.state_dict().items()}
        states.append(selected)
    torch.save(dict(states=states,channels=x.shape[1],classes=c['classes'],variant=c['variant']),out/'baseline.pt')
    if c['variant'] in ['sqi-svm','cascade']:
        features=sqi(d['x'])
        clean=x.numpy()
        damaged=corrupt(x,c.get('training_snr_db',12.),c['lead_dropout']).numpy()
        quality=make_pipeline(StandardScaler(),SVC(C=1.,kernel='rbf'))
        quality.fit(np.concatenate([sqi(clean),sqi(damaged)]),np.r_[d['defect'][source],np.ones(len(clean),int)])
        decision=quality.decision_function(features[dev])
        # Include separately generated source-dev defects; unknown rhythms do not define quality.
        dev_bad=corrupt(torch.tensor(d['x'][dev]),c.get('training_snr_db',12.),c['lead_dropout']).numpy()
        qcal=LogisticRegression().fit(np.r_[decision,quality.decision_function(sqi(dev_bad))][:,None],
                                     np.r_[d['defect'][dev],np.ones(dev.sum(),int)])
        saved=dict(quality=quality,quality_calibrator=qcal)
        if c['variant']=='sqi-svm':
            classifier=make_pipeline(StandardScaler(),SVC(C=1.,kernel='rbf',decision_function_shape='ovr'))
            classifier.fit(features[source],d['y'][source])
            score=classifier.decision_function(features[dev])
            if score.ndim==1: score=score[:,None]
            calibration=LogisticRegression(max_iter=2000).fit(score,d['y'][dev])
            saved.update(classifier=classifier,calibrator=calibration)
        joblib.dump(saved,out/'sqi.joblib')
    (out/'learning.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


def load(c,out):
    state=torch.load(out/'baseline.pt',weights_only=True)
    if state['variant']!=c['variant']: raise ValueError('Baseline setting differs from its fitted checkpoint')
    models=[]
    for weights in state['states']:
        model=ResNet18(state['channels'],state['classes'],technical=c['variant']=='unconstrained')
        model.load_state_dict(weights); models.append(model.eval())
    return models


def eucos(x,mean):
    euclidean=np.sqrt(((x-mean)**2).sum(-1))/200
    cosine=1-(x@mean)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(mean)+1e-12)
    return euclidean+cosine


def fit_openmax(activation,labels,classes,tail=20):
    means=[]; parameters=[]
    for k in range(classes):
        values=activation[(labels==k)&(activation.argmax(1)==k)]
        if len(values)<2:
            raise ValueError(f'OpenMax requires at least two correctly classified source examples for class {k}')
        mean=values.mean(0)
        distance=eucos(values,mean)
        tail_values=np.sort(distance)[-min(tail,len(distance)):].clip(1e-8)
        shape,location,scale=weibull_min.fit(tail_values,floc=0)
        means.append(mean); parameters.append([shape,location,scale])
    return np.asarray(means),np.asarray(parameters)


def openmax(activation,means,parameters):
    adjusted=activation.copy()
    count=min(10,activation.shape[1])
    ranking=np.argsort(-activation,axis=1)
    rank_weights=np.zeros_like(activation)
    np.put_along_axis(rank_weights,ranking[:,:count],np.broadcast_to(np.arange(count,0,-1)/count,(len(activation),count)),axis=1)
    for k,(mean,parameter) in enumerate(zip(means,parameters)):
        adjusted[:,k]*=1-rank_weights[:,k]*weibull_min.cdf(eucos(activation,mean),*parameter)
    unknown=(activation-adjusted).sum(1)
    probability=softmax(np.c_[adjusted,unknown],axis=1)
    return probability[:,:-1],probability[:,-1]


def raw_scores(c,out,models,x,fit=None):
    if c['variant']=='sqi-svm':
        saved=joblib.load(out/'sqi.joblib'); feature=sqi(x)
        decision=saved['classifier'].decision_function(feature)
        if decision.ndim==1: decision=decision[:,None]
        p=saved['calibrator'].predict_proba(decision)
        q=saved['quality_calibrator'].predict_proba(saved['quality'].decision_function(feature)[:,None])[:,1]
        return p,feature,q,1-p.max(1),np.log(p.clip(1e-12))
    rows=[logits(model,x,c['batch_size']) for model in models]
    activation=np.mean([row[0] for row in rows],axis=0)
    p=softmax(activation/(float(fit['temperature']) if fit is not None and 'temperature' in fit else 1.),axis=1)
    z=np.mean([row[1] for row in rows],axis=0)
    q=np.zeros(len(x)); u=1-p.max(1)
    if c['variant']=='ensemble':
        p=np.mean([softmax(row[0],axis=1) for row in rows],axis=0)
        u=-(p*np.log(p.clip(1e-12))).sum(1)/np.log(c['classes'])
    if c['variant']=='energy': u=-logsumexp(activation,axis=1)
    if c['variant']=='openmax' and fit is not None:
        p,u=openmax(activation,fit['openmax_means'],fit['openmax_parameters'])
    if c['variant']=='cascade':
        saved=joblib.load(out/'sqi.joblib')
        feature=sqi(x)
        q=saved['quality_calibrator'].predict_proba(saved['quality'].decision_function(feature)[:,None])[:,1]
    if c['variant']=='unconstrained':
        q=1/(1+np.exp(-np.clip(rows[0][2],-700,700)))
    return p,z,q,u,activation


def calibrate(c,out):
    from .__main__ import novelty_features,select_policy
    d=np.load(out/'data.npz')
    models=load(c,out)
    source=d['split']=='train'; dev=(d['split']=='dev')&(d['readable']==1); cal=d['split']=='cal'
    fit={}
    _,z,_,_,activation=raw_scores(c,out,models,d['x'][source])
    if c['variant']=='openmax':
        fit['openmax_means'],fit['openmax_parameters']=fit_openmax(activation,d['y'][source],c['classes'])
    if c['variant']=='unconstrained':
        fit['centers']=np.stack([z[d['y'][source]==k].mean(0) for k in range(c['classes'])])
        fit['scales']=np.stack([z[d['y'][source]==k].var(0).clip(1e-5) for k in range(c['classes'])])
    p,z,q,u,activation=raw_scores(c,out,models,d['x'][dev],fit)
    if c['variant']=='temperature':
        known=d['y'][dev]>=0
        def objective(logt):
            scores=activation[known]/np.exp(logt)
            return np.mean(logsumexp(scores,axis=1)-scores[np.arange(known.sum()),d['y'][dev][known]])
        fit['temperature']=np.exp(minimize_scalar(objective,bounds=(-4.,4.),method='bounded').x)
    if c['variant']=='energy':
        fit['energy_mean']=np.mean(u); fit['energy_scale']=max(np.std(u),1e-6)
    if c['variant']=='unconstrained':
        feature=novelty_features(z,q,-logsumexp(activation,axis=1),fit['centers'],fit['scales'],c)
        scaler=StandardScaler().fit(feature)
        semantic=d['y'][dev]<0
        if len(np.unique(semantic))==2:
            learner=LogisticRegression(max_iter=2000).fit(scaler.transform(feature),semantic)
            fit['coefficient'],fit['intercept']=learner.coef_[0],learner.intercept_[0]
        else:
            coefficient=np.zeros(feature.shape[1]); coefficient[0]=coefficient[c['classes']+1]=.5
            fit['coefficient'],fit['intercept']=coefficient,0.
        fit['mean'],fit['scale']=scaler.mean_,scaler.scale_
    p,z,q,u,activation=raw_scores(c,out,models,d['x'][cal],fit)
    u=novelty(c,z,q,u,activation,fit)
    policy=select_policy(p,q,u,d['y'][cal],d['readable'][cal],c)
    (out/'policy.json').write_text(json.dumps(policy),encoding='utf-8')
    np.savez(out/'baseline_calibration.npz',**fit)


def novelty(c,z,q,u,activation,fit):
    from .__main__ import novelty_features
    if c['variant']=='energy':
        u=1/(1+np.exp(-np.clip((u-fit['energy_mean'])/fit['energy_scale'],-700,700)))
    if c['variant']=='unconstrained':
        feature=novelty_features(z,q,-logsumexp(activation,axis=1),fit['centers'],fit['scales'],c)
        u=1/(1+np.exp(-np.clip(((feature-fit['mean'])/fit['scale'])@fit['coefficient']+fit['intercept'],-700,700)))
    return u


def run(c,out):
    from .__main__ import route
    policy=json.loads((out/'policy.json').read_text())
    if not policy['feasible']: raise ValueError('No feasible source-calibration operating point')
    data=np.load(out/'data.npz'); test=data['split']=='test'
    fit=dict(np.load(out/'baseline_calibration.npz'))
    p,z,q,u,activation=raw_scores(c,out,load(c,out),data['x'][test],fit)
    u=novelty(c,z,q,u,activation,fit)
    actions=route(p,q,u,policy['thresholds'],c)
    np.savez_compressed(out/'predictions.npz',probability=p,defect=q,novelty=u,action=actions,
        y=data['y'][test],readable=data['readable'][test],technical=data['defect'][test],patient=data['patient'][test])

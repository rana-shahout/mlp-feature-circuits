"""V2-only fidelity definitions; original-neuron recovery is outside this stage."""
import numpy as np
import torch
from src.models import OriginalMLP,SparseTranscoder
from src.runtime import load_checkpoint
from ..analysis import metrics
from ..data import RULES
from ..capacity256 import signed_statistics


def loss_terms(prediction,teacher_output,z,coefficient):
    if prediction.shape!=teacher_output.shape or prediction.ndim!=2 or prediction.shape[1]!=12:raise ValueError('Expected full 12D teacher outputs')
    if z.ndim!=2 or len(z)!=len(prediction) or coefficient<0:raise ValueError('Invalid ReLU activation penalty')
    reconstruction=(prediction-teacher_output.detach()).square().sum(1).mean()
    l1=z.abs().sum(1).mean()
    return reconstruction+coefficient*l1,reconstruction,l1


def frozen_teacher(path,device):
    cp=load_checkpoint(path);expected={'W1.weight':(256,12),'W1.bias':(256,),'W2.weight':(12,256),'W2.bias':(12,)}
    if {k:tuple(v.shape) for k,v in cp['model'].items()}!=expected:raise ValueError('Approved teacher must be 12→256→12')
    if cp['config']['dimensions']['hidden']!=256 or cp['config']['seeds']['teacher']!=3105:raise ValueError('Teacher config/seed mismatch')
    m=OriginalMLP(12,256,12).to(device);m.load_state_dict(cp['model']);m.eval().requires_grad_(False)
    return m,cp


@torch.no_grad()
def infer(model,x,batch=2048,latents=False):
    outputs=[];zs=[]
    for start in range(0,len(x),batch):
        xb=x[start:start+batch]
        if isinstance(model,SparseTranscoder):
            z=model.latents(xb);y=model.decoder(z)
            if latents:zs.append(z.cpu().numpy())
        else:y=model(xb)
        if not torch.isfinite(y).all():raise RuntimeError('Nonfinite predictions')
        outputs.append(y.cpu().numpy())
    return np.concatenate(outputs),np.concatenate(zs) if zs else None


def fidelity(student,teacher,g,q):
    s,t,q=[np.asarray(v,dtype=float) for v in (student,teacher,q)]
    ss,ts=s@q,t@q;sp,tp=ss>=.5,ts>=.5;correct=(tp==g.astype(bool)).all(1)
    variance=float(((t-t.mean(0))**2).mean());mse=float(((s-t)**2).mean())
    result={'n':len(g),'full_output_mse':mse,'full_output_squared_l2_mean':12*mse,'teacher_output_variance':variance,
            'variance_normalized_mse':mse/variance if variance>0 else None,'semantic_mse':float(((ss-ts)**2).mean()),
            'exact_vector_agreement':float((sp==tp).all(1).mean()),'per_output':[]}
    for name,mask in [('teacher_correct',correct),('teacher_incorrect',~correct)]:
        result[name]={'n':int(mask.sum()),'full_output_mse':float(((s[mask]-t[mask])**2).mean()) if mask.any() else None,'exact_vector_agreement':float((sp[mask]==tp[mask]).all(1).mean()) if mask.any() else None}
    for j,rule in enumerate(RULES):
        wrong=tp[:,j]!=g[:,j].astype(bool);res=ss[:,j]-ts[:,j]
        result['per_output'].append({'rule':rule,'agreement':float((sp[:,j]==tp[:,j]).mean()),'semantic_mse':float((res**2).mean()),'signed_residual_mean':float(res.mean()),'residual_mae':float(np.abs(res).mean()),'residual_quantiles':np.quantile(res,[0,.01,.1,.5,.9,.99,1]).tolist(),'teacher_error_n':int(wrong.sum()),'teacher_errors_preserved_fraction':float((sp[wrong,j]==tp[wrong,j]).mean()) if wrong.any() else None})
    return {'teacher_fidelity':result,'teacher_task':metrics(t,g,q,.5),'student_task':metrics(s,g,q,.5)}


def activation_summary(z):
    active=z>0
    return {'mean_l0':float(active.sum(1).mean()),'mean_l0_fraction':float(active.mean()),'mean_l1_sum':float(z.sum(1).mean()),'dead_on_population':int((~active.any(0)).sum()),'per_feature_prevalence':active.mean(0).tolist(),'per_feature_mean':z.mean(0).tolist(),'per_feature_std':z.std(0).tolist(),'quantile_levels':[0,.5,.9,.99,1],'per_feature_quantiles':np.quantile(z,[0,.5,.9,.99,1],axis=0).tolist()}


def pareto(rows):
    # Nondominated on validation full-output MSE and validation mean L0.
    return [r['name'] for r in rows if not any((other['mse']<=r['mse'] and other['l0']<=r['l0']) and (other['mse']<r['mse'] or other['l0']<r['l0']) for other in rows)]


def name(width,coefficient):return f'w{width}_l{coefficient:g}'

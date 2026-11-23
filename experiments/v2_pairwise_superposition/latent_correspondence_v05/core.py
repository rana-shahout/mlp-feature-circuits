import numpy as np
import torch
from ..workflow_v04.core import write,read,sha,now,keys,inputs,corners,lock,verify,polynomial,from_polynomial
GRID=[0.,1e-8,1e-6,1e-4,1e-2,1.]
FEATURES=[173,266,84,910,633,619]
PAIRS=[(7,8),(11,12),(0,1)]
FEATURE_CASE={173:(4,(7,8)),266:(4,(7,8)),84:(6,(11,12)),910:(6,(11,12)),633:(0,(0,1)),619:(0,(0,1))}
DT=torch.float64

def tensor(x):return torch.as_tensor(x,device='cuda',dtype=DT)
def numpy(x):return x.detach().cpu().numpy() if torch.is_tensor(x) else x

def target_statistics(z):
 mu=z.mean(0);v=z.var(0,unbiased=False);positive=v[v>0];assert len(positive),'No target variance';floor=max(1e-12,1e-8*float(torch.quantile(positive,.5)));tau=torch.maximum(torch.full_like(v,1e-8),1e-3*z.square().mean(0).sqrt())
 return {'mu_z':mu,'v':v,'v_floor':floor,'eligible':v>floor,'tau':tau,'strict_fit_activity':(z>0).sum(0),'meaningful_fit_activity':(z>tau).sum(0),'fit_second_moment':z.square().mean(0)}

@torch.no_grad()
def design_fit(predictors,targets):
 """Primary fitting API: predictor H and post-ReLU Z only. No outputs/labels/derivatives."""
 assert predictors.is_cuda and targets.is_cuda
 h=predictors.double();z=targets.double();mu=h.mean(0);std=h.std(0,unbiased=False);constant=std<=1e-12*max(1.,float(std.max()));retained=~constant;zc=z-z.mean(0);x=(h[:,retained]-mu[retained])/std[retained]
 if x.shape[1]:
  l,s,vh=torch.linalg.svd(x,full_matrices=False,driver='gesvd');ltz=l.T@zc;recon=(l*s)@vh;reconstruction=float((x-recon).abs().max());cut=1e-10*s.max();rank=int((s>cut).sum())
 else:l=torch.empty((len(h),0),device='cuda',dtype=DT);s=torch.empty(0,device='cuda',dtype=DT);vh=torch.empty((0,0),device='cuda',dtype=DT);ltz=torch.empty((0,z.shape[1]),device='cuda',dtype=DT);reconstruction=0.;cut=0.;rank=0
 assert reconstruction<1e-8
 return {'mu_h':mu,'s_h':std,'constant_mask':constant,'retained':retained,'mu_z':z.mean(0),'X':x,'Zc':zc,'s':s,'Vh':vh,'Ltz':ltz,'N':len(h),'rank':rank,'rcond':1e-10,'svd_reconstruction_max':reconstruction,'cutoff':float(cut)}

@torch.no_grad()
def solve(fit,alpha):
 s=fit['s'];cut=fit['rcond']*s.max() if len(s) else 0
 if alpha==0:factor=torch.where(s>cut,1/s,torch.zeros_like(s))
 else:factor=s/(s*s+fit['N']*alpha)
 u=fit['Vh'].T@(factor[:,None]*fit['Ltz']);m=torch.zeros((u.shape[1],len(fit['mu_h'])),device='cuda',dtype=DT);m[:,fit['retained']]=u.T/fit['s_h'][fit['retained']];d=fit['mu_z']-m@fit['mu_h'];assert torch.isfinite(m).all() and torch.isfinite(d).all()
 x=fit['X'];rhs=x.T@fit['Zc']/fit['N'];res=x.T@(x@u-fit['Zc'])/fit['N']+alpha*u;relative=float(torch.linalg.vector_norm(res)/torch.linalg.vector_norm(rhs).clamp_min(1e-30));return {'M':m,'d':d,'U':u,'alpha':alpha,'normal_equation_relative_residual':relative}

def predict(model,h):return h@model['M'].T+model['d']
def choose_alpha(scores):
 best=min(s['score'] for s in scores);return max(s['alpha'] for s in scores if s['score']<=best+1e-12)
def compact_model(model,fit):return {**model,**{k:v for k,v in fit.items() if k not in ['X','Zc','Ltz','Vh']}}
def to_cpu(d):return {k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in d.items()}

def safe_ratio(num,den):return float(num/den) if den>0 else None

def feature_metrics(z,p,stats):
 z=np.asarray(z);p=np.asarray(p);err=p-z;ae=np.abs(err);v=numpy(stats['v']);tau=numpy(stats['tau']);eligible=numpy(stats['eligible']);n=len(z);rows=[]
 for m in range(z.shape[1]):
  zz,pp,ee=z[:,m],p[:,m],err[:,m];abs_e=ae[:,m];active=zz>tau[m];inactive=zz==0;near=(zz>0)&~active;pred=pp>tau[m];mse=float(np.mean(ee**2));ev=float(np.var(zz));r2=1-mse/ev if ev>1e-12*max(1,float(np.mean(zz**2))) else None;normmse=mse/v[m] if eligible[m] else None;rel=np.sqrt(np.sum(ee[active]**2)/np.sum(zz[active]**2)) if active.any() and np.sum(zz[active]**2)>0 else None
  r={'feature':m,'n':n,'mse':mse,'rmse':np.sqrt(mse),'mae':abs_e.mean(),'signed_bias':ee.mean(),'p95_abs':np.quantile(abs_e,.95),'max_abs':abs_e.max(),'NMSE_fitvar':normmse,'R2_eval':r2,'fit_eligible':bool(eligible[m]),'eval_variance':ev,'true_mean':zz.mean(),'true_sd':zz.std(),'true_p95':np.quantile(zz,.95),'true_max':zz.max(),'strict_activity':int((zz>0).sum()),'meaningful_activity':int(active.sum()),'inactive_count':int(inactive.sum()),'nearzero_count':int(near.sum()),'fit_strict_activity':int(numpy(stats['strict_fit_activity'])[m]),'fit_meaningful_activity':int(numpy(stats['meaningful_fit_activity'])[m]),'new_activity':bool(numpy(stats['strict_fit_activity'])[m]==0 and (zz>0).any()),'tau':tau[m],'active_relative_RMSE':rel,'gate_precision':safe_ratio(np.sum(pred&active),np.sum(pred)),'gate_recall':safe_ratio(np.sum(pred&active),np.sum(active)),'negative_pred_count':int((pp<0).sum()),'negative_pred_fraction':np.mean(pp<0),'negative_pred_min':min(0.,float(pp.min())),'negative_pred_abs_mean':float(np.maximum(-pp,0).mean()),'well_read':bool(normmse is not None and normmse<=.05 and rel is not None and rel<=.20 and active.sum()>=32)}
  for name,mask in [('active',active),('inactive',inactive),('nearzero',near)]:r[name+'_mse']=float(np.mean(ee[mask]**2)) if mask.any() else None;r[name+'_mae']=float(np.mean(abs_e[mask])) if mask.any() else None
  rows.append(r)
 return rows

def response_metrics(target,pred,v):
 target=np.asarray(target).reshape(-1,1024);pred=np.asarray(pred).reshape(-1,1024);delta=.1*np.sqrt(numpy(v));rows=[]
 for m in range(1024):
  t=target[:,m];e=pred[:,m]-t;nontrivial=np.abs(t)>=delta[m];nontrivial &= np.abs(t)>0;den=np.sum(t**2);den_nt=np.sum(t[nontrivial]**2);rr=float(np.sqrt(np.sum(e[nontrivial]**2)/den_nt)) if den_nt>0 else None
  rows.append({'feature':m,'n_changes':len(t),'mae':np.abs(e).mean(),'rmse':np.sqrt(np.mean(e**2)),'bias':e.mean(),'max_abs':np.abs(e).max(),'relative_RMSE':float(np.sqrt(np.sum(e**2)/den)) if den>0 else None,'nontrivial_count':int(nontrivial.sum()),'nontrivial_relative_RMSE':rr,'small_count':int((~nontrivial).sum()),'small_change_MAE':float(np.abs(e[~nontrivial]).mean()) if (~nontrivial).any() else None,'finite_response_compatible':bool(nontrivial.sum()>=32 and rr is not None and rr<=.2)})
 return rows

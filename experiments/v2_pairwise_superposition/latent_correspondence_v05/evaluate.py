import numpy as np
import torch
from .core import *
from .run import models,load_models,csvwrite,npz,predictions

def residual_stats(a,q):
 a=tensor(a);sem=a@q;perp=a-sem@q.T
 return {'full_MSE':float(a.square().mean()),'full_MAE':float(a.abs().mean()),'semantic_MSE':float(sem.square().mean()),'semantic_MAE':float(sem.abs().mean()),'out_of_Q_span_MSE':float(perp.square().mean()),'max_abs':float(a.abs().max()),'signed_coordinate_mean':numpy(a.mean(0)).tolist()}

@torch.no_grad()
def evaluate_outputs(run):
 verify(run,'readout_lock.json');t,s,p,q=models(run);maps=load_models(run);M=maps['native']['M'];d=maps['native']['d'];B=s.decoder.weight;c=s.decoder.bias;dw=t.W2.weight-B@M;db=t.W2.bias-B@d-c;rows=[];groups=[];man=read(run/'context_manifest.json');params={'relative_Frobenius_DeltaW':float(dw.norm()/t.W2.weight.norm()),'DeltaW_Frobenius':float(dw.norm()),'Delta_b_norm':float(db.norm()),'Delta_b':numpy(db).tolist()}
 for label in ['fit','selection','confirmation','legacy']:
  if not (run/(label+'_activations.npz')).exists():continue
  a=dict(np.load(run/(label+'_activations.npz')));ps=dict(np.load(run/(label+'_predictions.npz')));z=tensor(a['z']);h=tensor(a['h']);teacher_y=tensor(a['y']);tc=z@B.T+c;lookup={int(k):i for i,k in enumerate(a['keys'])};populations=[(label,np.arange(len(z)))]
  if label=='confirmation':populations=[('primary_ordinary',np.array([lookup[k] for k in man['ordinary_confirmation_keys']])),('primary_union',np.arange(len(z)))]
  if label=='legacy':populations=[('legacy_union',np.arange(len(z)))]
  for method in ['native','input','output','semantic','shuffle','mean']:
   zh=tensor(ps[method]);via=zh@B.T+c;e_tc=teacher_y-tc;e_lat=(z-zh)@B.T;e_total=teacher_y-via;torch.testing.assert_close(e_total,e_tc+e_lat,atol=1e-9,rtol=1e-8)
   if method=='native':torch.testing.assert_close(e_total,h@dw.T+db,atol=1e-9,rtol=1e-8);npz(run/(label+'_output_residuals.npz'),teacher=teacher_y,TC=tc,via=via,e_TC=e_tc,e_latent=e_lat,e_total=e_total)
   for pop,idx in populations:
    for typ,e in [('teacher_minus_TC',e_tc),('decoder_latent_error',e_lat),('teacher_minus_via',e_total)]:rows.append({'population':pop,'method':method,'residual':typ,'n':len(idx),**residual_stats(e[idx],q)})
    dz=z[idx]-zh[idx];individual=float((dz.square()*B.square().sum(0)).sum(1).mean());summed=float((dz@B.T).square().sum(1).mean());rows.append({'population':pop,'method':method,'residual':'latent_output_cancellation','n':len(idx),'sum_individual_feature_energy':individual,'energy_after_feature_sum':summed,'net_over_individual_energy':summed/individual if individual>0 else None})
   if method=='native':
    for ids in [[173,266],[84,910],[633,619]]:
     actual=z[:,ids,None]*B.T[ids][None,:,:];recovered=zh[:,ids,None]*B.T[ids][None,:,:];err=recovered-actual
     for pop,idx in populations:groups.append({'population':pop,'features':ids,'n':len(idx),'sum_individual_error_energy':float(err[idx].square().sum((1,2)).mean()),'error_energy_after_group_sum':float(err[idx].sum(1).square().sum(1).mean()),**residual_stats(err[idx].sum(1),q)})
     if label=='confirmation':npz(run/f'group_{ids[0]}_{ids[1]}.npz',actual=actual,predicted=recovered,residual=err,keys=a['keys'])
    for m in FEATURES:
     delta=(zh[:,m]-z[:,m])[:,None]*B[:,m][None,:]
     for pop,idx in populations:groups.append({'population':pop,'features':[m],'n':len(idx),**residual_stats(delta[idx],q)})
    if label=='confirmation':params['primary_data_weighted_action_error']=residual_stats(h@dw.T+db,q);params['DeltaW_action_without_bias']=residual_stats(h@dw.T,q)
 csvwrite(run/'output_metrics.csv',rows);csvwrite(run/'fixed_group_metrics.csv',groups);npz(run/'parameter_map.npz',Delta_W=dw,Delta_b=db);write(run/'parameter_map_diagnostics.json',params)

@torch.no_grad()
def evaluate_responses(run):
 verify(run,'readout_lock.json');man=read(run/'context_manifest.json');stats=dict(np.load(run/'fit_statistics.npz'));maps=load_models(run);rows=[];rawcorners=[];offsets=[];t,s,p,q=models(run)
 for pop,banks,cache in [('primary',man['primary_contexts'],'confirmation'),('legacy',man['legacy_contexts'],'legacy')]:
  if not (run/(cache+'_activations.npz')).exists():continue
  a=dict(np.load(run/(cache+'_activations.npz')));ps=dict(np.load(run/(cache+'_predictions.npz')));lookup={int(k):i for i,k in enumerate(a['keys'])}
  for key,bank in banks.items():
   if not bank['corner_keys']:continue
   idx=np.array([[lookup[k] for k in row] for row in bank['corner_keys']]);z=a['z'][idx];alpha_z=polynomial(np.moveaxis(z,1,-1));blookup={b:i for i,b in enumerate(bank['background_keys'])};saved={'keys':np.array(bank['corner_keys']),'z':z,'z_alpha':alpha_z}
   for method,pr in ps.items():
    pred=pr[idx];ah=polynomial(np.moveaxis(pred,1,-1));saved[method]=pred;saved[method+'_alpha']=ah
    raw=feature_metrics(z.reshape(-1,1024),pred.reshape(-1,1024),stats)
    for r in raw:r.update({'population':pop,'pair':key,'method':method})
    rawcorners+=raw
    changes={'baseline':(z[:,0],pred[:,0]),'first_off':(z[:,2]-z[:,0],pred[:,2]-pred[:,0]),'first_on':(z[:,3]-z[:,1],pred[:,3]-pred[:,1]),'second_off':(z[:,1]-z[:,0],pred[:,1]-pred[:,0]),'second_on':(z[:,3]-z[:,2],pred[:,3]-pred[:,2]),'joint':(alpha_z[:,:,3],ah[:,:,3])}
    for zidx,name in [(0,'alpha0'),(1,'alpha_first'),(2,'alpha_second'),(3,'alpha_joint')]:changes[name]=(alpha_z[:,:,zidx],ah[:,:,zidx])
    for name,(target,recovered) in changes.items():
     rr=response_metrics(target,recovered,stats['v'])
     for r in rr:r.update({'population':pop,'pair':key,'method':method,'change_type':name,'n_contexts':len(idx),'unique_inputs':len(set(idx.flat))})
     rows+=rr
    if bank['nuisance_edges']:
     i0=[blookup[e[0]] for e in bank['nuisance_edges']];i1=[blookup[e[2]] for e in bank['nuisance_edges']];target=z[i1]-z[i0];recovered=pred[i1]-pred[i0];rr=response_metrics(target,recovered,stats['v'])
     for r in rr:r.update({'population':pop,'pair':key,'method':method,'change_type':'nuisance','n_contexts':len(idx),'n_edges':len(i0),'unique_inputs':len(set(idx[i0].flat)|set(idx[i1].flat))})
     rows+=rr
    if method=='native':
     delta_h=a['h'][idx[:,3]]-a['h'][idx[:,2]]-a['h'][idx[:,1]]+a['h'][idx[:,0]];np.testing.assert_allclose(numpy(tensor(delta_h)@maps['native']['M'].T),ah[:,:,3],atol=1e-9,rtol=1e-8)
   # Offset-change identities and gate switching, actual input changes only.
   C=t.W1.weight@p;K=s.encoder.weight@p
   for b,bit,b1 in bank['nuisance_edges']:
    aa0=tensor(inputs([b]));aa1=tensor(inputs([b1]));
    for name,w,bias in [('native',C,t.W1.bias),('TC',K,s.encoder.bias)]:
     q0=(aa0@w.T+bias).squeeze(0);q1=(aa1@w.T+bias).squeeze(0);err=(q1-q0-w[:,bit]).abs().max();assert err<1e-9
     aa=tensor(corners([b,b1],bank['pair']));pre=aa@w.T+bias;switch=((pre[0]>0)!=(pre[1]>0)).sum(0)
     offsets.append({'population':pop,'pair':key,'b0':b,'b1':b1,'nuisance':bit,'model':name,'q_change_identity_max':float(err),'units_with_gate_switch':int((switch>0).sum()),'total_corner_gate_switches':int(switch.sum()),'nonzero_shift_without_gate_switch_units':int(((q1-q0).abs()>1e-12).logical_and(switch==0).sum())})
   npz(run/f'corners_{pop}_{key.replace(",","_")}.npz',**saved)
 csvwrite(run/'finite_response_metrics.csv',rows);csvwrite(run/'context_corner_metrics.csv',rawcorners);csvwrite(run/'nuisance_offset_checks.csv',offsets)


def evaluate_jacobians(run):
 verify(run,'readout_lock.json');man=read(run/'context_manifest.json');ks=man['jacobian_keys'];t,s,p,q=models(run);maps=load_models(run);w=t.W1.weight;E=s.encoder.weight;B=s.decoder.weight;W=t.W2.weight;M=maps['native']['M'];rows=[];outrows=[];jx=[];jhat=[];boundary=[];margins=[];autograd=[];fd=[];semantic_targets=[];semantic_preds=[]
 if not ks:write(run/'jacobian_checks.json',{'status':'no_primary_support'});return
 for start in range(0,len(ks),16):
  x=tensor(inputs(ks[start:start+16]))@p.T;ph=x@w.T+t.W1.bias;pz=x@E.T+s.encoder.bias;gh=(ph>0).double();gz=(pz>0).double();target=gz[:,:,None]*E[None,:,:];native=torch.einsum('mr,br,rd->bmd',M,gh,w)
  mh=ph.abs()/w.norm(dim=1).clamp_min(1e-30);mz=pz.abs()/E.norm(dim=1).clamp_min(1e-30);margin=torch.minimum(mh.min(1).values,mz.min(1).values);boundary.extend(numpy(margin<=1e-8));margins.extend(numpy(margin));jx.append(numpy(target));jhat.append(numpy(native))
  teacher_j=torch.einsum('or,br,rd->bod',W,gh,w);tc_j=torch.einsum('om,bmd->bod',B,target);via_j=torch.einsum('om,bmd->bod',B,native)
  for name,err in [('teacher_minus_TC',teacher_j-tc_j),('teacher_minus_via',teacher_j-via_j),('TC_minus_via',tc_j-via_j)]:outrows.append({'start_index':start,'n':len(x),'residual':name,'squared_error_sum':float(err.square().sum()),'teacher_squared_norm':float(teacher_j.square().sum()),'max_abs':float(err.abs().max())})
  if start==0:
   for n in range(min(2,len(x))):
    xx=x[n].clone().requires_grad_(True)
    for m in FEATURES:
     zz=s.latents(xx)[m];grad=torch.autograd.grad(zz,xx,retain_graph=False)[0];hh=(xx@w.T+t.W1.bias).relu();zh=(M[m]*hh).sum()+maps['native']['d'][m];grad_h=torch.autograd.grad(zh,xx)[0];err1=float((grad-target[n,m]).abs().max());err2=float((grad_h-native[n,m]).abs().max());assert max(err1,err2)<1e-9;autograd.append({'key':ks[n],'feature':m,'TC_max':err1,'native_max':err2})
   for n in range(min(4,len(x))):
    if margin[n]<=1e-8:fd.append({'key':ks[n],'excluded_boundary':True,'margin':float(margin[n])});continue
    step=min(1e-4,float(margin[n])/4);xx=x[n];direction=torch.eye(12,device='cuda',dtype=DT)[0];plus=xx+step*direction;minus=xx-step*direction
    assert torch.equal((plus@w.T+t.W1.bias)>0,gh[n].bool()) and torch.equal((minus@w.T+t.W1.bias)>0,gh[n].bool());assert torch.equal((plus@E.T+s.encoder.bias)>0,gz[n].bool()) and torch.equal((minus@E.T+s.encoder.bias)>0,gz[n].bool())
    zfd=(s.latents(plus)-s.latents(minus))/(2*step);hfd=(predict(maps['native'],(plus@w.T+t.W1.bias).relu())-predict(maps['native'],(minus@w.T+t.W1.bias).relu()))/(2*step);er1=float((zfd-target[n,:,0]).abs().max());er2=float((hfd-native[n,:,0]).abs().max());assert max(er1,er2)<1e-6;fd.append({'key':ks[n],'step':step,'margin':float(margin[n]),'TC_max':er1,'native_max':er2,'excluded_boundary':False})
 target=np.concatenate(jx);native=np.concatenate(jhat);boundary=np.array(boundary);smooth=~boundary;npz(run/'jacobian_arrays.npz',keys=np.array(ks),J_TC_x=target,J_native_x=native,boundary=boundary,normalized_margin=np.array(margins))
 # Semantic columns are correlated directions; computed on CUDA, no reinterpretation as independent coordinates.
 ja=numpy(tensor(target)@p);jna=numpy(tensor(native)@p)
 for mask,name in [(np.ones(len(ks),bool),'all'),(smooth,'smooth')]:
  if not mask.any():continue
  er=native[mask]-target[mask];tn=target[mask];pn=native[mask];normt=np.linalg.norm(tn,axis=2);normp=np.linalg.norm(pn,axis=2);valid=(normt>1e-12)&(normp>1e-12);cos=np.sum(tn*pn,axis=2)/np.maximum(normt*normp,1e-30);angles=np.degrees(np.arccos(np.clip(cos,-1,1)))
  for m in range(1024):
   den=np.sum(tn[:,m]**2);rr={'feature':m,'population':name,'n':int(mask.sum()),'gradient_RMSE':np.sqrt(np.mean(er[:,m]**2)),'relative_Frobenius':float(np.sqrt(np.sum(er[:,m]**2)/den)) if den>0 else None,'valid_angle_n':int(valid[:,m].sum()),'mean_angle_degrees':float(angles[valid[:,m],m].mean()) if valid[:,m].any() else None,'target_zero_gradient_count':int((normt[:,m]<=1e-12).sum()),'native_zero_gradient_count':int((normp[:,m]<=1e-12).sum())}
   for i in range(20):rr[f'a{i}_RMSE']=float(np.sqrt(np.mean((jna[mask,m,i]-ja[mask,m,i])**2)))
   rows.append(rr)
 totalrelative=float(np.linalg.norm(native-target)/np.linalg.norm(target));csvwrite(run/'jacobian_feature_metrics.csv',rows);csvwrite(run/'output_jacobian_metrics.csv',outrows);write(run/'jacobian_checks.json',{'n':len(ks),'boundary_count':int(boundary.sum()),'boundary_rule':'min normalized preactivation distance over all teacher and TC units <=1e-8','gradient_norm_nontrivial':1e-12,'aggregate_relative_Frobenius':totalrelative,'autograd_checks':autograd,'smooth_finite_difference_checks':fd,'semantic_P_rank':int(torch.linalg.matrix_rank(p)),'interpretation':'Boolean training does not supervise local continuous extensions; P columns not independent directions','no_derivative_fitting':True})

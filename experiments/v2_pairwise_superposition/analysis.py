"""V2-specific metrics and original-neuron diagnostics (no transcoder)."""

from itertools import product

import numpy as np
import torch

from src.coordinates import original_coordinates, semantic_forward
from src.interactions import finite_interaction
from .data import PAIRS, DNF_PAIRS, RULES, targets, keys, clause_counts, contexts


def metrics(prediction,g,q,threshold):
    if not len(g):return {'n':0}
    y=prediction.astype(np.float64);q=q.astype(np.float64);truth=g.astype(bool)
    scores=y@q;label=scores>=threshold;residual=y-scores@q.T
    result={'n':len(g),'threshold':threshold,'output_mse':float(((y-g@q.T)**2).mean()),
            'semantic_mse':float(((scores-g)**2).mean()),'semantic_mae':float(np.abs(scores-g).mean()),
            'orthogonal_residual_mse':float((residual**2).mean()),'exact_vector_accuracy':float(np.all(label==truth,axis=1).mean()),
            'mean_feature_accuracy':float((label==truth).mean()),'per_rule':[]}
    for j,rule in enumerate(RULES):
        t,p=truth[:,j],label[:,j]
        tp,tn,fp,fn=[int(v.sum()) for v in (t&p,~t&~p,~t&p,t&~p)]
        pos,neg=int(t.sum()),int((~t).sum())
        recall=tp/pos if pos else None;specificity=tn/neg if neg else None
        margin=np.where(t,scores[:,j]-threshold,threshold-scores[:,j])
        result['per_rule'].append({'rule':rule,'n_positive':pos,'n_negative':neg,'prevalence':pos/len(g),
            'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':float((t==p).mean()),
            'precision':tp/(tp+fp) if tp+fp else None,'recall':recall,'specificity':specificity,
            'balanced_accuracy':(recall+specificity)/2 if pos and neg else None,
            'false_positive_rate':fp/neg if neg else None,'false_negative_rate':fn/pos if pos else None,
            'mse':float(((scores[:,j]-g[:,j])**2).mean()),'mae':float(np.abs(scores[:,j]-g[:,j]).mean()),
            'signed_bias':float((scores[:,j]-g[:,j]).mean()),'margin_min':float(margin.min()),
            'margin_quantiles':np.quantile(margin,[.01,.1,.5,.9,.99]).tolist(),
            'positive_mean_score':float(scores[t,j].mean()) if pos else None,
            'negative_mean_score':float(scores[~t,j].mean()) if neg else None})
    return result


@torch.no_grad()
def predict(model,a,p,device,batch,hidden=False):
    p=torch.as_tensor(p,device=device,dtype=next(model.parameters()).dtype)
    ys=[];pres=[];hs=[]
    model.eval()
    for start in range(0,len(a),batch):
        x=torch.as_tensor(a[start:start+batch],device=device,dtype=p.dtype)@p.T
        pre,h=model.hidden(x);y=model.W2(h)
        if not torch.isfinite(y).all():raise RuntimeError('Non-finite V2 prediction')
        ys.append(y.cpu().numpy())
        if hidden:pres.append(pre.cpu().numpy());hs.append(h.cpu().numpy())
    result={'output':np.concatenate(ys)}
    if hidden:result.update(preactivation=np.concatenate(pres),hidden=np.concatenate(hs))
    return result


def group_metrics(a,prediction,g,q,threshold):
    groups={}
    nuisance=a[:,15:];mask=nuisance.astype(np.int64)@(1<<np.arange(5))
    for kind,codes in [('total_active_count',a.sum(1)),('distractor_count',nuisance.sum(1)),
                       ('distractor_identity_mask_bits15to19',mask),('dnf_satisfied_clauses',clause_counts(a))]:
        groups[kind]={str(int(code)):metrics(prediction[codes==code],g[codes==code],q,threshold) for code in np.unique(codes)}
    for j,pair in enumerate(PAIRS):
        codes=2*a[:,pair[0]]+a[:,pair[1]]
        groups['antecedent_'+RULES[j]]={f'{code:02b}':metrics(prediction[codes==code],g[codes==code],q,threshold) for code in range(4)}
    return groups


def probe_bank(config):
    bases,names=contexts(config['probes'],config['seeds']['contrasts'])
    rows=[];meta=[]
    def add(a,kind,name,pattern,context):
        rows.append(a.copy());meta.append({'kind':kind,'designed_rule':name,'pattern':pattern,'context':context,
                                          'total_active':int(a.sum()),'distractor_count':int(a[15:].sum()),
                                          'outside_training_density':not 3<=a.sum()<=6})
    for j,pair in enumerate(PAIRS):
        for base,name in zip(bases,names):
            for bits in product((0,1),repeat=2):
                a=base.copy();a[list(pair)]=bits
                add(a,'pair_corner',RULES[j],''.join(map(str,bits)),name)
    for base,name in zip(bases,names):
        for bits in product((0,1),repeat=3):
            a=base.copy();a[[0,1,2]]=bits;add(a,'overlap_012','g0_g1_g8',''.join(map(str,bits)),name)
        for bit in range(15,20):
            for value in (0,1):
                a=base.copy();a[bit]=value;add(a,'distractor_flip',str(bit),str(value),name)
        for style in ('inactive00','near_miss_singletons'):
            for pattern in product((0,1),repeat=5):
                a=base.copy()
                for clause,(i,k) in zip(pattern,DNF_PAIRS):
                    a[i],a[k]=(1,1) if clause else ((1,0) if style=='near_miss_singletons' else (0,0))
                add(a,'dnf_clause_table',style,''.join(map(str,pattern)),name)
    return np.asarray(rows,dtype=np.uint8),meta,bases,names


def correlations(x,y):
    x=x.astype(float);y=y.astype(float);x-=x.mean(0);y-=y.mean(0)
    denominator=np.linalg.norm(x,axis=0)[:,None]*np.linalg.norm(y,axis=0)[None,:]
    return np.divide(x.T@y,denominator,out=np.zeros_like(denominator),where=denominator>0)


@torch.no_grad()
def native_diagnostics(model,p,q,bases,config,device):
    model.double()
    p,q=[torch.tensor(v,dtype=torch.float64,device=device) for v in (p,q)]
    coords=original_coordinates(model,p,q)
    context=torch.tensor(bases,dtype=torch.float64,device=device)
    torch.testing.assert_close(model(context@p.T)@q,semantic_forward(context,coords),atol=config['evaluation']['interaction_atol'],rtol=config['evaluation']['interaction_rtol'])
    results=[];all_contributions=[];all_values=[];all_truth=[];fixed_topk={}
    pairs=PAIRS+((1,2),)
    for i,pair in enumerate(pairs):
        result=finite_interaction(context,pair,coords)
        a=result['corners'];signs=result['signs']
        direct=(model((a@p.T).reshape(-1,12))@q).reshape(len(context),4,9)
        delta=(direct*signs[None,:,None]).sum(1)
        torch.testing.assert_close(result['value'],delta,atol=config['evaluation']['interaction_atol'],rtol=config['evaluation']['interaction_rtol'])
        torch.testing.assert_close(result['positive_sum']+result['negative_sum'],delta,atol=config['evaluation']['interaction_atol'],rtol=config['evaluation']['interaction_rtol'])
        c=result['contributions'].cpu().numpy();value=result['value'].cpu().numpy()
        truth=(targets(a.cpu().numpy())*signs.cpu().numpy()[None,:,None]).sum(1)
        all_contributions.append(c);all_values.append(value);all_truth.append(truth)
        reference=np.arange(0,len(context),2);held=np.arange(1,len(context),2)
        rank=np.argsort(-np.abs(c[reference]).mean(0),axis=-1)
        fixed_topk[str(pair)]={}
        for k in config['evaluation']['top_neuron_counts']:
            selected=np.take_along_axis(c,np.broadcast_to(rank[None,:,:k],(len(context),9,k)),axis=-1).sum(-1)
            fixed_topk[str(pair)][str(k)]={'selected_neurons_per_output':rank[:,:k].tolist(),
                'reference_context_indices':reference.tolist(),'held_context_indices':held.tolist(),
                'held_context_mae_to_full_teacher_interaction':np.abs(selected[held]-value[held]).mean(0).tolist()}
        for context_idx in range(len(context)):
            for j in range(9):
                vec=c[context_idx,j];mass=np.abs(vec).sum()
                results.append({'pair':list(pair),'context_index':context_idx,'output':j,'ground_truth':float(truth[context_idx,j]),
                    'native_value':float(value[context_idx,j]),'absolute_error':float(abs(value[context_idx,j]-truth[context_idx,j])),
                    'positive_sum':float(np.maximum(vec,0).sum()),'negative_sum':float(np.minimum(vec,0).sum()),
                    'effective_neurons':float(mass**2/np.square(vec).sum()) if mass else 0,
                    'largest_positive_neuron':int(vec.argmax()),'largest_negative_neuron':int(vec.argmin()),
                    'direct_sum_residual':float(abs(value[context_idx,j]-delta.cpu().numpy()[context_idx,j]))})
    return {'pairs':np.asarray(pairs),'contributions':np.stack(all_contributions),'native_values':np.stack(all_values),'ground_truth':np.stack(all_truth),
            **{key:value.cpu().numpy() for key,value in coords.items()}},results,fixed_topk

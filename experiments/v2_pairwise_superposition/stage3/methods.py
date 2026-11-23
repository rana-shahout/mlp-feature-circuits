"""Signed subset searches and discovery-only threshold hypotheses."""
import numpy as np
import torch


def relative(h):return h-h[:,0:1,:]


def signals(h,readout):
    effect=relative(h)*readout
    interaction=(h[:,3]-h[:,2]-h[:,1]+h[:,0])*readout
    return effect,interaction


def joint_design(effect,interaction,target_effect,target_interaction):
    # Equal expected squared-error weight to corner-relative output and finite interaction.
    e=max(float(target_effect.square().mean()),1e-12);i=max(float(target_interaction.square().mean()),1e-12)
    x=torch.cat([effect.reshape(-1,effect.shape[-1])*(.5/(target_effect.numel()*e))**.5,interaction.reshape(-1,interaction.shape[-1])*(.5/(target_interaction.numel()*i))**.5],0)
    y=torch.cat([target_effect.reshape(-1)*(.5/(target_effect.numel()*e))**.5,target_interaction.reshape(-1)*(.5/(target_interaction.numel()*i))**.5],0)
    return x,y


def sign_design(effect,interaction,weights,target_positive,target_negative):
    # Net plus opposing witness channels. Original contribution signs/writes are unchanged.
    pe,pi=target_positive;ne,ni=target_negative
    x,y=joint_design(effect,interaction,pe+ne,pi+ni);xs=[x*(.5**.5)];ys=[y*(.5**.5)]
    for mask,te,ti in [(weights>0,pe,pi),(weights<0,ne,ni)]:
        xx,yy=joint_design(effect*mask,interaction*mask,te,ti);xs.append(xx*(.25**.5));ys.append(yy*(.25**.5))
    return torch.cat(xs),torch.cat(ys)


def subset_search(x,y,budgets,passes=8,anchors=()):
    """Binary coefficients fixed at one; signed forward selection and one-swap refinement.

    No original readout is refitted. Refinement is independently performed per budget;
    sets need not be nested. Anchors may be fixed only for transcoder group construction.
    """
    gram=x.T@x;cross=x.T@y;diag=gram.diag();solutions={}
    for k in budgets:
        if k<len(anchors):raise ValueError('Budget smaller than fixed anchors')
        selected=list(anchors)
        while len(selected)<k:
            score=diag-2*cross
            if selected:score=score+2*gram[:,selected].sum(1)
            score[selected]=float('inf');selected.append(int(score.argmin()))
        for _ in range(passes):
            best_delta=0.;best=None;sumg=gram[:,selected].sum(1)
            for old in selected:
                if old in anchors:continue
                # f(S-old+new)-f(S), using fixed original signed contributions.
                delta=diag-2*cross+2*(sumg-gram[:,old])-(diag[old]-2*cross[old]+2*(sumg[old]-gram[old,old]))
                delta[selected]=float('inf');new=int(delta.argmin());value=float(delta[new])
                if value<best_delta-1e-12:best_delta=value;best=(old,new)
            if best is None:break
            selected[selected.index(best[0])]=best[1]
        solutions[k]=sorted(selected)
    return solutions


def gate_parameters(c1,bias,bases,context_count=2):
    """Keep exact pair coefficients plus two discovery-variable context reads per neuron."""
    mean=bases.mean(0);variation=bases.std(0);importance=np.abs(c1)*variation[None,:];importance[:,[7,8]]=-np.inf
    chosen=np.argsort(-importance,axis=1)[:,:context_count]
    retained=np.zeros_like(c1);retained[:,[7,8]]=c1[:,[7,8]]
    np.put_along_axis(retained,chosen,np.take_along_axis(c1,chosen,axis=1),axis=1)
    offset=bias+((c1-retained)*mean).sum(1)
    invariant_bias=bias+(c1*mean).sum(1)
    return {'retained_C1':retained,'offset':offset,'context_indices':chosen,'invariant_offset':invariant_bias,'pair_C1':c1[:,[7,8]],'discovery_mean_base':mean}


def predicted_hidden(a,parameters,kind):
    if kind=='context2':return np.maximum(a@parameters['retained_C1'].T+parameters['offset'],0)
    if kind=='invariant':return np.maximum(a[...,[7,8]]@parameters['pair_C1'].T+parameters['invariant_offset'],0)
    raise ValueError(kind)


def group_metrics(h,output,readout,bias,indices,offset):
    v=h[...,indices]@readout[:,indices].T
    approximation=v+offset
    pair=v-v[:,0:1,:];truth=output-output[:,0:1,:]
    delta=v[:,3]-v[:,2]-v[:,1]+v[:,0];target=output[:,3]-output[:,2]-output[:,1]+output[:,0]
    return {'absolute_g4_mse':float(((approximation[:,:,4]-output[:,:,4])**2).mean()),'pair_response_g4_rmse':float(((pair[:,:,4]-truth[:,:,4])**2).mean()**.5),'interaction_g4_mae':float(np.abs(delta[:,4]-target[:,4]).mean()),'interaction_g4_rmse':float(((delta[:,4]-target[:,4])**2).mean()**.5),'mean_interaction':float(delta[:,4].mean()),'interaction_sd':float(delta[:,4].std()),'per_context_interaction_error':delta[:,4]-target[:,4],'per_context_pair_response_mse':((pair[:,:,4]-truth[:,:,4])**2).mean(1)}

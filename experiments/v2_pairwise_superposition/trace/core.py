"""Exact four-corner values and Boolean polynomial coefficients."""
import numpy as np
import torch

BITS=np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.uint8)


def polynomial(values):
    """Last axis order00,01,10,11 -> alpha0,alpha7,alpha8,alpha78."""
    s=np.asarray(values)
    if s.shape[-1]!=4:raise ValueError('Expected four corners on last axis')
    return np.stack([s[...,0],s[...,2]-s[...,0],s[...,1]-s[...,0],s[...,3]-s[...,2]-s[...,1]+s[...,0]],-1)


def from_polynomial(alpha):
    c=np.asarray(alpha);return np.stack([c[...,0],c[...,0]+c[...,2],c[...,0]+c[...,1],c.sum(-1)],-1)


def select_pair(manifest,known,old_probe_keys):
    """Metadata-only lexicographic rule. No model, output, latent or attribution input.

    Prefer discovery, then heldout; require documented reserved exclusions for every
    corner, and reject any historical-test corner. Never open reserved identities.
    """
    certified=set(known)|set(old_probe_keys)
    for split in ('discovery','heldout'):
        candidates=[];pool=set(map(int,manifest[split]['context_keys']))
        for base in sorted(pool):
            if base.bit_count()!=3 or base&((1<<7)|(1<<8)):continue
            for bit in range(15,20):
                if base&(1<<bit):continue
                other=base|(1<<bit)
                if other not in pool:continue
                corners=[b|p for b in (base,other) for p in (0,1<<8,1<<7,(1<<7)|(1<<8))]
                if any(known.get(k)=='test' for k in corners):continue
                if all(k in certified for k in corners):candidates.append((base,bit,other,corners))
        if candidates:
            base,bit,other,corners=candidates[0]
            return {'partition':split,'background_keys':[base,other],'flipped_input':bit,'input_keys':np.array(corners,dtype=np.uint64).reshape(2,4).astype(int).tolist(),'qualified_pair_count_in_partition':len(candidates)}
    raise RuntimeError('No existing pair satisfies density and documented reserved-exclusion constraints. No model outputs were inspected; do not silently select unverified inputs.')


def corner_inputs(backgrounds):
    a=np.repeat(np.asarray(backgrounds)[:,None,:],4,axis=1);a[:,:,[7,8]]=BITS[None,:,:];return a


@torch.no_grad()
def trace(coordinates,backgrounds,direct_preactivation,direct_output,displayed):
    """Use GPU tensors throughout algebraic model checks; return FP64 exports."""
    w,d,b,beta=[coordinates[k] for k in ('C1','D2_eff','b1','semantic_bias')]
    if w.shape[1]!=20 or d.shape!=(9,len(w)) or backgrounds.shape!=(2,20):raise ValueError('Trace dimensions do not match V2')
    terms=backgrounds[:,None,:]*w[None,:,:];q=b+terms.sum(-1);u=w[:,7];v=w[:,8]
    bits=torch.tensor(BITS,device=w.device,dtype=w.dtype)
    pre=q[:,None,:]+bits[None,:,0,None]*u+bits[None,:,1,None]*v
    torch.testing.assert_close(pre,direct_preactivation,atol=1e-10,rtol=1e-10)
    activation=pre.relu();contributions=activation[:,:,None,:]*d[None,None,:,:];semantic=contributions.sum(-1)+beta
    torch.testing.assert_close(semantic,direct_output,atol=1e-10,rtol=1e-10)
    selected=contributions[:,:,:,displayed].sum(-1);keep=torch.ones(len(w),dtype=torch.bool,device=w.device);keep[displayed]=False
    remainder=contributions[:,:,:,keep].sum(-1)
    torch.testing.assert_close(selected+remainder+beta,direct_output,atol=1e-10,rtol=1e-10)
    result={'input_coefficients':w,'semantic_readout':d,'hidden_bias':b,'semantic_bias':beta,'background_terms':terms,'q':q,'u':u,'v':v,'preactivation':pre,'activation':activation,'gate':pre>0,'semantic_contributions':contributions,'semantic_total':semantic,'semantic_remainder':remainder,'displayed':torch.tensor(displayed,device=w.device),'coordinate_vs_direct_residual':semantic-direct_output,'displayed_remainder_bias_residual':selected+remainder+beta-direct_output}
    result={k:v.cpu().numpy() for k,v in result.items()}
    result['g4_alpha']=polynomial(np.moveaxis(result['semantic_contributions'][:,:,4,:],1,-1))
    result['g4_total_alpha']=polynomial(result['semantic_total'][:,:,4]);result['g4_remainder_alpha']=polynomial(result['semantic_remainder'][:,:,4])
    result['g4_bias_alpha']=polynomial(np.full((2,4),result['semantic_bias'][4]))
    reconstructed=result['g4_alpha'][:,displayed].sum(1)+result['g4_remainder_alpha']+result['g4_bias_alpha']
    np.testing.assert_allclose(reconstructed,result['g4_total_alpha'],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(from_polynomial(result['g4_alpha']),np.moveaxis(result['semantic_contributions'][:,:,4,:],1,-1),atol=1e-10,rtol=1e-10)
    positive=(result['u']>0)&(result['v']>0)
    result['positive_u_v']=positive;result['positive_11_only_band']=np.where(positive[None,:],(result['q']>-(result['u']+result['v']))&(result['q']<=-np.maximum(result['u'],result['v'])),False)
    observed=np.all(result['gate']==np.array([False,False,False,True])[None,:,None],axis=1)
    np.testing.assert_array_equal(result['positive_11_only_band'][:,positive],observed[:,positive])
    return result

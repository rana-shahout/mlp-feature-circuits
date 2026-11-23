"""SPEC v0.3 data only. Row batches: X=a@P.T, Y=g@Q.T."""

from itertools import combinations, product

import numpy as np


RULES=('and_01','and_02','and_34','and_56','and_78','and_910','xor_1112','inhibition_1314','dnf_five')
PAIRS=((0,1),(0,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14))
DNF_PAIRS=(PAIRS[0],PAIRS[2],PAIRS[3],PAIRS[4],PAIRS[5])


def targets(a):
    a=np.asarray(a)
    if a.ndim<1 or a.shape[-1]!=20 or not np.all((a==0)|(a==1)):
        raise ValueError('V2 expects binary inputs (...,20)')
    b=a.astype(bool)
    primitive=[b[...,i]&b[...,j] for i,j in PAIRS[:6]]
    primitive += [b[...,11]^b[...,12],b[...,13]&~b[...,14]]
    dnf=primitive[0]|primitive[2]|primitive[3]|primitive[4]|primitive[5]
    return np.stack(primitive+[dnf],axis=-1).astype(np.float32)


def keys(a):
    return np.asarray(a,dtype=np.uint64)@(np.uint64(1)<<np.arange(20,dtype=np.uint64))


def clause_counts(a):
    return np.stack([np.asarray(a)[...,i]*np.asarray(a)[...,j] for i,j in DNF_PAIRS],axis=-1).sum(-1,dtype=np.int64)


def make_embeddings(seed_p,seed_q):
    p=np.random.default_rng(seed_p).normal(size=(12,20));p/=np.linalg.norm(p,axis=0,keepdims=True)
    q,r=np.linalg.qr(np.random.default_rng(seed_q).normal(size=(12,9)),mode='reduced')
    q*=np.where(np.diag(r)<0,-1.,1.)[None,:]
    p,q=p.astype(np.float32),q.astype(np.float32)
    if np.linalg.matrix_rank(p)!=12:raise ValueError('P must have full row rank')
    np.testing.assert_allclose(np.linalg.norm(p,axis=0),1,atol=1e-6)
    np.testing.assert_allclose(q.T@q,np.eye(9),atol=1e-6)
    return p,q


def local_pools(rule):
    bits=np.asarray(list(product((0,1),repeat=2)),dtype=np.uint8)
    a=np.zeros((4,20),dtype=np.uint8);a[:,list(PAIRS[rule])]=bits
    positive=targets(a)[:,rule].astype(bool)
    distance=np.abs(bits[:,None,:].astype(int)-bits[positive][None,:,:].astype(int)).sum(-1)
    return bits[positive],bits[(~positive)&(distance.min(1)==1)]


def generate(settings,seed):
    rng=np.random.default_rng(seed)
    seen=set();rows=[];modes=[];requested=[]
    proposed=np.zeros(3,dtype=int);rejected=np.zeros(3,dtype=int)
    counts=np.asarray(settings['active_counts']);prob=np.asarray(settings['active_probabilities'])
    pools=[local_pools(i) for i in range(8)]
    attempts=0
    while len(rows)<settings['n_samples'] and attempts<settings['max_attempts']:
        attempts+=1
        mode=int(rng.choice(3,p=settings['mode_probabilities']));proposed[mode]+=1
        a=np.zeros(20,dtype=np.uint8);rule=-1;fixed=[]
        if mode:
            rule=int(rng.integers(9))
            if rule<8:
                fixed=list(PAIRS[rule]);pool=pools[rule][0 if mode==1 else 1]
                a[fixed]=pool[int(rng.integers(len(pool)))]
            elif mode==1:
                number=int(rng.choice(settings['dnf_positive_clause_counts'],p=settings['dnf_positive_clause_probabilities']))
                for j in rng.choice(5,number,replace=False):
                    for bit in DNF_PAIRS[j]:a[bit]=1;fixed.append(bit)
            else:
                # No complete DNF clause; multiple one-bit near misses.
                fixed=[i for pair in DNF_PAIRS for i in pair]
                for j in rng.choice(5,int(rng.integers(3,6)),replace=False):
                    a[DNF_PAIRS[j][int(rng.integers(2))]]=1
        current=int(a.sum());available=np.setdiff1d(np.arange(20),fixed)
        valid=(counts>=current)&(counts-current<=len(available))
        if not valid.any():rejected[mode]+=1;continue
        k=int(rng.choice(counts[valid],p=prob[valid]/prob[valid].sum()))
        a[rng.choice(available,k-current,replace=False)]=1
        if mode and targets(a)[rule]!=(mode==1):raise AssertionError('Constructed label failed verification')
        key=int(keys(a))
        if key in seen:rejected[mode]+=1;continue
        seen.add(key);rows.append(a);modes.append(mode);requested.append(rule)
    if len(rows)<settings['min_samples']:raise RuntimeError(f'Only {len(rows)} unique rows in {attempts} attempts')
    order=rng.permutation(len(rows))
    return np.asarray(rows)[order],np.asarray(modes,dtype=np.int8)[order],np.asarray(requested,dtype=np.int8)[order],{
        'requested':settings['n_samples'],'accepted':len(rows),'attempts':attempts,
        'proposed_by_mode':proposed.tolist(),'rejected_by_mode':rejected.tolist(),
        'generator_version':'v2_pairwise_1','note':'Deduplication/feasibility rejection changes accepted mixture and density.'}


def stratified_split(a,fractions,seed,minimum):
    rng=np.random.default_rng(seed);g=targets(a)
    label=(g.astype(np.int64)@(1<<np.arange(9)))
    strata=label+512*(2*a[:,11]+a[:,12]).astype(np.int64)+2048*np.minimum(clause_counts(a),2)
    values,inverse,counts=np.unique(strata,return_inverse=True,return_counts=True)
    rare=counts[inverse]<minimum
    pooled=strata.astype(np.int64).copy();pooled[rare]=-1
    values,inverse,counts=np.unique(pooled,return_inverse=True,return_counts=True)
    fractions=np.asarray(fractions,dtype=float)
    quotas=np.floor(len(a)*fractions).astype(int)
    for p in np.argsort(-(len(a)*fractions-quotas),kind='stable')[:len(a)-quotas.sum()]:quotas[p]+=1
    ideals=counts[:,None]*fractions[None,:];allocation=np.floor(ideals).astype(int)
    remaining=counts-allocation.sum(1);deficit=quotas-allocation.sum(0)
    tie=rng.random(ideals.shape)*1e-12
    while remaining.sum():
        score=ideals-allocation+tie
        score[remaining==0,:]=-np.inf;score[:,deficit==0]=-np.inf
        s,p=np.unravel_index(np.argmax(score),score.shape)
        allocation[s,p]+=1;remaining[s]-=1;deficit[p]-=1
    parts=[[],[],[]];manifest=[]
    for s,value in enumerate(values):
        idx=rng.permutation(np.flatnonzero(inverse==s));splits=np.split(idx,np.cumsum(allocation[s])[:2])
        for p in range(3):parts[p].extend(splits[p].tolist())
        manifest.append({'stratum':int(value),'n':int(counts[s]),'allocation':allocation[s].tolist()})
    return {name:rng.permutation(part).astype(np.int64) for name,part in zip(('train','validation','test'),parts)}, {
        'strategy':'combined nine-label pattern + XOR corner + DNF 0/1/multiple; strata below minimum pooled',
        'minimum_stratum_size':minimum,'pooled_rare_rows':int(rare.sum()),'strata':manifest}


def audit(a,mode=None):
    g=targets(a);n=len(a);density=a.sum(1);clauses=clause_counts(a)
    result={'n':n,'unique':len(np.unique(keys(a))),'positive_counts':g.sum(0).astype(int).tolist(),
            'negative_counts':(n-g.sum(0)).astype(int).tolist(),'prevalence':g.mean(0).tolist(),
            'mean_active':float(density.mean()),'density_histogram':{str(int(k)):int((density==k).sum()) for k in np.unique(density)},
            'pair_corners_00_01_10_11':{RULES[j]:np.bincount(2*a[:,i]+a[:,k],minlength=4).tolist() for j,(i,k) in enumerate(PAIRS)},
            'dnf_clause_histogram':np.bincount(clauses,minlength=6).tolist(),
            'dnf_zero_one_multiple':[int((clauses==0).sum()),int((clauses==1).sum()),int((clauses>=2).sum())],
            'distractor_count_histogram':np.bincount(a[:,15:].sum(1).astype(int),minlength=6).tolist()}
    if mode is not None:
        result['mode_counts']=np.bincount(mode,minlength=3).tolist()
        result['prevalence_by_mode']={str(i):g[mode==i].mean(0).tolist() if (mode==i).any() else None for i in range(3)}
    nuisance=a[:,15:].astype(float);c=nuisance-nuisance.mean(0);d=g-g.mean(0)
    denom=np.linalg.norm(c,axis=0)[:,None]*np.linalg.norm(d,axis=0)[None,:]
    corr=np.divide(c.T@d,denom,out=np.zeros((5,9)),where=denom>0)
    result['nuisance_phi']=corr.tolist()
    return result


def contexts(settings,seed):
    order=np.random.default_rng(seed).permutation(np.arange(15,20));rows=[];names=[]
    for semantic in settings['semantic_contexts']:
        for count in settings['distractor_counts']:
            a=np.zeros(20,dtype=np.uint8);a[semantic['active']]=1;a[order[:count]]=1
            rows.append(a);names.append(f"{semantic['name']}:d{count}")
    return np.asarray(rows),names

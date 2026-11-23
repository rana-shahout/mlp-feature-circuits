"""Discovery-frozen AND(7,8) study and actual original-neuron interventions."""
import json,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models import SparseTranscoder
from src.runtime import load_checkpoint,select_device,seed_everything,write_json
from src.coordinates import original_coordinates,transcoder_coordinates
from src.interactions import corners_and_signs,decompose_corners
from ..run import BASE,ROOT,digest,write_csv
from ..data import generate,keys,targets
from ..capacity256 import load_frozen_data,signed_statistics
from ..analysis import metrics
from ..stage2.core import frozen_teacher,infer,fidelity
from .methods import signals,joint_design,sign_design,subset_search,gate_parameters,predicted_hidden,group_metrics

C=yaml.safe_load((BASE/'configs/stage3_and78.yaml').read_text());ID=C['run_id'];RUN=BASE/'runs'/ID;OUT=BASE/'reports'/ID
TEACHER=ROOT/C['teacher'];TC=ROOT/C['transcoder']


def corners_np(bases):
    a=np.repeat(bases[:,None,:],4,axis=1);a[:,:,[7,8]]=np.array([[0,0],[0,1],[1,0],[1,1]])[None,:,:];return a


def development(d,settings):
    cfg=yaml.safe_load((BASE/'runs/capacity256-20261027/config.yaml').read_text())['data'];cfg=dict(cfg);cfg['n_samples']=settings['proposals_unique'];cfg['max_attempts']=1000000;cfg['min_samples']=min(cfg['min_samples'],cfg['n_samples'])
    raw,_,_,meta=generate(cfg,settings['generator_seed']);base=raw.copy();base[:,[7,8]]=0
    _,idx=np.unique(keys(base),return_index=True);base=base[np.sort(idx)]
    oldtest=set(map(int,keys(d['a'][d['test']])))
    with np.load(BASE/'reports/capacity256-20261027/signed_interactions_all_widths.npz') as p:historical=set(map(int,keys(p['p4_corners'][:,0,:].astype(np.uint8))))
    eligible=[];stress=[];excluded_test=0
    for v in base:
        corners=corners_np(v[None])[0]
        if any(int(k) in oldtest for k in keys(corners)):excluded_test+=1;continue
        if int(keys(v)) in historical:continue
        if int(v.sum()) in (3,4):eligible.append(v)
        else:stress.append(v)
    rng=np.random.default_rng(settings['split_seed']);eligible=np.array(eligible,dtype=np.uint8);stress=np.array(stress,dtype=np.uint8)
    rng.shuffle(eligible);rng.shuffle(stress);n=settings['discovery'];m=settings['heldout'];s=settings['stress_heldout']
    if len(eligible)<n+m or len(stress)<s:raise RuntimeError(f'Insufficient new context support: {len(eligible)} supported, {len(stress)} stress')
    banks={'discovery':eligible[:n],'heldout':eligible[n:n+m],'stress':stress[:s]}
    known={int(k):split for split in ('train','validation','test') for k in keys(d['a'][d[split]])}
    metadata={};manifest={};allkeys=[]
    for split,b in banks.items():
        a=corners_np(b);ks=keys(a);allkeys.extend(ks.reshape(-1).tolist());overlap=[[known.get(int(k),'unseen_in_original_dataset') for k in row] for row in ks]
        metadata[split]={'n_contexts':len(b),'n_corner_memberships':len(a)*4,'background_density':{str(int(v)):int((b.sum(1)==v).sum()) for v in np.unique(b.sum(1))},'corner_overlap_counts':{label:sum(row.count(label) for row in overlap) for label in ('train','validation','test','unseen_in_original_dataset')},'distractor_identities':{str(int(mask)):int(count) for mask,count in zip(*np.unique(b[:,15:].astype(np.int64)@(1<<np.arange(5)),return_counts=True))}}
        manifest[split]={'context_keys':keys(b).astype(int).tolist(),'corner_keys':ks.astype(int).tolist(),'corner_original_overlap':overlap}
    assert len(allkeys)==len(set(allkeys)), 'Input overlap across context banks'
    metadata['generation']={'original_generator':meta,'unique_raw_backgrounds':len(base),'eligible_supported_contexts':len(eligible),'eligible_stress_contexts':len(stress),'excluded_old_test_contexts':excluded_test,'rule':'Clear a7/a8 then deduplicate; supported backgrounds3/4 yield all corners3..6; shuffle without model outcomes','reserved_final':'File not accessed; unknown overlap cannot be ruled out. No final-population evaluation is performed. Future final confirmation must audit/exclude these development keys before claiming independence.'}
    return banks,metadata,manifest


@torch.no_grad()
def evaluate_bank(bases,teacher,student,p,q):
    a,signs=corners_and_signs(torch.tensor(bases,device=p.device,dtype=p.dtype),(7,8));x=a@p.T
    pre,h=teacher.hidden(x.reshape(-1,12));h=h.reshape(len(bases),4,256);y=teacher.W2(h);z=student.latents(x);ty=student.decoder(z)
    native=original_coordinates(teacher,p,q);tc=transcoder_coordinates(student,p,q)
    nd=decompose_corners(a,signs,native);td=decompose_corners(a,signs,tc)
    for decomposition,raw in [(nd,y),(td,ty)]:torch.testing.assert_close(decomposition['value'],(raw@q*signs[None,:,None]).sum(1),atol=1e-9,rtol=1e-8)
    return {'a':a.cpu().numpy(),'h':h.cpu().numpy(),'z':z.cpu().numpy(),'output':(y@q).cpu().numpy(),'tc_output':(ty@q).cpu().numpy(),'native_contributions':nd['contributions'].cpu().numpy(),'tc_contributions':td['contributions'].cpu().numpy(),'native_value':nd['value'].cpu().numpy(),'tc_value':td['value'].cpu().numpy()}


def tensors(bank,readout,device,key='h'):
    h=torch.tensor(bank[key],device=device);r=torch.tensor(readout[4],device=device);return signals(h,r)


def select(bank,native,tc,device):
    effect,interaction=tensors(bank,native['D2_eff'],device);te,ti=tensors(bank,tc['D2_eff'],device,'z')
    tx,ty=joint_design(te,ti,te.sum(-1),ti.sum(-1))
    tcsets=subset_search(tx,ty,C['tc_budgets'],C['selection']['swap_passes'],C['tc_anchors']);tcrows=[];chosen=C['tc_budgets'][-1]
    for k in C['tc_budgets']:
        group=tcsets[k];pair=float((te[:,:,group].sum(-1)-te.sum(-1)).square().mean().sqrt());inter=float((ti[:,group].sum(-1)-ti.sum(-1)).abs().mean());tcrows.append({'k':k,'units':group,'discovery_vs_tc_pair_rmse':pair,'discovery_vs_tc_interaction_mae':inter})
    for r in tcrows:
        if r['discovery_vs_tc_pair_rmse']<=C['selection']['useful_pair_response_rmse'] and r['discovery_vs_tc_interaction_mae']<=C['selection']['useful_interaction_mae']:chosen=r['k'];break
    group=tcsets[chosen];weights=torch.tensor(native['D2_eff'][4],device=device);tw=torch.tensor(tc['D2_eff'][4],device=device)
    dx,dy=joint_design(effect,interaction,effect.sum(-1),interaction.sum(-1));direct=subset_search(dx,dy,C['budgets'],C['selection']['swap_passes'])
    sx,sy=sign_design(effect,interaction,weights,(effect[:,:,weights>0].sum(-1),interaction[:,weights>0].sum(-1)),(effect[:,:,weights<0].sum(-1),interaction[:,weights<0].sum(-1)));signed=subset_search(sx,sy,C['budgets'],C['selection']['swap_passes'])
    ge,gi=te[:,:,group],ti[:,group];gw=tw[group]
    gx,gy=sign_design(effect,interaction,weights,(ge[:,:,gw>0].sum(-1),gi[:,gw>0].sum(-1)),(ge[:,:,gw<0].sum(-1),gi[:,gw<0].sum(-1)));guided=subset_search(gx,gy,C['budgets'],C['selection']['swap_passes'])
    ranking=np.argsort(-np.abs(bank['native_contributions'][:,4,:]).mean(0))
    groups=[]
    for method,sets in [('tc_guided_signed',guided),('direct_joint',direct),('direct_signed',signed),('direct_abs', {k:ranking[:k].tolist() for k in C['budgets']})]:
        for k in C['budgets']:groups.append({'name':f'{method}_k{k}','method':method,'k':k,'neurons':sets[k],'control':False})
    groups.append({'name':'historical_224_192_30','method':'historical_native','k':3,'neurons':[224,192,30],'control':False})
    return groups,{'tc_sets':tcrows,'selected_tc_budget':chosen,'selected_tc_units':group,'direct_objective':'0.5 normalized pair-relative output MSE +0.5 normalized pair-interaction MSE; subset coefficients fixed1','signed_objective':'0.5 net joint objective +0.25 positive-readout joint objective +0.25 negative-readout joint objective; sign channels defined by g4 write sign, not assumed interaction sign','tc_guidance':'Same signed objective but target is the selected transcoder group, using nonlinear corner profiles, actual signed g4 writes and context dependence; no one-to-one matching/cosine-only selection'}


def controls(groups,bank,readout):
    rng=np.random.default_rng(C['controls']['seed']);v=bank['native_contributions'][:,4,:];h=bank['h'];effect=(h-h[:,0:1])*readout[4];mass=np.abs(v).mean(0);rms=(effect**2).mean((0,1))**.5
    features=np.stack([mass,rms],1);prob=(mass+mass.mean()*.02);prob/=prob.sum();out=[]
    for k in C['budgets']:
        for rep in range(C['controls']['random_replicates']):out.append({'name':f'random_k{k}_{rep}','method':'random','k':k,'neurons':sorted(rng.choice(256,k,replace=False).tolist()),'control':True})
    for target in groups:
        if target['method']=='historical_native':continue
        selected=target['neurons'];k=target['k'];want=features[selected].sum(0);signs=readout[4]>0;npositive=int(signs[selected].sum());pos=np.flatnonzero(signs);neg=np.flatnonzero(~signs)
        pool=set()
        for _ in range(C['controls']['matched_pool']):
            a=rng.choice(pos,npositive,replace=False,p=prob[pos]/prob[pos].sum());b=rng.choice(neg,k-npositive,replace=False,p=prob[neg]/prob[neg].sum());candidate=tuple(sorted(map(int,np.r_[a,b])))
            if candidate!=tuple(sorted(selected)):pool.add(candidate)
        ranked=sorted(pool,key=lambda g:float((((features[list(g)].sum(0)-want)/np.maximum(want,1e-12))**2).sum()))
        for rep,g in enumerate(ranked[:C['controls']['matched_replicates']]):
            achieved=features[list(g)].sum(0);relative=(achieved-want)/np.maximum(want,1e-12)
            out.append({'name':f'matched_{target["name"]}_{rep}','method':'contribution_matched','k':k,'neurons':list(g),'control':True,'matched_to':target['name'],'matching_relative_errors':relative.tolist(),'within_25pct':bool((np.abs(relative)<=.25).all()),'overlap_fraction':len(set(g)&set(selected))/k,'note':'Controls may overlap the candidate to make strong-contribution matching feasible; distinct set required. Poor matching is retained and flagged.'})
    return out


def predictions(groups,bank,native):
    parameters=gate_parameters(native['C1'],native['b1'],bank['a'][:,0],C['prediction']['retained_context_coefficients_per_neuron']);readout=native['D2_eff'];clamp=bank['h'][:,0].mean(0);records=[]
    for group in groups:
        s=group['neurons'];actual=bank['h'][:,:,s]@readout[:,s].T
        record={**group,'discovery_intercept':(bank['output']-actual).mean((0,1)).tolist(),'discovery_zero_effect_mean_corners':(-actual).mean(0).tolist(),'discovery_clamp_effect_mean_corners':((clamp[s]-bank['h'][:,:,s])@readout[:,s].T).mean(0).tolist(),'discovery_interaction_effect_mean':(-bank['native_contributions'][:,:,s].sum(-1)).mean(0).tolist()}
        records.append(record)
    return records,parameters,clamp


def model_verification(teacher,student,d,device):
    p=torch.tensor(d['P'],device=device);q=d['Q'];x=torch.tensor(d['a'][d['validation']],device=device,dtype=torch.float32)@p.T
    y,_=infer(teacher,x);ty,_=infer(student,x)
    for root,output in [(BASE/'reports/capacity256-20261027',y),(BASE/'reports/stage2-transcoders-20261027/w1024_l0.0001',ty)]:
        with np.load(root/'predictions_validation.npz') as saved:np.testing.assert_allclose(output,saved['output'],atol=2e-6,rtol=2e-5)
    m=metrics(y,d['g'][d['validation']],q,.5);f=fidelity(ty,y,d['g'][d['validation']],q)
    return {'teacher_sha256':digest(TEACHER),'transcoder_sha256':digest(TC),'dataset_sha256':digest(BASE/'data/first-teacher-20261027/dataset.npz'),'manifest_sha256':digest(BASE/'data/first-teacher-20261027/split_manifest.json'),'teacher_validation':m,'transcoder_validation':f['teacher_fidelity'],'teacher_frozen':all(not p.requires_grad for p in teacher.parameters()),'transcoder_frozen':all(not p.requires_grad for p in student.parameters())}


def main():
    device=select_device('cuda');seed_everything(5310300,True)
    for p in (RUN,OUT):p.mkdir(parents=True,exist_ok=False)
    before={str(p):digest(p) for root in (BASE/'checkpoints',ROOT/'results') for p in root.rglob('*.pt')}
    before.update({str(p):digest(p) for root in (BASE/'reports',BASE/'runs',BASE/'execution_logs') for p in root.rglob('*') if p.is_file() and ID not in p.parts})
    write_json(RUN/'protected_before.json',before);shutil.copyfile(BASE/'configs/stage3_and78.yaml',RUN/'config.yaml');snap=RUN/'sources';snap.mkdir()
    for p in [*(BASE/'stage3').glob('*.py'),*(BASE/'tests').glob('test_stage3*.py'),ROOT/'src/models.py',ROOT/'src/coordinates.py',ROOT/'src/runtime.py',BASE/'data.py']:shutil.copyfile(p,snap/str(p.relative_to(ROOT)).replace('/','__'))
    shutil.copyfile(ROOT/'SPEC.md',RUN/'SPEC.snapshot.md')
    teacher,cp=frozen_teacher(TEACHER,device);student=SparseTranscoder(12,1024,12).to(device);tcstate=load_checkpoint(TC);student.load_state_dict(tcstate['model']);student.eval().requires_grad_(False)
    assert tcstate['identity']['teacher_sha256']==digest(TEACHER);d=load_frozen_data();verification=model_verification(teacher,student,d,device);write_json(RUN/'model_verification.json',verification)
    started=time.monotonic();banks,metadata,manifest=development(d,C['contexts']);np.savez_compressed(OUT/'development_contexts.npz',**banks);write_json(OUT/'development_manifest.json',manifest);write_json(OUT/'development_audit.json',metadata)
    teacher=teacher.double();student=student.double();p=torch.tensor(d['P'],device=device,dtype=torch.float64);q=torch.tensor(d['Q'],device=device,dtype=torch.float64)
    native={k:v.cpu().numpy() for k,v in original_coordinates(teacher,p,q).items()};tc={k:v.cpu().numpy() for k,v in transcoder_coordinates(student,p,q).items()};np.savez_compressed(OUT/'native_coordinates.npz',**native);np.savez_compressed(OUT/'transcoder_coordinates.npz',**tc)
    discovery=evaluate_bank(banks['discovery'],teacher,student,p,q);np.savez_compressed(OUT/'discovery_exact.npz',**discovery);groups,selection=select(discovery,native,tc,device);groups+=controls(groups,discovery,native['D2_eff']);hypotheses,parameters,clamp=predictions(groups,discovery,native)
    write_json(RUN/'fixed_hypotheses.json',{'selection':selection,'groups':hypotheses,'useful_thresholds':C['selection'],'heldout_forward_has_occurred':False});np.savez_compressed(RUN/'fixed_gate_parameters.npz',**parameters,clamp=clamp)
    frozen=digest(RUN/'fixed_hypotheses.json');write_json(RUN/'discovery_frozen.json',{'hypotheses_sha256':frozen,'gate_parameters_sha256':digest(RUN/'fixed_gate_parameters.npz'),'discovery_sha256':digest(OUT/'discovery_exact.npz'),'heldout_outcomes_used':False})
    print('DISCOVERY FROZEN',frozen,'groups',len(groups),flush=True)
    from .evaluate import evaluate_all,threshold_analysis
    threshold_analysis(discovery,native,tc,OUT/'discovery_thresholds')
    for split in ('heldout','stress'):
        held=evaluate_bank(banks[split],teacher,student,p,q);np.savez_compressed(OUT/f'{split}_exact.npz',**held);evaluate_all(split,held,hypotheses,parameters,clamp,native,teacher,p,q,device);threshold_analysis(held,native,tc,OUT/f'{split}_thresholds');print('EVALUATED',split,flush=True)
    assert digest(RUN/'fixed_hypotheses.json')==frozen
    assert all(digest(Path(path))==h for path,h in before.items())
    write_json(RUN/'execution.json',{'status':'complete','seconds':time.monotonic()-started,'GPU':torch.cuda.get_device_name(device),'torch':torch.__version__,'python':sys.version,'protected_files_verified':len(before),'hypotheses_sha256':frozen,'command':sys.argv,'models_trained':0,'reserved_final_file_accessed':False})
    from .report import produce
    produce();print('COMPLETE',OUT/'report.md',flush=True)

if __name__=='__main__':main()

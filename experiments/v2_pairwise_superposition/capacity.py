"""One approved V2 capacity trial: width 128, otherwise original protocol."""
import json,math,time,shutil,sys,subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models import OriginalMLP
from src.runtime import seed_everything,select_device,load_checkpoint,save_checkpoint,write_json
from src.coordinates import original_coordinates
from src.interactions import finite_interaction,corners_and_signs
from .run import BASE,ROOT,digest,loss_on,write_csv,validate
from .analysis import metrics,predict,group_metrics
from .balanced import distributions,rankings
from .data import RULES,PAIRS,keys,targets
OLD='first-teacher-20261027'
NEW='capacity128-20261027'
OUT=BASE/'reports'/NEW
RUN=BASE/'runs'/NEW
CK=BASE/'checkpoints'/NEW
SOURCE=BASE/'reports'/OLD

def train(c,paths,device):
    if (paths['runs']/'test_evaluated.json').exists():raise RuntimeError('Test already scored: no additional training of this run')
    seed_everything(c['seeds']['teacher'],c['training']['deterministic'])
    data=load_frozen_data();t=c['training'];model=OriginalMLP(12,128,12).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    generator=torch.Generator().manual_seed(c['seeds']['teacher'])
    p,q=[torch.as_tensor(data[key],device=device) for key in ('P','Q')]
    cache={name:(torch.as_tensor(data['a'][data[name]],dtype=torch.float32,device=device)@p.T,
                 torch.as_tensor(data['g'][data[name]],device=device)@q.T) for name in ('train','validation')}
    identity={'config_sha256':digest(paths['runs']/'config.yaml'),'dataset_sha256':digest(paths['data']/'dataset.npz')}
    first,best,best_epoch,stale,early_reference=1,float('inf'),None,0,float('inf');history=[]
    last=paths['checkpoints']/'last.pt'
    if last.exists():
        cp=load_checkpoint(last)
        if cp['identity']!=identity:raise RuntimeError('Checkpoint identity differs')
        model.load_state_dict(cp['model']);optimizer.load_state_dict(cp['optimizer']);generator.set_state(cp['shuffle_rng'])
        first,best,best_epoch,history,stale,early_reference=cp['epoch']+1,cp['best_mse'],cp['best_epoch'],cp['history'],cp['stale'],cp['early_reference']
    start_time=time.monotonic()
    for epoch in range(first,t['max_epochs']+1):
        if stale>=t['early_stopping_patience']:break
        epoch_start=time.monotonic();model.train()
        lr=t['minimum_learning_rate']+(t['learning_rate']-t['minimum_learning_rate'])*(1+math.cos(math.pi*(epoch-1)/max(1,t['max_epochs']-1)))/2
        for group in optimizer.param_groups:group['lr']=lr
        x,y=cache['train'];order=torch.randperm(len(x),generator=generator).to(device);total=0
        for start in range(0,len(order),t['batch_size']):
            idx=order[start:start+t['batch_size']];optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.mse_loss(model(x[idx]),y[idx])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite teacher loss')
            loss.backward()
            if any(param.grad is not None and not torch.isfinite(param.grad).all() for param in model.parameters()):raise RuntimeError('Non-finite gradient')
            optimizer.step();total+=loss.item()*len(idx)
        model.eval();val=loss_on(model,*cache['validation'],c['evaluation']['batch_size'])
        improved=val<best
        if improved:best,best_epoch=val,epoch
        if val<early_reference-t['early_stopping_min_delta']:early_reference,stale=val,0
        else:stale+=1
        row={'epoch':epoch,'train_online_mse':total/len(x),'validation_mse':val,'best_epoch':best_epoch,'learning_rate':lr,'seconds':time.monotonic()-epoch_start,'stale':stale}
        history.append(row)
        cp={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_mse':best,
            'identity':identity,'config':c,'history':history,'shuffle_rng':generator.get_state(),'stale':stale,'early_reference':early_reference}
        if improved:save_checkpoint(paths['checkpoints']/'best.pt',cp)
        save_checkpoint(last,cp);write_json(paths['runs']/'history.json',history)
        if epoch==1 or epoch%50==0:print('TRAIN',json.dumps(row),flush=True)
    write_json(paths['runs']/'training_complete.json',{'epochs':history[-1]['epoch'],'best_epoch':best_epoch,'best_validation_mse':best,
               'elapsed_this_invocation_seconds':time.monotonic()-start_time,'stop_reason':'validation_patience' if stale>=t['early_stopping_patience'] else 'max_epoch_budget',
               'no_test_or_contrast_metrics_used':True})

def compare(m,d,device,phase):
    allmetrics={};scores={};dist={}
    for split in ('train','validation'):
        g=d['g'][d[split]];a=d['a'][d[split]]
        with np.load(SOURCE/f'predictions_{split}.npz') as old:yp=old['output'].copy()
        yn=predict(m,a,d['P'],device,2048)['output']
        np.savez_compressed(OUT/f'predictions_{split}.npz',output=yn,semantic=yn.astype(float)@d['Q'].astype(float),vector_key=keys(a))
        for label,y in [('original',yp),('capacity',yn)]:
            allmetrics[f'{label}_{split}']=metrics(y,g,d['Q'],.5);scores[label,split]=y.astype(float)@d['Q'].astype(float)
            dist[f'{label}_{split}']=distributions(scores[label,split],g)
        write_json(OUT/f'capacity_groups_{split}.json',group_metrics(a,yn,g,d['Q'],.5))
    write_json(OUT/'metrics.json',allmetrics);write_json(OUT/'score_distributions.json',dist)
    write_csv(OUT/'per_rule.csv',({'model_population':k,**v} for k,mets in allmetrics.items() for v in mets['per_rule']))
    # Historical constructed probes, excluding ALL rows that coincide with old test.
    oldtest=set(map(int,keys(d['a'][d['test']])))
    with np.load(SOURCE/'constructed_probes.npz') as p:bank={k:p[k] for k in p.files}
    keep=np.array([int(k) not in oldtest for k in keys(bank['a'])]);a=bank['a'][keep];g=bank['g'][keep]
    yn=predict(m,a,d['P'],device,2048)['output'];yp=bank['output'][keep];density=a.sum(1)
    probes={}
    for label,y in [('original',yp),('capacity',yn)]:
        for regime,mask in [('density_3_to_6',(density>=3)&(density<=6)),('OOD_density',(density<3)|(density>6))]:probes[label+'_'+regime]=metrics(y[mask],g[mask],d['Q'],.5)
    write_json(OUT/'controlled_metrics.json',{'policy':'historical diagnostic probes, old-test-overlap rows excluded; never used for selection','excluded_memberships':int((~keep).sum()),'metrics':probes})
    np.savez_compressed(OUT/'controlled_predictions.npz',a=a,g=g,original=yp,capacity=yn)
    # Complete pair/triple contexts only when every corner avoids historical test.
    teacher=OriginalMLP(12,64,12).to(device).double();teacher.load_state_dict(load_checkpoint(BASE/'checkpoints'/OLD/'best.pt')['model'])
    capacity=OriginalMLP(12,128,12).to(device).double();capacity.load_state_dict(m.state_dict())
    models={'original':teacher,'capacity':capacity};rows=[];arrays={};maxres=0
    for pi,indices in enumerate(list(PAIRS)+[(1,2),(0,1,2)]):
        bases=torch.tensor(bank['bases'],device=device,dtype=torch.float64)
        corners,signs=corners_and_signs(bases,indices)
        good=np.array([not any(int(k) in oldtest for k in keys(v)) for v in corners.cpu().numpy().astype(np.uint8)])
        bases=bases[torch.tensor(good,device=device)];assert len(bases)>0
        for label,model in models.items():
            coords=original_coordinates(model,torch.tensor(d['P'],device=device,dtype=torch.float64),torch.tensor(d['Q'],device=device,dtype=torch.float64))
            result=finite_interaction(bases,indices,coords)
            # Direct network forward check, including embeddings, not just coordinate algebra.
            with torch.no_grad():direct=(model((result['corners']@torch.tensor(d['P'].T,device=device,dtype=torch.float64)).reshape(-1,12)).reshape(len(bases),len(signs),12)@torch.tensor(d['Q'],device=device,dtype=torch.float64)*signs[None,:,None]).sum(1)
            torch.testing.assert_close(result['value'],direct,atol=1e-9,rtol=1e-8)
            maxres=max(maxres,float((result['value']-direct).abs().max()))
            ground=(targets(result['corners'].cpu().numpy())*signs.cpu().numpy()[None,:,None]).sum(1)
            for name in ('contributions','value','positive_sum','negative_sum','corners'):arrays[f'{label}_p{pi}_{name}']=result[name].cpu().numpy()
            arrays[f'p{pi}_ground_truth']=ground
            for j,rule in enumerate(RULES):rows.append({'model':label,'indices':str(indices),'rule':rule,'n_contexts':len(bases),'mean_signed_interaction':float(result['value'][:,j].mean()),'mean_positive_sum':float(result['positive_sum'][:,j].mean()),'mean_negative_sum':float(result['negative_sum'][:,j].mean()),'mean_absolute_target_error':float(np.abs(result['value'][:,j].cpu().numpy()-ground[:,j]).mean())})
    np.savez_compressed(OUT/'signed_interactions.npz',**arrays);write_csv(OUT/'interactions.csv',rows)
    write_json(OUT/'interaction_checks.json',{'max_signed_sum_vs_direct_network_residual':maxres,'old_test_corners_excluded':True,'pairs_and_shared_input_triple':True})
    plots(d,scores,allmetrics,dist)
    write_json(OUT/'comparison_ready.json', {'complete':True})

def plots(d,scores,mets,dist):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for j,ax in enumerate(axes.flat):
        g=d['g'][d['validation'],j].astype(bool)
        for label,color in [('original','C0'),('capacity','C1')]:
            for positive,style in [(True,'-'),(False,'--')]:
                v=np.sort(scores[label,'validation'][g if positive else ~g,j]);ax.plot(v,np.arange(1,len(v)+1)/len(v),color=color,ls=style,label=label+(' positive' if positive else ' negative'))
        ax.axvline(.5,color='black',lw=.6);ax.set_title(RULES[j]);ax.set_ylabel('empirical CDF')
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'score_cdfs.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label in ('original','capacity'):
        v=mets[label+'_validation']['per_rule'];axes[0].plot([r['recall'] for r in v],'o-',label=label);axes[1].plot([r['false_positive_rate'] for r in v],'o-',label=label)
    for ax,title in zip(axes,['Validation recall','Validation FPR']):ax.set_title(title);ax.set_xticks(range(9),[f'g{j}' for j in range(9)]);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'recall_fpr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label,rundir in [('original',BASE/'runs'/OLD),('capacity',RUN)]:
        h=json.loads((rundir/'history.json').read_text());axes[0].plot([r['epoch'] for r in h],[r['validation_mse'] for r in h],label=label)
        for j in range(8):
            rr=dist[label+'_validation'][j]['ranking'];axes[1].plot(rr['recall'],rr['precision'],alpha=.55,label=f'{label} g{j}')
    axes[0].set_title('Validation full-output MSE (selection)');axes[0].set_yscale('log');axes[0].legend();axes[1].set_title('Primitive precision–recall diagnostics');axes[1].set_xlabel('Recall');axes[1].set_ylabel('Precision');fig.tight_layout();fig.savefig(OUT/'learning_and_pr.png',dpi=160);plt.close(fig)


def load_frozen_data():
    with np.load(BASE/'data'/OLD/'dataset.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}


def validate_control(c,baseline):
    expected=json.loads(json.dumps(baseline));expected['dimensions']['hidden']=128
    if c!=expected:raise ValueError('Only hidden width 64→128 may change')
    restored=json.loads(json.dumps(c));restored['dimensions']['hidden']=64;validate(restored)
    if c['dimensions']!={'semantic_input':20,'input':12,'hidden':128,'output':12,'semantic_output':9}:raise ValueError('Invalid width128 shapes')


@torch.no_grad()
def nearest_labels(query,truth,reference,reftruth,self_indices=None,batch=256):
    """Exact Euclidean nearest same/opposite label in frozen TRAIN reference only.

    Each semantic output is a separate binary neighborhood; column 9 denotes
    same/different complete target vector. Train self-reference is excluded.
    """
    result={k:[] for k in ('opposite_distance','same_distance','opposite_index')}
    for start in range(0,len(query),batch):
        stop=min(start+batch,len(query));x=query[start:stop];g=truth[start:stop]
        distances=torch.cdist(x,reference,p=2)
        if self_indices is not None:distances[torch.arange(len(x),device=x.device),self_indices[start:stop]]=float('inf')
        opposite=[];same=[];indices=[]
        for j in range(10):
            match=(g[:,j,None]==reftruth[None,:,j]) if j<9 else (g[:,None,:]==reftruth[None,:,:]).all(-1)
            od,oi=distances.masked_fill(match,float('inf')).min(1)
            sd=distances.masked_fill(~match,float('inf')).min(1).values
            opposite.append(od);same.append(sd);indices.append(oi)
        for name,values in [('opposite_distance',opposite),('same_distance',same),('opposite_index',indices)]:result[name].append(torch.stack(values,1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def quantiles(v):
    finite=v[np.isfinite(v)]
    return {'n':len(v),'n_finite':len(finite),'quantiles_0_10_25_50_75_90_100':np.quantile(finite,[0,.1,.25,.5,.75,.9,1]).tolist() if len(finite) else None}


def geometry(d,device):
    p=torch.tensor(d['P'],device=device,dtype=torch.float64);train_a=d['a'][d['train']]
    ref=torch.tensor(train_a,device=device,dtype=torch.float64)@p.T;rg=torch.tensor(d['g'][d['train']],device=device)
    summary={};pairs=[];started=time.monotonic()
    for split in ('train','validation'):
        a=d['a'][d[split]];g=d['g'][d[split]]
        x=torch.tensor(a,device=device,dtype=torch.float64)@p.T;tg=torch.tensor(g,device=device)
        near=nearest_labels(x,tg,ref,rg,torch.arange(len(a),device=device) if split=='train' else None)
        assert np.isfinite(near['opposite_distance']).all()
        np.savez_compressed(OUT/f'geometry_{split}.npz',**near,vector_key=keys(a))
        with np.load(SOURCE/f'predictions_{split}.npz') as cache:
            np.testing.assert_array_equal(cache['vector_key'],keys(a));score=cache['semantic']
        summary[split]={}
        for j,rule in enumerate(list(RULES)+['complete_target_vector']):
            if j<9:
                positive=g[:,j].astype(bool);wrong=(score[:,j]>=.5)!=positive
                groups={'all':np.ones(len(a),bool),'positive_false_negative':positive&wrong,'positive_correct':positive&~wrong,'negative_false_positive':~positive&wrong,'negative_correct':~positive&~wrong}
            else:
                wrong=((score>=.5)!=g.astype(bool)).any(1);groups={'all':np.ones(len(a),bool),'baseline_error':wrong,'baseline_correct':~wrong}
            od,sd=near['opposite_distance'][:,j],near['same_distance'][:,j]
            summary[split][rule]={label:{'opposite_distance':quantiles(od[mask]),'same_distance':quantiles(sd[mask]),'opposite_to_same_ratio':quantiles(np.divide(od[mask],sd[mask],out=np.full(int(mask.sum()),np.nan),where=np.isfinite(sd[mask])&(sd[mask]>0))),'fraction_opposite_closer_than_same':float((od[mask&np.isfinite(sd)]<sd[mask&np.isfinite(sd)]).mean()) if (mask&np.isfinite(sd)).any() else None} for label,mask in groups.items()}
            for row in np.argsort(od)[:20]:
                ri=int(near['opposite_index'][row,j]);pairs.append({'population':split,'rule':rule,'query_key':int(keys(a[row:row+1])[0]),'reference_train_key':int(keys(train_a[ri:ri+1])[0]),'distance':float(od[row]),'query_active':str(np.flatnonzero(a[row]).tolist()),'reference_active':str(np.flatnonzero(train_a[ri]).tolist()),'query_labels':str(g[row].astype(int).tolist()),'reference_labels':str(d['g'][d['train'][ri]].astype(int).tolist()),'baseline_error':bool(wrong[row])})
        print('GEOMETRY',split,'complete',flush=True)
    gram=d['P'].astype(float).T@d['P'].astype(float)
    write_json(OUT/'pretraining_geometry.json',{'reference':'training examples only; same-row self matches excluded; float64 CUDA exact Euclidean nearest neighbor','groups':summary,'P_gram':gram.tolist(),'P_mutual_coherence':float(np.abs(gram-np.diag(np.diag(gram))).max()),'seconds':time.monotonic()-started,'interpretation':'observational association, confounded by input density and context; not proof of unavoidable target ambiguity'})
    write_csv(OUT/'closest_conflicting_label_pairs.csv',pairs)


def main():
    device=select_device('cuda');baseline=yaml.safe_load((BASE/'runs'/OLD/'config.yaml').read_text())
    c=yaml.safe_load((BASE/'configs'/'capacity128.yaml').read_text());validate_control(c,baseline)
    for p in (OUT,RUN,CK):p.mkdir(parents=True,exist_ok=False)
    protected=[p for root in (BASE/'checkpoints',ROOT/'results') for p in root.rglob('*.pt') if CK not in p.parents]
    # Preserve every preexisting V2 run artifact, not just model weights.
    protected += [p for root in (BASE/'runs',BASE/'reports',BASE/'execution_logs') for p in root.rglob('*') if p.is_file() and NEW not in p.parts]
    protected += [BASE/'data'/OLD/'dataset.npz',BASE/'data'/OLD/'split_manifest.json',ROOT/'SPEC.md',ROOT/'CODEX_PROMPT.md']
    hashes={str(p):digest(p) for p in protected};write_json(RUN/'protected_before.json',hashes)
    shutil.copyfile(BASE/'configs'/'capacity128.yaml',RUN/'config.yaml')
    for name in ('SPEC.md','CODEX_PROMPT.md'):shutil.copyfile(ROOT/name,RUN/(name+'.snapshot'))
    source=[BASE/'capacity.py',BASE/'capacity_report.py',BASE/'analysis.py',BASE/'data.py',BASE/'balanced.py',BASE/'run.py',ROOT/'src/models.py',ROOT/'src/runtime.py',ROOT/'src/coordinates.py',ROOT/'src/interactions.py']
    snap=RUN/'source_snapshot';snap.mkdir()
    for p in source:shutil.copyfile(p,snap/(str(p.relative_to(ROOT)).replace('/','__')))
    d=load_frozen_data();assert d['P'].shape==(12,20) and d['Q'].shape==(12,9) and len(d['a'])==20000
    seed_everything(3105,True);initial=OriginalMLP(12,128,12).to(device)
    assert initial.W1.weight.shape==(128,12) and initial.W2.weight.shape==(12,128)
    save_checkpoint(CK/'initial.pt',{'model':initial.state_dict(),'seed':3105})
    write_json(RUN/'provenance.json',{'command':sys.argv,'python':sys.version,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(device),'dataset_sha256':digest(BASE/'data'/OLD/'dataset.npz'),'split_manifest_sha256':digest(BASE/'data'/OLD/'split_manifest.json'),'source_hashes':{str(p):digest(p) for p in source},'only_config_change':'dimensions.hidden:64→128','architecture':[12,128,12],'initialization':'same seed and constructor; changed tensor shapes mean weights and fan-in initialization ranges differ; no nested or identical initial weights claimed','final_population_accessed':False,'historical_test_scored':False})
    geometry(d,device)
    paths={'runs':RUN,'reports':OUT,'checkpoints':CK,'data':BASE/'data'/OLD}
    train(c,paths,device)
    cp=load_checkpoint(CK/'best.pt');m=OriginalMLP(12,128,12).to(device);m.load_state_dict(cp['model'])
    assert cp['config']==c
    compare(m,d,device,None)
    from .capacity_report import produce
    produce(d)
    changed=[p for p,h in hashes.items() if digest(Path(p))!=h]
    if changed:raise RuntimeError('Protected artifacts changed: '+str(changed))
    write_json(RUN/'complete.json',{'status':'complete','protected_files_verified':len(hashes),'historical_test_scored':False,'reserved_final_accessed':False,'single_training_variant':True})
    print('COMPLETE',OUT/'report.md',flush=True)

if __name__=='__main__':main()

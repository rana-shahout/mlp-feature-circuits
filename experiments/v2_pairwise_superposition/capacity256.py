"""Final authorized V2 capacity trial, width256, original loss; no transcoder."""
import json,math,time,shutil,sys
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models import OriginalMLP
from src.runtime import seed_everything,select_device,load_checkpoint,save_checkpoint,write_json
from src.coordinates import original_coordinates
from src.interactions import decompose_corners
from .run import BASE,ROOT,digest,loss_on,write_csv,validate
from .capacity import load_frozen_data
from .analysis import metrics,predict,group_metrics
from .balanced import distributions
from .data import RULES,PAIRS,keys,targets
OLD='first-teacher-20261027'
PREVIOUS='capacity128-20261027'
NEW='capacity256-20261027'
OUT=BASE/'reports'/NEW
RUN=BASE/'runs'/NEW
CK=BASE/'checkpoints'/NEW
WIDTH_RUNS={64:OLD,128:PREVIOUS,256:NEW}

def train(c,paths,device):
    if (paths['runs']/'test_evaluated.json').exists():raise RuntimeError('Test already scored: no additional training of this run')
    seed_everything(c['seeds']['teacher'],c['training']['deterministic'])
    data=load_frozen_data();t=c['training'];model=OriginalMLP(12,256,12).to(device)
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

def validate_control(c,previous):
    expected=json.loads(json.dumps(previous));expected['dimensions']['hidden']=256
    if c!=expected:raise ValueError('Only hidden width128→256 is authorized')
    restored=json.loads(json.dumps(c));restored['dimensions']['hidden']=64;validate(restored)


def signed_statistics(v):
    absolute=np.abs(v);total=absolute.sum(-1);squared=(v*v).sum(-1)
    p=np.divide(absolute,total[...,None],out=np.zeros_like(absolute),where=total[...,None]>0)
    effective=np.divide(total**2,squared,out=np.zeros_like(total),where=squared>0)
    entropy=np.exp(-(p*np.log(np.maximum(p,np.finfo(float).tiny))).sum(-1));entropy[total==0]=0
    top=np.argmax(absolute,axis=-1);topvalue=np.take_along_axis(v,top[...,None],axis=-1)[...,0]
    return {'net':v.sum(-1),'positive':np.maximum(v,0).sum(-1),'negative':np.minimum(v,0).sum(-1),
        'effective_units':effective,'effective_fraction_of_width':effective/v.shape[-1],
        'entropy_effective_units':entropy,'entropy_effective_fraction_of_width':entropy/v.shape[-1],
        'nonzero_units':(absolute>0).sum(-1),'nonzero_fraction_of_width':(absolute>0).mean(-1),
        'top1_absolute_share':p.max(-1),'top5_absolute_share':np.sort(p,axis=-1)[...,-min(5,v.shape[-1]):].sum(-1),
        'cancellation':np.divide(total-np.abs(v.sum(-1)),total,out=np.zeros_like(total),where=total>0),
        'top1_signed_residual':np.abs(v.sum(-1)-topvalue),'top1_neuron':top}


@torch.no_grad()
def native_checks(model,d,device):
    """Reuse precisely the historical-test-excluded width128 context/corner bank."""
    saved=BASE/'reports'/PREVIOUS/'signed_interactions.npz';arrays={};summary=[];context_rows=[];units=[];maxres=0
    oldtest=set(map(int,keys(d['a'][d['test']])))
    native=OriginalMLP(12,256,12).to(device).double();native.load_state_dict(model.state_dict())
    p=torch.tensor(d['P'],device=device,dtype=torch.float64);q=torch.tensor(d['Q'],device=device,dtype=torch.float64);coords=original_coordinates(native,p,q)
    with np.load(saved) as f:
        for pi,indices in enumerate(list(PAIRS)+[(1,2),(0,1,2)]):
            a=f[f'original_p{pi}_corners'];np.testing.assert_array_equal(a,f[f'capacity_p{pi}_corners'])
            assert all(int(k) not in oldtest for k in keys(a.reshape(-1,20).astype(np.uint8)))
            corners=torch.tensor(a,device=device,dtype=torch.float64)
            signs=torch.tensor([(-1)**(len(indices)-int(v.sum())) for v in a[0][:,list(indices)]],device=device,dtype=torch.float64)
            result=decompose_corners(corners,signs,coords)
            direct=(native((corners@p.T).reshape(-1,12)).reshape(len(a),len(signs),12)@q*signs[None,:,None]).sum(1)
            torch.testing.assert_close(result['value'],direct,atol=1e-9,rtol=1e-8)
            residual=result['value']-direct;maxres=max(maxres,float(residual.abs().max()))
            ground=(targets(a)*signs.cpu().numpy()[None,:,None]).sum(1);np.testing.assert_array_equal(ground,f[f'p{pi}_ground_truth'])
            arrays[f'p{pi}_corners']=a;arrays[f'p{pi}_ground_truth']=ground;arrays[f'p{pi}_indices']=np.array(indices)
            arrays[f'w256_p{pi}_direct']=direct.cpu().numpy();arrays[f'w256_p{pi}_identity_residual']=residual.cpu().numpy()
            density=a.sum(-1);inside=(density.min(1)>=3)&(density.max(1)<=6)
            for width,contribution,value in [(64,f[f'original_p{pi}_contributions'],f[f'original_p{pi}_value']),(128,f[f'capacity_p{pi}_contributions'],f[f'capacity_p{pi}_value']),(256,result['contributions'].cpu().numpy(),result['value'].cpu().numpy())]:
                assert contribution.shape==(len(a),9,width)
                np.testing.assert_allclose(contribution.sum(-1),value,atol=1e-9,rtol=1e-8)
                arrays[f'w{width}_p{pi}_contributions']=contribution;arrays[f'w{width}_p{pi}_value']=value
                for j,rule in enumerate(RULES):
                    v=contribution[:,j,:];stats=signed_statistics(v)
                    for regime,mask in [('all',np.ones(len(a),bool)),('every_corner_density_3_to_6',inside),('some_corner_OOD_density',~inside)]:
                        if not mask.any():continue
                        summary.append({'width':width,'indices':str(indices),'rule':rule,'regime':regime,'n_contexts':int(mask.sum()),'target_interaction_mae':float(np.abs(value[mask,j]-ground[mask,j]).mean()),**{k:float(vv[mask].mean()) for k,vv in stats.items() if k!='top1_neuron'}})
                    for ci in range(len(a)):context_rows.append({'width':width,'indices':str(indices),'rule':rule,'context_index':ci,'all_corners_density_supported':bool(inside[ci]),'ground_truth':float(ground[ci,j]),'target_error':float(value[ci,j]-ground[ci,j]),**{k:float(vv[ci]) for k,vv in stats.items()}})
                    for neuron in range(width):units.append({'width':width,'indices':str(indices),'rule':rule,'neuron':neuron,'mean_signed':float(v[:,neuron].mean()),'mean_absolute':float(np.abs(v[:,neuron]).mean()),'fraction_positive_contexts':float((v[:,neuron]>0).mean()),'fraction_negative_contexts':float((v[:,neuron]<0).mean())})
    np.savez_compressed(OUT/'signed_interactions_all_widths.npz',**arrays)
    write_csv(OUT/'interaction_summary.csv',summary);write_csv(OUT/'interaction_contexts.csv',context_rows);write_csv(OUT/'all_neuron_contributions.csv',units)
    write_json(OUT/'interaction_checks.json',{'new_width_max_sum_vs_direct_forward_error':maxres,'all_three_widths_identical_corners':True,'old_test_corners_excluded':True,'prior_width_checks_source':str(BASE/'reports'/PREVIOUS/'interaction_checks.json'),'old_widths_use_frozen_previously_validated_decompositions':True})
    return summary


def evaluate(model,d,device):
    allmetrics={};dist={};hidden={}
    for split in ('train','validation'):
        a=d['a'][d[split]];g=d['g'][d[split]];key=keys(a)
        for width in (64,128,256):
            if width==256:
                pred=predict(model,a,d['P'],device,2048,hidden=split=='train');y=pred['output']
                if split=='train':
                    h=pred['hidden'];hidden={'mean_active':float((h>0).sum(1).mean()),'mean_active_fraction':float((h>0).mean()),'dead_units':int(((h>0).sum(0)==0).sum()),'activation_prevalence':(h>0).mean(0).tolist()}
                    np.savez_compressed(OUT/'hidden_train.npz',hidden=h,preactivation=pred['preactivation'],vector_key=key)
                np.savez_compressed(OUT/f'predictions_{split}.npz',output=y,semantic=y.astype(float)@d['Q'].astype(float),vector_key=key)
            else:
                with np.load(BASE/'reports'/WIDTH_RUNS[width]/f'predictions_{split}.npz') as f:
                    np.testing.assert_array_equal(f['vector_key'],key);y=f['output'].copy()
            allmetrics[f'w{width}_{split}']=metrics(y,g,d['Q'],.5);dist[f'w{width}_{split}']=distributions(y.astype(float)@d['Q'].astype(float),g)
            write_json(OUT/f'w{width}_groups_{split}.json',group_metrics(a,y,g,d['Q'],.5))
    write_json(OUT/'metrics.json',allmetrics);write_json(OUT/'score_distributions.json',dist);write_json(OUT/'hidden_activation_summary.json',hidden)
    write_csv(OUT/'per_rule.csv',({'model_population':label,**r} for label,m in allmetrics.items() for r in m['per_rule']))
    with np.load(BASE/'reports'/PREVIOUS/'controlled_predictions.npz') as f:bank={k:f[k] for k in f.files}
    a=bank['a'];g=bank['g'];oldtest=set(map(int,keys(d['a'][d['test']])))
    assert all(int(k) not in oldtest for k in keys(a))
    new=predict(model,a,d['P'],device,2048)['output'];density=a.sum(1);probes={}
    for width,y in [(64,bank['original']),(128,bank['capacity']),(256,new)]:
        for label,mask in [('density_3_to_6',(density>=3)&(density<=6)),('OOD_density',(density<3)|(density>6))]:probes[f'w{width}_{label}']=metrics(y[mask],g[mask],d['Q'],.5)
    write_json(OUT/'controlled_metrics.json',probes)
    np.savez_compressed(OUT/'controlled_predictions_all_widths.npz',a=a,g=g,w64=bank['original'],w128=bank['capacity'],w256=new)
    interactions=native_checks(model,d,device)
    from .capacity256_report import produce
    produce(d,allmetrics,dist,probes,interactions)


def main():
    device=select_device('cuda');c=yaml.safe_load((BASE/'configs/capacity256.yaml').read_text());previous=yaml.safe_load((BASE/'runs'/PREVIOUS/'config.yaml').read_text());validate_control(c,previous)
    for p in (OUT,RUN,CK):p.mkdir(parents=True,exist_ok=False)
    protected=[p for root in (BASE/'checkpoints',ROOT/'results') for p in root.rglob('*.pt') if CK not in p.parents]
    protected += [p for root in (BASE/'runs',BASE/'reports',BASE/'execution_logs') for p in root.rglob('*') if p.is_file() and NEW not in p.parts]
    protected += [BASE/'data'/OLD/'dataset.npz',BASE/'data'/OLD/'split_manifest.json',ROOT/'SPEC.md',ROOT/'CODEX_PROMPT.md']
    hashes={str(p):digest(p) for p in protected};write_json(RUN/'protected_before.json',hashes)
    shutil.copyfile(BASE/'configs/capacity256.yaml',RUN/'config.yaml')
    snap=RUN/'source_snapshot';snap.mkdir()
    sources=[BASE/'capacity256.py',BASE/'capacity256_report.py',BASE/'capacity.py',BASE/'balanced.py',BASE/'run.py',BASE/'analysis.py',BASE/'data.py',ROOT/'src/models.py',ROOT/'src/runtime.py',ROOT/'src/coordinates.py',ROOT/'src/interactions.py',*(BASE/'tests').glob('*.py')]
    for p in sources:shutil.copyfile(p,snap/(str(p.relative_to(ROOT)).replace('/','__')))
    for p in (ROOT/'SPEC.md',ROOT/'CODEX_PROMPT.md'):shutil.copyfile(p,RUN/(p.name+'.snapshot'))
    seed_everything(3105,True);initial=OriginalMLP(12,256,12).to(device)
    assert initial.W1.weight.shape==(256,12) and initial.W2.weight.shape==(12,256)
    save_checkpoint(CK/'initial.pt',{'model':initial.state_dict(),'seed':3105})
    write_json(RUN/'provenance.json',{'command':sys.argv,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'GPU':torch.cuda.get_device_name(device),'source_hashes':{str(p):digest(p) for p in sources},'dataset_sha256':digest(BASE/'data'/OLD/'dataset.npz'),'split_manifest_sha256':digest(BASE/'data'/OLD/'split_manifest.json'),'only_config_change':'dimensions.hidden:128→256','architecture':[12,256,12],'initialization':'OriginalMLP standard nn.Linear initialization, seed3105; shapes and W2 fan-in scaling differ across widths, no identical/nested weights claimed','reserved_final_accessed':False,'historical_test_scored':False})
    paths={'runs':RUN,'checkpoints':CK,'data':BASE/'data'/OLD};train(c,paths,device)
    cp=load_checkpoint(CK/'best.pt');assert cp['config']==c
    model=OriginalMLP(12,256,12).to(device);model.load_state_dict(cp['model']);evaluate(model,load_frozen_data(),device)
    last=load_checkpoint(CK/'last.pt');baseline=load_checkpoint(BASE/'checkpoints'/OLD/'last.pt')
    generator=torch.Generator().manual_seed(3105)
    for _ in range(last['epoch']):torch.randperm(14000,generator=generator)
    assert torch.equal(generator.get_state(),last['shuffle_rng'])
    if last['epoch']==baseline['epoch']:assert torch.equal(last['shuffle_rng'],baseline['shuffle_rng'])
    assert all(a['learning_rate']==b['learning_rate'] for a,b in zip(last['history'],baseline['history']))
    seed_everything(3105,True);reconstructed=OriginalMLP(12,256,12)
    assert all(torch.equal(v.cpu(),reconstructed.state_dict()[k]) for k,v in load_checkpoint(CK/'initial.pt')['model'].items())
    assert all(digest(Path(p))==h for p,h in hashes.items())
    write_json(RUN/'complete.json',{'status':'complete','protected_files_verified':len(hashes),'shuffle_reconstruction_matches':True,'LRs_match_baseline':True,'initialization_reconstruction_matches':True,'reserved_final_accessed':False,'historical_test_scored':False,'single_variant':True,'report_sha256':digest(OUT/'report.md')})
    print('COMPLETE',OUT/'report.md',flush=True)

if __name__=='__main__':main()

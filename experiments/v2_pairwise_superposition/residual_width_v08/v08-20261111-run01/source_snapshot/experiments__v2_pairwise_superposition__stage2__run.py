"""CUDA preparation and six fixed, independent transcoder trainings."""
import argparse,json,math,shutil,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models import SparseTranscoder
from src.runtime import select_device,seed_everything,write_json,save_checkpoint,load_checkpoint
from ..run import BASE,ROOT,digest,write_csv
from ..capacity256 import load_frozen_data
from ..data import keys,targets
from ..analysis import metrics
from .core import frozen_teacher,infer,loss_terms,name,fidelity

CONFIG=BASE/'configs/stage2_transcoders.yaml'
C=yaml.safe_load(CONFIG.read_text())
ID=C['run_id']
RUN=BASE/'runs'/ID
OUT=BASE/'reports'/ID
CK=BASE/'checkpoints'/ID
SOURCE=BASE/'reports'/C['teacher_run']
TEACHER=ROOT/C['teacher_checkpoint']


def validate(c):
    if c['widths']!=[512,1024] or c['lambdas']!=[0.,.0001,.001]:raise ValueError('Only the six approved configurations')
    t=c['training']
    if t['checkpoint_selection']!='validation_teacher_full_output_mse' or t['reconstruction_reduction']!='mean_examples_sum_12_output_coordinates' or t['l1_reduction']!='mean_examples_sum_units':raise ValueError('Unexpected loss or checkpoint policy')
    if c['evaluation']['threshold']!=.5 or t['decoder_constraint']!='unit_l2_columns_after_every_optimizer_step':raise ValueError('Fixed thresholds and decoder norm required')
    if c['teacher_checkpoint']!='experiments/v2_pairwise_superposition/checkpoints/capacity256-20261027/best.pt':raise ValueError('Wrong approved teacher')


def protected():
    files=[p for root in (BASE/'checkpoints',ROOT/'results') for p in root.rglob('*.pt') if CK not in p.parents]
    files += [p for root in (BASE/'runs',BASE/'reports',BASE/'execution_logs') for p in root.rglob('*') if p.is_file() and ID not in p.parts]
    files += [ROOT/C['dataset'],ROOT/C['split_manifest']]
    return {str(p):digest(p) for p in files}


def verify_frozen():
    record=json.loads((RUN/'teacher_verification.json').read_text())
    for field,path in [('checkpoint_sha256',TEACHER),('dataset_sha256',ROOT/C['dataset']),('split_manifest_sha256',ROOT/C['split_manifest'])]:
        if digest(path)!=record[field]:raise RuntimeError('Frozen teacher/data changed')
    if digest(RUN/'teacher_targets.npz')!=record['target_cache_sha256']:raise RuntimeError('Teacher cache changed')
    return record


def prepare(device):
    validate(C)
    for p in (RUN,OUT,CK):p.mkdir(parents=True,exist_ok=False)
    write_json(RUN/'protected_before.json',protected());shutil.copyfile(CONFIG,RUN/'config.yaml')
    sources=[*(BASE/'stage2').glob('*.py'),*(BASE/'tests').glob('test_stage2*.py'),ROOT/'src/models.py',ROOT/'src/runtime.py',ROOT/'src/coordinates.py',ROOT/'src/interactions.py',BASE/'analysis.py',BASE/'data.py']
    snap=RUN/'source_snapshot';snap.mkdir()
    for p in sources:shutil.copyfile(p,snap/str(p.relative_to(ROOT)).replace('/','__'))
    for p in (ROOT/'SPEC.md',ROOT/'docs/v2_stage2.md'):shutil.copyfile(p,RUN/(p.name+'.snapshot'))
    d=load_frozen_data();teacher,cp=frozen_teacher(TEACHER,device);seed_everything(4105,True)
    old=json.loads((BASE/'runs'/C['teacher_run']/'provenance.json').read_text())
    assert digest(ROOT/C['dataset'])==old['dataset_sha256'] and digest(ROOT/C['split_manifest'])==old['split_manifest_sha256']
    assert cp['identity']['dataset_sha256']==old['dataset_sha256']
    assert cp['identity']['config_sha256']==digest(BASE/'runs'/C['teacher_run']/'config.yaml')
    assert cp['config']==yaml.safe_load((BASE/'runs'/C['teacher_run']/'config.yaml').read_text())
    np.testing.assert_allclose(d['Q'].T@d['Q'],np.eye(9),atol=2e-7)
    assert d['P'].shape==(12,20) and d['Q'].shape==(12,9) and np.linalg.matrix_rank(d['P'])==12
    np.testing.assert_allclose(np.linalg.norm(d['P'],axis=0),1,atol=2e-7)
    manifest=json.loads((ROOT/C['split_manifest']).read_text())
    for split in ('train','validation'):
        np.testing.assert_array_equal(manifest['partitions'][split]['indices'],d[split]);np.testing.assert_array_equal(manifest['partitions'][split]['vector_keys'],keys(d['a'][d[split]]))
    cached={};reference=json.loads((SOURCE/'metrics.json').read_text())
    for split in ('train','validation'):
        x=torch.tensor(d['a'][d[split]],device=device,dtype=torch.float32)@torch.tensor(d['P'].T,device=device)
        output,_=infer(teacher,x);g=d['g'][d[split]];m=metrics(output,g,d['Q'],.5)
        with np.load(SOURCE/f'predictions_{split}.npz') as p:
            np.testing.assert_array_equal(p['vector_key'],keys(d['a'][d[split]]));np.testing.assert_allclose(output,p['output'],atol=2e-6,rtol=2e-5)
        for field in ('output_mse','semantic_mse','exact_vector_accuracy'):np.testing.assert_allclose(m[field],reference[f'w256_{split}'][field],atol=2e-7,rtol=2e-5)
        cached[split]=output;write_json(OUT/f'teacher_{split}_task.json',m)
    np.savez_compressed(RUN/'teacher_targets.npz',**cached)
    record={'checkpoint':str(TEACHER),'checkpoint_sha256':digest(TEACHER),'dataset_sha256':digest(ROOT/C['dataset']),'split_manifest_sha256':digest(ROOT/C['split_manifest']),'target_cache_sha256':digest(RUN/'teacher_targets.npz'),'architecture':[12,256,12],'config':cp['config'],'selected_epoch':cp['epoch'],'eval_mode':not teacher.training,'all_parameters_frozen':all(not p.requires_grad for p in teacher.parameters()),'teacher_validation':reference['w256_validation'],'teacher_training_provenance':old,'comparison_tolerance':{'absolute':2e-7,'relative':2e-5},'reserved_final_accessed':False}
    write_json(RUN/'teacher_verification.json',record)
    write_json(RUN/'provenance.json',{'command':sys.argv,'python':sys.version,'torch':torch.__version__,'CUDA':torch.version.cuda,'numpy':np.__version__,'GPU':torch.cuda.get_device_name(device),'source_hashes':{str(p):digest(p) for p in sources},'source_config_sha256':digest(CONFIG),'teacher_sha256':record['checkpoint_sha256'],'git_status':subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True).stdout})
    # CUDA float64 affine regression with intercept, fitted solely on frozen training outputs.
    xt=torch.tensor(d['a'][d['train']],device=device,dtype=torch.float64)@torch.tensor(d['P'].T,device=device,dtype=torch.float64)
    design=torch.cat([xt,torch.ones(len(xt),1,device=device,dtype=torch.float64)],1);assert torch.linalg.matrix_rank(design)==13
    target=torch.tensor(cached['train'],device=device,dtype=torch.float64)
    beta=torch.linalg.lstsq(design,target,driver='gels').solution;np.save(OUT/'affine_coefficients.npy',beta.cpu().numpy())
    result={}
    for split in ('train','validation'):
        x=torch.tensor(d['a'][d[split]],device=device,dtype=torch.float64)@torch.tensor(d['P'].T,device=device,dtype=torch.float64)
        y=(torch.cat([x,torch.ones(len(x),1,device=device,dtype=torch.float64)],1)@beta).cpu().numpy()
        result[split]=fidelity(y,cached[split],d['g'][d[split]],d['Q'])
        np.savez_compressed(OUT/f'affine_{split}.npz',output=y)
    write_json(OUT/'affine_baseline.json',result)
    print('TEACHER VERIFIED',record['checkpoint_sha256'],'validation',reference['w256_validation']['exact_vector_accuracy'],flush=True)


@torch.no_grad()
def validation(model,x,y):
    squared,l1,l0=0.,0.,0.
    for start in range(0,len(x),C['evaluation']['batch_size']):
        z=model.latents(x[start:start+C['evaluation']['batch_size']]);out=model.decoder(z);truth=y[start:start+len(z)]
        squared+=(out-truth).square().sum().item();l1+=z.sum().item();l0+=(z>0).sum().item()
    return {'mse':squared/y.numel(),'mean_l1_sum':l1/len(x),'mean_l0':l0/len(x)}


def fit(width,coefficient,xt,yt,xv,yv,folder,identity):
    """No Boolean targets or embeddings accepted: teacher full outputs only."""
    t=C['training'];seed_everything(C['student_seed'],True);m=SparseTranscoder(12,width,12).to(xt.device)
    save_checkpoint(folder/'initial.pt',{'model':m.state_dict(),'seed':C['student_seed']})
    optimizer=torch.optim.Adam(m.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);generator=torch.Generator().manual_seed(C['shuffle_seed'])
    best=early=float('inf');best_epoch=None;stale=0;history=[];started=time.monotonic()
    for epoch in range(1,t['epochs']+1):
        if stale>=t['patience']:break
        lr=t['minimum_learning_rate']+(t['learning_rate']-t['minimum_learning_rate'])*(1+math.cos(math.pi*(epoch-1)/(t['epochs']-1)))/2
        for group in optimizer.param_groups:group['lr']=lr
        m.train();order=torch.randperm(len(xt),generator=generator).to(xt.device);total_rec=total_l1=0.
        for start in range(0,len(order),t['batch_size']):
            idx=order[start:start+t['batch_size']];optimizer.zero_grad(set_to_none=True);z=m.latents(xt[idx]);loss,rec,l1=loss_terms(m.decoder(z),yt[idx],z,coefficient)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite imitation loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('Nonfinite gradient')
            optimizer.step();m.normalize_decoder();total_rec+=rec.item()*len(idx);total_l1+=l1.item()*len(idx)
        m.eval();val=validation(m,xv,yv)
        if not np.isfinite(val['mse']):raise RuntimeError('Nonfinite validation')
        improved=val['mse']<best
        if improved:best=val['mse'];best_epoch=epoch
        if val['mse']<early-t['min_delta']:early=val['mse'];stale=0
        else:stale+=1
        row={'epoch':epoch,'learning_rate':lr,'train_reconstruction_squared_l2_mean':total_rec/len(xt),'train_mean_l1_sum':total_l1/len(xt),'validation':val,'best_epoch':best_epoch,'stale':stale};history.append(row)
        state={'model':m.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_mse':best,'width':width,'lambda':coefficient,'identity':identity,'config':C,'history':history,'shuffle_rng':generator.get_state()}
        if improved:save_checkpoint(folder/'best.pt',state)
        save_checkpoint(folder/'last.pt',state);write_json(folder/'history.json',history)
        if epoch==1 or epoch%100==0:print('TRAIN',name(width,coefficient),json.dumps(row),flush=True)
    write_json(folder/'training_complete.json',{'epochs':history[-1]['epoch'],'selected_epoch':best_epoch,'selected_mse':best,'seconds':time.monotonic()-started,'stop_reason':'patience' if stale>=t['patience'] else 'budget'})
    m.load_state_dict(load_checkpoint(folder/'best.pt')['model']);m.eval()
    norms=m.decoder.weight.norm(dim=0);torch.testing.assert_close(norms,torch.ones_like(norms),atol=3e-7,rtol=3e-7)
    return m


def worker(width,device):
    validate(C);record=verify_frozen();d=load_frozen_data();p=torch.tensor(d['P'],device=device)
    with np.load(RUN/'teacher_targets.npz') as cache:ys={s:torch.tensor(cache[s],device=device) for s in ('train','validation')}
    xs={s:torch.tensor(d['a'][d[s]],device=device,dtype=torch.float32)@p.T for s in ('train','validation')}
    for coefficient in C['lambdas']:
        folder=CK/name(width,coefficient);folder.mkdir(exist_ok=False);report=OUT/name(width,coefficient);report.mkdir(exist_ok=False)
        identity={'teacher_sha256':record['checkpoint_sha256'],'dataset_sha256':record['dataset_sha256'],'target_cache_sha256':record['target_cache_sha256'],'config_sha256':digest(RUN/'config.yaml'),'width':width,'lambda':coefficient}
        m=fit(width,coefficient,xs['train'],ys['train'],xs['validation'],ys['validation'],folder,identity)
        from .evaluate import evaluate
        evaluate(m,width,coefficient,report,d,{k:v.cpu().numpy() for k,v in ys.items()},xs,device)
        verify_frozen();print('CONFIG COMPLETE',name(width,coefficient),flush=True)
    write_json(RUN/f'worker_{width}_complete.json',{'width':width,'configurations':len(C['lambdas']),'GPU':torch.cuda.get_device_name(device)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','worker']);parser.add_argument('--width',type=int,choices=[512,1024]);args=parser.parse_args();device=select_device('cuda')
    if args.command=='prepare':prepare(device)
    else:
        if args.width is None:raise ValueError('Specify one approved worker width')
        worker(args.width,device)

if __name__=='__main__':main()

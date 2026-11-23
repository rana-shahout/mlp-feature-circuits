"""Autonomous V2 Stage 0/1 runner; no V2 transcoder implementation."""

import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

import numpy as np
import torch
import yaml

from src.models import OriginalMLP
from src.runtime import load_checkpoint,save_checkpoint,seed_everything,select_device,write_json
from .data import RULES,PAIRS,DNF_PAIRS,targets,keys,make_embeddings,generate,stratified_split,audit,clause_counts
from .analysis import metrics,predict,group_metrics,probe_bank,correlations,native_diagnostics

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def protected_v1():
    return {str(p.resolve()):digest(p) for p in sorted((ROOT/'results').rglob('*.pt'))}


def validate(c):
    expected={'semantic_input':20,'input':12,'hidden':64,'output':12,'semantic_output':9}
    if c['spec_version']!='0.3' or c['dimensions']!=expected or tuple(c['rules'])!=RULES:
        raise ValueError('V2 first-run architecture and all nine target rules are frozen by SPEC v0.3')
    d,t,e=c['data'],c['training'],c['evaluation']
    if d['active_counts']!=[3,4,5,6] or d['stratification']!='combined_labels_xor_corner_dnf_count_with_pooled_rare_strata':
        raise ValueError('Unknown V2 population protocol')
    for key in ('active_probabilities','mode_probabilities','split_fractions','dnf_positive_clause_probabilities'):
        v=np.asarray(d[key],dtype=float)
        if np.any(v<=0) or not np.isfinite(v).all() or not np.isclose(v.sum(),1):raise ValueError(f'Invalid probabilities: {key}')
    if len(d['active_probabilities'])!=4 or len(d['mode_probabilities'])!=3 or len(d['split_fractions'])!=3:raise ValueError('Invalid probability dimensions')
    if len(d['dnf_positive_clause_counts'])!=len(d['dnf_positive_clause_probabilities']) or any(k not in (1,2,3) for k in d['dnf_positive_clause_counts']):raise ValueError('Invalid DNF positive counts')
    if not 3<=d['min_samples']<=d['n_samples'] or d['max_attempts']<=0:raise ValueError('Invalid sample budget')
    if t['optimizer']!='adam' or t['schedule']!='cosine' or t['checkpoint_selection']!='validation_full_output_mse':raise ValueError('Unknown teacher optimization policy')
    if not 0<t['minimum_learning_rate']<=t['learning_rate'] or t['weight_decay']<0:raise ValueError('Invalid learning rate or regularization')
    if min(t['batch_size'],t['max_epochs'],t['early_stopping_patience'],e['batch_size'])<=0 or t['early_stopping_min_delta']<0:raise ValueError('Invalid training budget')
    if e['semantic_threshold']!=0.5:raise ValueError('First V2 teacher uses declared threshold 0.5; no silent threshold tuning')
    if any(not isinstance(seed,int) or not 0<=seed<2**32 for seed in c['seeds'].values()):raise ValueError('Invalid seed')


def locations(run_id):
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('Use a simple unique V2 run ID')
    return {name:BASE/name/run_id for name in ('runs','data','checkpoints','reports')}


def write_csv(path,rows,fields=None):
    import csv
    with open(path,'w',newline='') as f:
        writer=None
        for row in rows:
            if writer is None:writer=csv.DictWriter(f,fieldnames=fields or list(row));writer.writeheader()
            writer.writerow(row)
        if writer is None and fields:csv.DictWriter(f,fieldnames=fields).writeheader()


def prepare(c,paths,command):
    run=paths['runs']
    if run.exists():
        saved=yaml.safe_load((run/'config.yaml').read_text())
        if saved!=c:raise RuntimeError('Existing V2 config differs; no reuse/overwrite permitted')
        manifest=json.loads((run/'prepared.json').read_text())
        if manifest['dataset_sha256']!=digest(paths['data']/'dataset.npz'):raise RuntimeError('Frozen dataset changed')
        return
    for path in paths.values():path.mkdir(parents=True,exist_ok=False)
    (run/'config.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    shutil.copyfile(ROOT/'SPEC.md',run/'SPEC.snapshot.md');shutil.copyfile(ROOT/'CODEX_PROMPT.md',run/'CODEX_PROMPT.snapshot.md')
    write_json(run/'v1_protected_hashes_before.json',protected_v1())
    p,q=make_embeddings(c['seeds']['P'],c['seeds']['Q'])
    a,mode,requested,generation=generate(c['data'],c['seeds']['data'])
    split,manifest=stratified_split(a,c['data']['split_fractions'],c['seeds']['split'],c['data']['minimum_stratum_size'])
    np.savez_compressed(paths['data']/'dataset.npz',a=a,g=targets(a),P=p,Q=q,mode=mode,requested_rule=requested,**split)
    np.save(paths['data']/'P.npy',p);np.save(paths['data']/'Q.npy',q)
    # The manifest contains exact vector identities, not only row ranges.
    write_json(paths['data']/'split_manifest.json',{'strategy':manifest,'partitions':{name:{'indices':idx.tolist(),'vector_keys':keys(a[idx]).astype(int).tolist()} for name,idx in split.items()}})
    gram=p.T@p
    coverage={'generator':generation,'full':audit(a,mode),'splits':{name:audit(a[idx],mode[idx]) for name,idx in split.items()},
              'embedding':{'P_shape':list(p.shape),'P_rank':int(np.linalg.matrix_rank(p)), 'P_norms':np.linalg.norm(p,axis=0).tolist(),
                'P_singular_values':np.linalg.svd(p,compute_uv=False).tolist(),
                'P_mutual_coherence':float(np.abs(gram-np.diag(np.diag(gram))).max()),'P_gram':gram.tolist(),
                'Q_shape':list(q.shape),'Q_orthogonality_max_error':float(np.abs(q.T@q-np.eye(9)).max()),
                'method':'P independent Gaussian unit columns; Q reduced QR, positive diagonal R convention; stored float32'}}
    write_json(paths['data']/'audit.json',coverage)
    print('V2 coverage audit:',json.dumps(coverage),flush=True)
    for name,record in coverage['splits'].items():
        if min(record['positive_counts'])<10 or min(record['negative_counts'])<10:
            raise RuntimeError(f'Insufficient class support in {name}; review population before fitting teacher')
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy','PyYAML','matplotlib','pytest')}
    code=[*BASE.glob('*.py'),*BASE.glob('configs/*.yaml'),ROOT/'src/models.py',ROOT/'src/runtime.py',ROOT/'src/coordinates.py',ROOT/'src/interactions.py']
    write_json(run/'provenance.json',{'command':command,'git_commit':revision.stdout.strip() if revision.returncode==0 else None,
              'git_status':subprocess.run(['git','status','--short'],cwd=ROOT,text=True,capture_output=True).stdout,
              'python':sys.version,'packages':versions,'source_hashes':{str(path.relative_to(ROOT)):digest(path) for path in code}})
    write_json(run/'prepared.json',{'dataset_sha256':digest(paths['data']/'dataset.npz'),'P_sha256':digest(paths['data']/'P.npy'),'Q_sha256':digest(paths['data']/'Q.npy')})


def load_data(paths):
    with np.load(paths['data']/'dataset.npz',allow_pickle=False) as data:return {k:data[k] for k in data.files}


@torch.no_grad()
def loss_on(model,x,y,batch):
    total=0
    for start in range(0,len(x),batch):total+=(model(x[start:start+batch])-y[start:start+batch]).square().sum().item()
    result=total/y.numel()
    if not np.isfinite(result):raise RuntimeError('Non-finite validation MSE')
    return result


def train(c,paths,device):
    if (paths['runs']/'test_evaluated.json').exists():raise RuntimeError('Test already scored: no additional training of this run')
    seed_everything(c['seeds']['teacher'],c['training']['deterministic'])
    data=load_data(paths);t=c['training'];model=OriginalMLP(12,64,12).to(device)
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


def plots(paths,c,all_metrics,data,hdiag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=paths['reports'];history=json.loads((paths['runs']/'history.json').read_text())
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['epoch'] for r in history],[r['train_online_mse'] for r in history],label='train online');ax.plot([r['epoch'] for r in history],[r['validation_mse'] for r in history],label='validation');ax.set_yscale('log');ax.set_xlabel('epoch');ax.set_ylabel('12D MSE');ax.legend();fig.tight_layout();fig.savefig(report/'learning_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ('train','validation','test'):
        axes[0].plot(range(9),[r['balanced_accuracy'] for r in all_metrics[name]['per_rule']],'o-',label=name)
        axes[1].plot(range(9),[r['recall'] for r in all_metrics[name]['per_rule']],'o-',label=name)
    for ax,title in zip(axes,('Balanced accuracy','Recall')):
        ax.set_xticks(range(9),[f'g{i}' for i in range(9)]);ax.set_ylim(0,1.02);ax.set_title(title);ax.legend()
    axes[0].axhline(.99,color='black',ls='--',lw=1);fig.tight_layout();fig.savefig(report/'per_feature.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,array,title in [(axes[0],data['P'].T@data['P'],'P column cosine similarities'),(axes[1],hdiag['input_correlation'],'Hidden activation/input correlation'),(axes[2],hdiag['interaction_usage_overlap'],'Interaction-usage cosine overlap')]:
        ax.imshow(array,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1);ax.set_title(title)
    fig.tight_layout();fig.savefig(report/'geometry_selectivity.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));axes[0].bar(range(64),hdiag['activation_prevalence']);axes[0].set_title('Hidden ReLU prevalence, train');axes[0].set_xlabel('original neuron');axes[1].imshow(hdiag['output_correlation'],aspect='auto',cmap='coolwarm',vmin=-1,vmax=1);axes[1].set_title('Hidden activation / semantic target correlation');fig.tight_layout();fig.savefig(report/'hidden_diagnostics.png',dpi=160);plt.close(fig)


def evaluate(c,paths,device):
    run,report=paths['runs'],paths['reports']
    if (run/'complete.json').exists():print('Already evaluated:',report/'report.md',flush=True);return
    if (run/'test_evaluated.json').exists():raise RuntimeError('Test already evaluated; regenerate report from saved metrics without rescoring')
    if not (run/'training_complete.json').exists():raise RuntimeError('Training must finish before test evaluation')
    data=load_data(paths);cp=load_checkpoint(paths['checkpoints']/'best.pt')
    if cp['identity']!={'config_sha256':digest(run/'config.yaml'),'dataset_sha256':digest(paths['data']/'dataset.npz')}:raise RuntimeError('Checkpoint/config/data mismatch')
    model=OriginalMLP(12,64,12).to(device);model.load_state_dict(cp['model']);model.eval().requires_grad_(False)
    threshold=c['evaluation']['semantic_threshold'];all_metrics={};grouped={};cached={}
    for name in ('train','validation','test'):
        idx=data[name];a,g=data['a'][idx],data['g'][idx]
        pred=predict(model,a,data['P'],device,c['evaluation']['batch_size'],hidden=name=='train')
        all_metrics[name]=metrics(pred['output'],g,data['Q'],threshold)
        grouped[name]=group_metrics(a,pred['output'],g,data['Q'],threshold)
        np.savez_compressed(report/f'predictions_{name}.npz',**pred,semantic=pred['output'].astype(float)@data['Q'].astype(float),vector_key=keys(a))
        cached[name]=pred
        print('EVALUATE',name,json.dumps(all_metrics[name]),flush=True)
    # Single planned test use is recorded; subsequent training of this run is forbidden.
    write_json(run/'test_evaluated.json',{'checkpoint_sha256':digest(paths['checkpoints']/'best.pt'),'best_epoch':cp['epoch'],
               'time_utc':datetime.now(timezone.utc).isoformat(),'policy_frozen_before_test':True,'test_use_count':1})
    a,metadata,bases,names=probe_bank(c)
    pred=predict(model,a,data['P'],device,c['evaluation']['batch_size'])['output']
    g=targets(a);scores=pred.astype(float)@data['Q'].astype(float)
    known={int(key):name for name in ('train','validation','test') for key in keys(data['a'][data[name]])}
    for meta,key in zip(metadata,keys(a)):meta['population_overlap']=known.get(int(key),'unseen_by_saved_dataset')
    np.savez_compressed(report/'constructed_probes.npz',a=a,g=g,output=pred,semantic=scores,bases=bases)
    write_json(report/'probe_metadata.json',metadata);write_json(report/'context_names.json',names)
    ordinary=np.asarray([not m['outside_training_density'] for m in metadata]);unseen=np.asarray([m['population_overlap']=='unseen_by_saved_dataset' for m in metadata])
    probes={label:metrics(pred[mask],g[mask],data['Q'],threshold) for label,mask in {
        'all_constructed':np.ones(len(a),dtype=bool),'density_3_to_6':ordinary,'outside_density_3_to_6':~ordinary,'unseen_rows':unseen,'unseen_density_3_to_6':unseen&ordinary}.items()}
    probe_rows=[]
    for i,meta in enumerate(metadata):
        for j,rule in enumerate(RULES):probe_rows.append({**meta,'row':i,'output_rule':rule,'target':int(g[i,j]),'score':float(scores[i,j]),'predicted':int(scores[i,j]>=threshold),'error':bool((scores[i,j]>=threshold)!=g[i,j])})
    write_csv(report/'controlled_truth_tables.csv',probe_rows)
    for kind in sorted(set(m['kind'] for m in metadata)):
        mask=np.asarray([m['kind']==kind for m in metadata]);probes[kind]=metrics(pred[mask],g[mask],data['Q'],threshold)
    # Every nuisance flip is paired within the identical semantic/background context.
    nuisance=[]
    for context in names:
        for bit in range(15,20):
            indices=[i for i,m in enumerate(metadata) if m['kind']=='distractor_flip' and m['context']==context and m['designed_rule']==str(bit)]
            off,on=indices
            np.testing.assert_array_equal(g[off],g[on])
            for j,rule in enumerate(RULES):nuisance.append({'context':context,'distractor':bit,'output_rule':rule,'score_change':float(scores[on,j]-scores[off,j]),'ground_truth_change':0,'class_flip':bool((scores[on,j]>=threshold)!=(scores[off,j]>=threshold))})
    write_csv(report/'controlled_distractor_flips.csv',nuisance)
    write_json(report/'metrics.json',all_metrics);write_json(report/'grouped_metrics.json',grouped);write_json(report/'probe_metrics.json',probes)
    write_csv(report/'per_feature_metrics.csv',({'population':name,**row} for name,m in all_metrics.items() for row in m['per_rule']))
    # Preserve all ordinary-example errors, with distractor identities and clause counts.
    def errors():
        for name in ('train','validation','test'):
            idx=data[name];inputs=data['a'][idx];truth=data['g'][idx];s=cached[name]['output'].astype(float)@data['Q'].astype(float)
            for i,j in np.argwhere((s>=threshold)!=truth.astype(bool)):
                yield {'population':name,'row':int(i),'input_key':int(keys(inputs[i])),'rule':RULES[j],'target':int(truth[i,j]),'score':float(s[i,j]),
                       'active_inputs':' '.join(map(str,np.flatnonzero(inputs[i]))),'active_distractors':' '.join(map(str,np.flatnonzero(inputs[i,15:])+15)),
                       'density':int(inputs[i].sum()),'dnf_satisfied_clauses':int(clause_counts(inputs[i]))}
    write_csv(report/'errors.csv',errors(),['population','row','input_key','rule','target','score','active_inputs','active_distractors','density','dnf_satisfied_clauses'])
    h=cached['train']['hidden'];train_a=data['a'][data['train']];train_g=data['g'][data['train']]
    hdiag={'activation_prevalence':(h>0).mean(0),'mean':h.mean(0),'std':h.std(0),'quantiles':np.quantile(h,[0,.5,.9,.99,1],axis=0),
           'input_correlation':correlations(h,train_a),'output_correlation':correlations(h,train_g)}
    native,interaction_rows,topk=native_diagnostics(model,data['P'],data['Q'],bases,c,device)
    np.savez_compressed(report/'native_coordinates_interactions.npz',**native)
    write_json(report/'dominant_neuron_approximation.json',topk);write_csv(report/'native_pair_interactions.csv',interaction_rows)
    usage=np.stack([np.abs(native['contributions'][j,:,j,:]).mean(0) for j in range(8)]+[np.abs(native['contributions'][:,:,8,:]).mean((0,1))])
    norms=np.linalg.norm(usage,axis=1);den=norms[:,None]*norms[None,:]
    hdiag['interaction_usage']=usage;hdiag['interaction_usage_overlap']=np.divide(usage@usage.T,den,out=np.zeros((9,9)),where=den>0)
    np.savez_compressed(report/'hidden_diagnostics.npz',**hdiag)
    plots(paths,c,all_metrics,data,hdiag)
    make_report(c,paths,all_metrics,probes,grouped,interaction_rows,nuisance)
    before=json.loads((run/'v1_protected_hashes_before.json').read_text());after=protected_v1()
    write_json(run/'v1_protected_hashes_after.json',after)
    if before!=after:raise RuntimeError('A protected V1 checkpoint changed or a V1 checkpoint was added during V2')
    write_json(run/'complete.json',{'stage':1,'v1_checkpoints_unchanged':True,'native_forward_and_signed_sum_checks_passed':True,'report':str(report/'report.md'),'test_use_count':1})


def make_report(c,paths,m,probes,grouped,interactions,nuisance):
    audit_data=json.loads((paths['data']/'audit.json').read_text());training=json.loads((paths['runs']/'training_complete.json').read_text())
    fail=[r['rule'] for r in m['test']['per_rule'][:8] if r['balanced_accuracy'] is None or r['balanced_accuracy']<.99]
    lines=['# V2 pairwise-superposition original teacher: Stage 0–1','',
        '**Teacher only. No V2 transcoder, architecture sweep, or original-neuron intervention was run.**','',
        '## Settings and scientific boundaries','',
        'SPEC v0.3; 20 Boolean inputs; P is 12×20 with independent Gaussian unit columns; Q is 12×9 with QR-orthonormal columns; teacher is exactly 12→64→12 ReLU. Targets: six ANDs (g0=a0a1, g1=a0a2, g2=a3a4, g3=a5a6, g4=a7a8, g5=a9a10), g6=a11 XOR a12, g7=a13 AND NOT a14, and g8=g0 OR g2 OR g3 OR g4 OR g5. g1 is intentionally excluded from g8; a15..a19 are distractors.',
        f"Seeds: {c['seeds']}. MSE against Qg, Adam, lr {c['training']['learning_rate']}→{c['training']['minimum_learning_rate']} cosine, batch {c['training']['batch_size']}, max {c['training']['max_epochs']} epochs, patience {c['training']['early_stopping_patience']}, min-delta {c['training']['early_stopping_min_delta']}. Threshold fixed at 0.5 before training.",
        'Good task accuracy and exact coordinate identities are not evidence of a unique internal circuit or superposed feature-channel code. All negative results are preserved.', '',
        '## Data audit','',
        f"Accepted {audit_data['generator']['accepted']} unique vectors in {audit_data['generator']['attempts']} proposals. Actual mean active bits: {audit_data['full']['mean_active']:.3f}. Mode counts background/positive/hard-negative: {audit_data['full']['mode_counts']}. These differ from proposal probabilities after uniqueness rejection.",
        f"P rank {audit_data['embedding']['P_rank']}, mutual coherence {audit_data['embedding']['P_mutual_coherence']:.4f}; Q maximum orthogonality error {audit_data['embedding']['Q_orthogonality_max_error']:.3g}.",
        'Combined label patterns, XOR corner and DNF count category drive feasible stratification, with rare strata pooled. Exact vector-key split manifests are saved. Supports and nuisance correlations are in data/audit.json. Sampling mode/labels never enter the model input.', '',
        '| Population | N | Exact-vector accuracy | Output MSE | Semantic MSE |','|---|---:|---:|---:|---:|']
    for name,row in m.items():lines.append(f"| {name} | {row['n']} | {row['exact_vector_accuracy']:.4f} | {row['output_mse']:.6g} | {row['semantic_mse']:.6g} |")
    lines += ['',f"Training stopped at epoch {training['epochs']} ({training['stop_reason']}); validation selected epoch {training['best_epoch']} with MSE {training['best_validation_mse']:.6g}. Test was scored once after this policy/checkpoint was frozen; it never selected epochs or hyperparameters.",'',
        '## Per-feature held-out results','',
        '| Output | Positive / negative support | Precision | Recall | Balanced accuracy | FPR | FNR | Semantic MSE |','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in m['test']['per_rule']:
        def fmt(value):return f'{value:.4f}' if value is not None else 'undefined'
        lines.append(f"| {row['rule']} | {row['n_positive']} / {row['n_negative']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['balanced_accuracy'])} | {fmt(row['false_positive_rate'])} | {fmt(row['false_negative_rate'])} | {row['mse']:.6g} |")
    lines += ['', 'Classification uses real semantic regression scores, not calibrated probabilities. Exact-vector accuracy means all nine outputs must match; mean-feature accuracy is reported separately. Positive frequencies, bias, MAE and continuous signed margins are retained for every output/partition.', '',
        '## Controlled contrasts and distractors','',
        '| Constructed population | N (memberships) | Exact-vector accuracy | Semantic MSE |','|---|---:|---:|---:|']
    for name,row in probes.items():
        if row['n']:lines.append(f"| {name} | {row['n']} | {row['exact_vector_accuracy']:.4f} | {row['semantic_mse']:.6g} |")
    lines += ['', 'Constructed probes repeat vectors across rule/context memberships. They are not an IID test; each row carries train/validation/test overlap and a flag for density outside 3–6. Full four-corner pair tables, all eight overlapping a0/a1/a2 cases, zero/one/multiple DNF clauses, and matched distractor flips are saved. DNF is secondary; stress corners can exceed training density.', '',
        '### DNF held-out zero/one/multiple clause cases','', '| True clauses | N | g8 recall | g8 FPR | g8 MSE |','|---|---:|---:|---:|---:|']
    for count,record in grouped['test']['dnf_satisfied_clauses'].items():
        row=record['per_rule'][8];lines.append(f"| {count} | {record['n']} | {row['recall']} | {row['false_positive_rate']} | {row['mse']:.6g} |")
    strongest=sorted(nuisance,key=lambda row:abs(row['score_change']),reverse=True)[:5]
    lines += ['', 'Largest controlled distractor score changes (ground truth unchanged; descriptive, not neuron-causal evidence):']
    for row in strongest:lines.append(f"- Distractor {row['distractor']} → {row['output_rule']}, context {row['context']}: {row['score_change']:+.5f}; classification flip={row['class_flip']}.")
    lines += ['', '## Original-neuron diagnostics','',
        'C1=W1P is (64,20), D2_eff=Q.T W2 is (9,64). Original bias and ReLU gates are retained; D2_eff C1 is not treated as the nonlinear circuit. Direct input-space forward predictions and all signed original-neuron finite differences passed numerical checks.',
        'native_coordinates_interactions.npz contains all signed contributions for the eight primitive pairs and (a1,a2). native_pair_interactions.csv reports signed positive/negative sums, actual/target effects, effective participation, and identity residuals. DNF output contributions are retained alongside every primitive.',
        'dominant_neuron_approximation.json chooses a fixed top-1/3/5/10 neuron set by mean absolute contribution on reference contexts and reports its residual on distinct context-bank entries. Context-bank entries may overlap training/test and are flagged in probe metadata; this is a descriptive truncation test, not a blinded mechanism validation or intervention.',
        'Activation prevalence/magnitude, correlations with all semantic inputs/outputs, and overlapping neuron usage are saved in hidden_diagnostics.npz. Correlation and shared usage do not establish monosemanticity, feature-channel coding, or causal necessity.', '',
        '## Review decision','',
        f"Primitive rules below the 99% balanced-accuracy review target on held-out test: {', '.join(fail) if fail else 'none'}. DNF is evaluated separately.",
        ('**Not ready for automatic progression to a V2 transcoder.** The primitive behavioral target is not met. Preserve this run and diagnose sampling, superposition and optimization before proposing a research-approved revision; do not silently increase width or remove rules.' if fail else '**Primitive IID target met**, but review conditional contrasts, distractor dependence and DNF stress behavior before explicitly authorizing a transcoder.'),
        'This experiment alone cannot show whether extra optimization, a data revision or a capacity change would solve failures. No such change was attempted after reading test results. No V2 transcoder was trained.', '',
        '## Artifacts and reproducibility','',f"Run metadata/logs: `{paths['runs']}`. Checkpoints: `{paths['checkpoints']}`. Frozen data/embeddings/splits: `{paths['data']}`.",
        'provenance.json records command, source hashes, installed package versions, Git commit/status; environment.json records actual CUDA GPU/device. All protected V1 checkpoint hashes are checked before/after. All results derive from saved data and the validation-selected checkpoint.', '',
        '![Learning curves](learning_curves.png)','![Per-feature performance](per_feature.png)','![Embedding and neuron selectivity](geometry_selectivity.png)','![Hidden neuron diagnostics](hidden_diagnostics.png)']
    (paths['reports']/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['all','prepare','train','evaluate'])
    p.add_argument('--config',default=str(BASE/'configs/teacher_v2.yaml'));p.add_argument('--run-id')
    args=p.parse_args();run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid4().hex[:8]
    paths=locations(run_id);c=yaml.safe_load(Path(args.config).read_text());validate(c)
    print('V2 paths:',{name:str(path) for name,path in paths.items()},flush=True)
    try:
        prepare(c,paths,' '.join(sys.argv))
        if args.command=='prepare':return
        if (paths['runs']/'complete.json').exists():print('Completed report:',paths['reports']/'report.md',flush=True);return
        seed_everything(c['seeds']['teacher'],c['training']['deterministic']);device=select_device(c['device'])
        from src.train_mlp import environment_record
        write_json(paths['runs']/'environment.json',environment_record(device))
        if args.command in ('all','train'):train(c,paths,device)
        if args.command in ('all','evaluate'):evaluate(c,paths,device)
        print('V2 command completed successfully:',args.command,flush=True)
    except BaseException:
        if paths['runs'].exists():(paths['runs']/('failure-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'.log')).write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()

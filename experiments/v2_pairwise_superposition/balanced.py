"""Single objective-only controlled trial; historical test is never scored."""
import json, math, time, shutil, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
from src.models import OriginalMLP
from src.runtime import seed_everything, select_device, load_checkpoint, save_checkpoint, write_json
from src.coordinates import original_coordinates
from src.interactions import finite_interaction, corners_and_signs
from .run import BASE, ROOT, digest, loss_on, validate, write_csv
from .analysis import metrics, predict, group_metrics
from .data import RULES, PAIRS, keys, targets, generate

OLD='first-teacher-20261027'
NEW='balanced-semantic-20261027'
OUT=BASE/'reports'/NEW
RUN=BASE/'runs'/NEW
CK=BASE/'checkpoints'/NEW
SOURCE=BASE/'reports'/OLD


def objective(y,g,q,prevalence):
    if y.ndim!=2 or y.shape[1]!=12 or g.shape!=(len(y),9) or q.shape!=(12,9):raise ValueError('Objective tensor shape mismatch')
    if prevalence.shape!=(9,) or not torch.all((prevalence>0)&(prevalence<1)):raise ValueError('Both training classes required')
    s=y@q
    weights=g/(2*prevalence)+(1-g)/(2*(1-prevalence))
    semantic=(weights*(s-g).square()).mean()
    residual=y-s@q.T
    orthogonal=residual.square().sum(1).mean()/12
    return semantic+orthogonal


def rankings(s,t):
    # Group tied scores before integration: ROC trapezoids and non-interpolated AP.
    order=np.argsort(-s,kind='stable');s=s[order];t=t[order].astype(int)
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.r_[0,np.cumsum(t)[ends]];fp=np.r_[0,np.cumsum(1-t)[ends]]
    recall=tp/t.sum();fpr=fp/(len(t)-t.sum());precision=np.divide(tp,tp+fp,out=np.ones_like(tp,dtype=float),where=tp+fp>0)
    return {'roc_auc':float(np.trapezoid(recall,fpr)), 'average_precision':float((np.diff(recall)*precision[1:]).sum()),
            'threshold':np.r_[s[ends][0]+1,s[ends]].tolist(),'recall':recall.tolist(),'precision':precision.tolist(),'fpr':fpr.tolist()}


def distributions(s,g):
    result=[]
    for j,rule in enumerate(RULES):
        t=g[:,j].astype(bool);v=s[:,j];fn=v[t&(v<.5)]
        row={'rule':rule,'quantile_levels':[0,.01,.1,.25,.5,.75,.9,.99,1]}
        for label,values in [('positive',v[t]),('negative',v[~t]),('false_negative',fn)]:
            row[label]={'n':len(values),'quantiles':np.quantile(values,row['quantile_levels']).tolist() if len(values) else None}
        row['fn_fraction_within_0.05_below_threshold']=float((fn>=.45).mean()) if len(fn) else None
        row['fn_fraction_within_0.10_below_threshold']=float((fn>=.4).mean()) if len(fn) else None
        row['ranking']=rankings(v,t);result.append(row)
    return result


def phase_a(d):
    result={}
    for split in ('train','validation'):
        with np.load(SOURCE/f'predictions_{split}.npz') as p:
            np.testing.assert_array_equal(p['vector_key'],keys(d['a'][d[split]]))
            result[split]=distributions(p['semantic'],d['g'][d[split]])
    history=json.loads((BASE/'runs'/OLD/'history.json').read_text())
    slopes={}
    for n in (100,250,500):
        h=history[-n:];x=np.array([v['epoch'] for v in h]);slopes[str(n)]={k:float(np.polyfit(x,[v[k] for v in h],1)[0]) for k in ('train_online_mse','validation_mse')}
    gram=d['P'].astype(float).T@d['P'].astype(float);geometry=[]
    for j,(i,k) in enumerate(PAIRS):
        other=[v for v in range(20) if v not in (i,k)]
        geometry.append({'rule':RULES[j],'pair_cosine':float(gram[i,k]),'pair_singular_values':np.linalg.svd(d['P'][:,[i,k]],compute_uv=False).tolist(),
             'antecedent_max_other_abs_cosines':[float(np.abs(gram[v,other]).max()) for v in (i,k)],
             'antecedent_distractor_cosines':gram[[i,k],15:].tolist()})
    write_json(OUT/'phase_a.json',{'scores':result,'tail_slopes_per_epoch':slopes,'geometry':geometry,'P_gram':gram.tolist()})
    return result,slopes,geometry


def train(c,d,device):
    seed_everything(c['seeds']['teacher'],True)
    m=OriginalMLP(12,64,12).to(device)
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    seed_everything(c['seeds']['teacher'],True)
    reference=OriginalMLP(12,64,12)
    assert all(torch.equal(initial[k],v) for k,v in reference.state_dict().items())
    save_checkpoint(CK/'initial.pt',{'model':initial,'seed':3105})
    t=c['training'];opt=torch.optim.Adam(m.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    generator=torch.Generator().manual_seed(c['seeds']['teacher'])
    p,q=[torch.as_tensor(d[k],device=device) for k in ('P','Q')]
    cache={s:(torch.as_tensor(d['a'][d[s]],device=device,dtype=torch.float32)@p.T,torch.as_tensor(d['g'][d[s]],device=device)) for s in ('train','validation')}
    prevalence=cache['train'][1].mean(0)
    write_json(RUN/'objective.json',{'name':'class_balanced_semantic_mse','gamma':1/12,'training_prevalence':prevalence.cpu().tolist(),'minibatch_estimator':'fixed training prevalence inverse class weighting; unbiased for empirical conditional loss','residual_reduction':'mean examples of squared L2 norm; multiplied by 1/12','selection':'original validation full-output unweighted MSE'})
    history=[];best=float('inf');ref=float('inf');stale=0;starttime=time.monotonic()
    for epoch in range(1,t['max_epochs']+1):
        if stale>=t['early_stopping_patience']:break
        m.train();lr=t['minimum_learning_rate']+(t['learning_rate']-t['minimum_learning_rate'])*(1+math.cos(math.pi*(epoch-1)/(t['max_epochs']-1)))/2
        for group in opt.param_groups:group['lr']=lr
        x,g=cache['train'];order=torch.randperm(len(x),generator=generator).to(device);total=0
        for start in range(0,len(order),t['batch_size']):
            idx=order[start:start+t['batch_size']];opt.zero_grad(set_to_none=True);loss=objective(m(x[idx]),g[idx],q,prevalence)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            if any(not torch.isfinite(v.grad).all() for v in m.parameters()):raise RuntimeError('Nonfinite gradient')
            opt.step();total+=loss.item()*len(idx)
        m.eval();vx,vg=cache['validation'];val=loss_on(m,vx,vg@q.T,c['evaluation']['batch_size'])
        improved=val<best
        if improved:best=val;best_epoch=epoch
        if val<ref-t['early_stopping_min_delta']:ref=val;stale=0
        else:stale+=1
        row={'epoch':epoch,'train_online_balanced_loss':total/len(x),'validation_mse':val,'learning_rate':lr,'stale':stale}
        history.append(row)
        cp={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best_epoch':best_epoch,'best_mse':best,'config':c,'history':history,'shuffle_rng':generator.get_state(),'dataset_sha256':digest(BASE/'data'/OLD/'dataset.npz')}
        if improved:save_checkpoint(CK/'best.pt',cp)
        save_checkpoint(CK/'last.pt',cp);write_json(RUN/'history.json',history)
        if epoch==1 or epoch%100==0:print('TRAIN',row,flush=True)
    write_json(RUN/'training_complete.json',{'epochs':epoch if stale<t['early_stopping_patience'] else history[-1]['epoch'],'best_epoch':best_epoch,'best_validation_mse':best,'seconds':time.monotonic()-starttime,'stop_reason':'patience' if stale>=t['early_stopping_patience'] else 'budget'})
    m.load_state_dict(load_checkpoint(CK/'best.pt')['model']);return m


def reserve(d,c):
    # Same versioned generator, independent seed; reject every previously seen vector.
    cfg=dict(c['data']);cfg['n_samples']=12000;cfg['max_attempts']=500000
    a,_,_,generation=generate(cfg,87103103)
    with np.load(SOURCE/'constructed_probes.npz') as old:excluded=set(map(int,keys(old['a'])))
    excluded.update(map(int,keys(d['a'])))
    a=np.array([v for v,k in zip(a,keys(a)) if int(k) not in excluded],dtype=np.uint8)[:4000]
    if len(a)!=4000:raise RuntimeError('Insufficient unseen final population')
    final=BASE/'data'/NEW;final.mkdir(exist_ok=False)
    np.savez_compressed(final/'reserved_final_UNSCORED.npz',a=a,g=targets(a))
    write_json(final/'manifest.json',{'status':'UNSCORED: explicit finalization decision required','seed':87103103,'n':len(a),'sha256':digest(final/'reserved_final_UNSCORED.npz'),'excluded':'all original train/validation/test and all previously constructed probes','sampling':'original generator, rejection conditional on disjointness; not identical unconditional IID population','generation':generation})


def compare(m,d,device,phase):
    allmetrics={};scores={};dist={}
    for split in ('train','validation'):
        g=d['g'][d[split]];a=d['a'][d[split]]
        with np.load(SOURCE/f'predictions_{split}.npz') as old:yp=old['output'].copy()
        yn=predict(m,a,d['P'],device,2048)['output']
        np.savez_compressed(OUT/f'predictions_{split}.npz',output=yn,semantic=yn.astype(float)@d['Q'].astype(float),vector_key=keys(a))
        for label,y in [('original',yp),('balanced',yn)]:
            allmetrics[f'{label}_{split}']=metrics(y,g,d['Q'],.5);scores[label,split]=y.astype(float)@d['Q'].astype(float)
            dist[f'{label}_{split}']=distributions(scores[label,split],g)
        write_json(OUT/f'balanced_groups_{split}.json',group_metrics(a,yn,g,d['Q'],.5))
    write_json(OUT/'metrics.json',allmetrics);write_json(OUT/'score_distributions.json',dist)
    write_csv(OUT/'per_rule.csv',({'model_population':k,**v} for k,mets in allmetrics.items() for v in mets['per_rule']))
    # Historical constructed probes, excluding ALL rows that coincide with old test.
    oldtest=set(map(int,keys(d['a'][d['test']])))
    with np.load(SOURCE/'constructed_probes.npz') as p:bank={k:p[k] for k in p.files}
    keep=np.array([int(k) not in oldtest for k in keys(bank['a'])]);a=bank['a'][keep];g=bank['g'][keep]
    yn=predict(m,a,d['P'],device,2048)['output'];yp=bank['output'][keep];density=a.sum(1)
    probes={}
    for label,y in [('original',yp),('balanced',yn)]:
        for regime,mask in [('density_3_to_6',(density>=3)&(density<=6)),('OOD_density',(density<3)|(density>6))]:probes[label+'_'+regime]=metrics(y[mask],g[mask],d['Q'],.5)
    write_json(OUT/'controlled_metrics.json',{'policy':'historical diagnostic probes, old-test-overlap rows excluded; never used for selection','excluded_memberships':int((~keep).sum()),'metrics':probes})
    np.savez_compressed(OUT/'controlled_predictions.npz',a=a,g=g,original=yp,balanced=yn)
    # Complete pair/triple contexts only when every corner avoids historical test.
    teacher=OriginalMLP(12,64,12).to(device).double();teacher.load_state_dict(load_checkpoint(BASE/'checkpoints'/OLD/'best.pt')['model'])
    balanced=OriginalMLP(12,64,12).to(device).double();balanced.load_state_dict(m.state_dict())
    models={'original':teacher,'balanced':balanced};rows=[];arrays={};maxres=0
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
    report(allmetrics,phase,dist,probes,rows)


def plots(d,scores,mets,dist):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(13,10))
    for j,ax in enumerate(axes.flat):
        g=d['g'][d['validation'],j].astype(bool)
        for label,color in [('original','C0'),('balanced','C1')]:
            for positive,style in [(True,'-'),(False,'--')]:
                v=np.sort(scores[label,'validation'][g if positive else ~g,j]);ax.plot(v,np.arange(1,len(v)+1)/len(v),color=color,ls=style,label=label+(' positive' if positive else ' negative'))
        ax.axvline(.5,color='black',lw=.6);ax.set_title(RULES[j]);ax.set_ylabel('empirical CDF')
    axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'score_cdfs.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label in ('original','balanced'):
        v=mets[label+'_validation']['per_rule'];axes[0].plot([r['recall'] for r in v],'o-',label=label);axes[1].plot([r['false_positive_rate'] for r in v],'o-',label=label)
    for ax,title in zip(axes,['Validation recall','Validation FPR']):ax.set_title(title);ax.set_xticks(range(9),[f'g{j}' for j in range(9)]);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'recall_fpr.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for label,rundir in [('original',BASE/'runs'/OLD),('balanced',RUN)]:
        h=json.loads((rundir/'history.json').read_text());axes[0].plot([r['epoch'] for r in h],[r['validation_mse'] for r in h],label=label)
        for j in range(8):
            rr=dist[label+'_validation'][j]['ranking'];axes[1].plot(rr['recall'],rr['precision'],alpha=.55,label=f'{label} g{j}')
    axes[0].set_title('Validation full-output MSE (selection)');axes[0].set_yscale('log');axes[0].legend();axes[1].set_title('Primitive precision–recall diagnostics');axes[1].set_xlabel('Recall');axes[1].set_ylabel('Precision');fig.tight_layout();fig.savefig(OUT/'learning_and_pr.png',dpi=160);plt.close(fig)


def report(mets,phase,dist,probes,interactions):
    a,slopes,geometry=phase
    lines=['# Objective-only V2 teacher comparison','','Only the training objective changed. Both models use the identical frozen 20,000-vector dataset, split indices, P/Q, 12→64→12 ReLU architecture, initialization seed 3105, minibatch ordering, Adam, cosine learning-rate schedule, 2,000-epoch cap, patience 200, and fixed 0.5 threshold. Best checkpoint still minimizes unweighted validation full-output MSE. No historical test predictions were loaded or scored; old-test-overlap constructed rows/corners were excluded. All prior artifacts are preserved.','','Balanced loss uses fixed training class prevalences: mean of inverse-frequency weighted semantic squared errors plus (1/12) times mean squared orthogonal-residual L2 norm. Missing-class minibatches remain well-defined. This changes class importance and relative semantic/residual scaling, exactly as requested.','','## Train and validation results','','| Model / population | Exact vector | Semantic MSE | Full-output MSE |','|---|---:|---:|---:|']
    for label,m in mets.items():lines.append(f"| {label} | {m['exact_vector_accuracy']:.4f} | {m['semantic_mse']:.6f} | {m['output_mse']:.6f} |")
    lines+=['','## Validation per-rule results','','| Rule | Original recall | Balanced recall | Original FPR | Balanced FPR | Original precision | Balanced precision | Original BA | Balanced BA |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for j,rule in enumerate(RULES):
        old=mets['original_validation']['per_rule'][j];new=mets['balanced_validation']['per_rule'][j]
        vals=[old['recall'],new['recall'],old['false_positive_rate'],new['false_positive_rate'],old['precision'],new['precision'],old['balanced_accuracy'],new['balanced_accuracy']]
        lines.append('| '+rule+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','## Before retraining: score and optimization diagnosis','','FN proximity bands were fixed at [0.45,0.5) and [0.4,0.5), not optimized. ROC-AUC and average precision are threshold-free diagnostics; thresholds were never changed. Average precision is a tied-score grouped, non-interpolated PR integral.','','| Rule | Validation ROC-AUC | AP | FN within .05 | FN within .10 | FN median | Pair cosine |','|---|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    for j in range(8):
        r=a['validation'][j];q=r['false_negative']['quantiles'];lines.append('| '+RULES[j]+' | '+' | '.join([fmt(r['ranking']['roc_auc']),fmt(r['ranking']['average_precision']),fmt(r['fn_fraction_within_0.05_below_threshold']),fmt(r['fn_fraction_within_0.10_below_threshold']),fmt(q[4] if q else None),fmt(geometry[j]['pair_cosine'])])+' |')
    lines+=['',f'Tail slopes per epoch (unweighted online training MSE and validation MSE): `{json.dumps(slopes)}`. Online training averages and validation snapshots are different estimators. A decreasing tail at very low cosine LR indicates incomplete convergence, not proof that additional epochs resolve the task.','','High ranking performance with subthreshold positives supports an objective/calibration component; FNs far below 0.5 cannot be repaired by a tiny threshold adjustment. Class balancing tests this without changing the threshold. Input geometry is unchanged, so any improvement cannot be attributed to a better embedding. Pair cosines alone cannot establish a causal interference mechanism. Full geometry, score quantiles, and ROC/PR arrays are in phase_a.json and score_distributions.json.','','## Controlled comparisons and nonlinear interactions','','These are historical constructed diagnostics, not an unbiased held-out test. Density 3–6 is within training density support, not necessarily the training distribution. Other densities are OOD stress tests. Probe memberships may repeat vectors; counts are membership counts. No probe scores select checkpoints.','','| Model / density | N | Exact vector | Full-output MSE |','|---|---:|---:|---:|']
    for k,m in probes.items():lines.append(f"| {k} | {m['n']} | {m['exact_vector_accuracy']:.4f} | {m['output_mse']:.6f} |")
    lines+=['','| Primitive interaction | Original target MAE | Balanced target MAE |','|---|---:|---:|']
    for j in range(8):
        v=[r for r in interactions if r['indices']==str(PAIRS[j]) and r['rule']==RULES[j]]
        lines.append(f"| {RULES[j]} | {v[0]['mean_absolute_target_error']:.5f} | {v[1]['mean_absolute_target_error']:.5f} |")
    lines+=['','Full signed contributions from all 64 original neurons, positive/negative sums, pair and shared-input three-way differences, ground-truth differences, and direct-forward consistency checks are saved. Contexts are identical between models; complete contexts touching historical test were excluded. Similar interaction values do not establish identical mechanisms.','','## Scientific assessment','','ASSESSMENT_PENDING','','## Artifacts and execution','','- [Score distributions](score_cdfs.png), [recall/FPR](recall_fpr.png), [learning curves and PR](learning_and_pr.png).','- metrics.json and per_rule.csv: full train/validation precision, recall, FPR, balanced accuracy and regression errors.','- balanced_groups_train.json / balanced_groups_validation.json: density, nuisance identities, antecedent corners, DNF clauses.','- signed_interactions.npz / interactions.csv / interaction_checks.json: complete signed nonlinear diagnostics.','- New independently seeded final population: ../../data/'+NEW+'/reserved_final_UNSCORED.npz. All original dataset and old probes excluded; no model was evaluated on it. This is a rejection-conditioned population, not unconditional IID. Do not inspect scores before explicit finalization.','- Training/checkpoints/provenance and tests: ../../runs/'+NEW+'/ and ../../checkpoints/'+NEW+'/.','- No transcoder or additional training variant was launched.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    device=select_device('cuda');c=yaml.safe_load((BASE/'runs'/OLD/'config.yaml').read_text());validate(c)
    for p in (OUT,RUN,CK):p.mkdir(parents=True,exist_ok=False)
    oldfiles=list((BASE/'checkpoints'/OLD).glob('*.pt'))+list((ROOT/'results').rglob('*.pt'))
    hashes={str(p):digest(p) for p in oldfiles};write_json(RUN/'protected_before.json',hashes)
    shutil.copyfile(BASE/'runs'/OLD/'config.yaml',RUN/'config.yaml')
    shutil.copyfile(__file__,RUN/'balanced.source.py')
    for name in ('SPEC.md','CODEX_PROMPT.md'):shutil.copyfile(ROOT/name,RUN/(name+'.snapshot'))
    with np.load(BASE/'data'/OLD/'dataset.npz') as file:d={k:file[k] for k in file.files}
    write_json(RUN/'provenance.json',{'command':sys.argv,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(device),'dataset_sha256':digest(BASE/'data'/OLD/'dataset.npz'),'source_sha256':digest(__file__),'objective_only_override':{'name':'balanced_semantic_mse','gamma':1/12},'P_shape':list(d['P'].shape),'Q_shape':list(d['Q'].shape),'checkpoint_dimensions':{k:list(v.shape) for k,v in load_checkpoint(BASE/'checkpoints'/OLD/'best.pt')['model'].items()}})
    phase=phase_a(d);reserve(d,c);m=train(c,d,device);compare(m,d,device,phase)
    assert hashes=={str(p):digest(p) for p in oldfiles},'Protected checkpoints changed'
    write_json(RUN/'complete.json',{'protected_checkpoints_unchanged':True,'historical_test_scored':False,'reserved_final_scored':False,'status':'complete'})
    print('COMPLETE',OUT/'report.md',flush=True)

if __name__=='__main__':main()

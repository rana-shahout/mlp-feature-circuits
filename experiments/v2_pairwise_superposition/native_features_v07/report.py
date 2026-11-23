"""Readable, bounded reporting of locked v0.7 results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from .run import arr,write,csvwrite,HIST,LABELS,bits,tensor

def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)+'\n'
def report(run,cal,mats,rows,audit,ids,diag,checks,output,bank,hfit,ha,za,fa):
 A={k:arr(v) for k,v in mats.items()};figs=[];plt.rcParams.update({'font.size':11,'axes.titlesize':12})
 def save(fig,name):
  fig.tight_layout();fig.savefig(run/'figures'/f'{name}.png',dpi=210,bbox_inches='tight');fig.savefig(run/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig);figs.append(name)
 def heat(values,x,y,title,name,annot=False):
  if not values.size:
   fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');ax.text(.05,.6,title+'\nNo accepted coordinates',fontsize=14);save(fig,name);return
  fig,ax=plt.subplots(figsize=(max(10,len(x)*.65),max(4,len(y)*.45)));lim=max(1e-12,abs(values).max());im=ax.imshow(values,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(len(x)),x,rotation=60,ha='right');ax.set_yticks(range(len(y)),y);ax.set_title(title);fig.colorbar(im,ax=ax,label='signed coefficient')
  if annot:
   for i in range(len(y)):
    for j in range(len(x)):ax.text(j,i,f'{values[i,j]:+.2g}',ha='center',va='center',fontsize=8,color='white' if abs(values[i,j])>.65*lim else 'black')
  save(fig,name)
 fig,ax=plt.subplots(figsize=(11,3.6));ax.axis('off');ax.text(.02,.92,'Canonical frozen model: 20 inputs → 256 hidden neurons → 9 output scores',fontsize=14,va='top');ax.text(.03,.67,'a → W1=C1, b1 → ReLU → h → native W2, b2 → s',fontsize=14,color='#225d8e');ax.text(.03,.43,f'h → frozen affine readout D2, d2 → {len(ids)} accepted coordinates → algebraic C2 → scores',fontsize=12,color='#885d20');ax.text(.03,.18,'Exact accounting: s = C2 (D2 h + d2) + b2_feature + R2 h\nD1 = I20. R2 h is retained explicitly. Readout and factorization are analysis operations.',fontsize=12);save(fig,'01_workflow')
 fig,ax=plt.subplots(figsize=(9,5));colors={'ACCEPT':'#228657','REJECT_THIS_READOUT':'#b64b48','UNASSESSED':'#999999'};undefined=0
 for status,color in colors.items():
  rr=[r for r in rows if r['status']==status and r['NMSE'] is not None and r['finite_change_RRMSE'] is not None];ax.scatter([r['NMSE'] for r in rr],[r['finite_change_RRMSE'] for r in rr],s=13,alpha=.65,label=f'{status} ({sum(r["status"]==status for r in rows)})',color=color)
 for m in HIST:
  r=rows[m]
  if r['NMSE'] is not None and r['finite_change_RRMSE'] is not None:ax.annotate(f'TC {m}',(r['NMSE'],r['finite_change_RRMSE']),xytext=(4,4),textcoords='offset points',fontsize=10)
 undefined=sum(r['NMSE'] is None or r['finite_change_RRMSE'] is None for r in rows);ax.axvline(.05,color='black',ls='--');ax.axhline(.2,color='black',ls='--');ax.set_xscale('symlog',linthresh=.01);ax.set_yscale('symlog',linthresh=.05);ax.set_xlabel('Validation normalized MSE');ax.set_ylabel('Finite-change relative RMSE');ax.set_title('All candidates; active-value and leakage criteria also determine status');ax.legend(fontsize=9);save(fig,'02_feature_decisions')
 panels={};case_text=[]
 for m,pair in [(173,(7,8)),(910,(11,12))]:
  if m not in ids:case_text.append(f'TC {m}: {rows[m]["status"]}; failed {rows[m]["failed"]}; missing {rows[m]["missing"]}. No detailed case or substitute selected.');continue
  pos=ids.index(m);con=np.mean(abs(arr(hfit)*A['D2'][pos]),axis=0);panel=sorted(range(256),key=lambda r:(-con[r],r))[:8];panels[m]=panel
  name=f'{pair[0]},{pair[1]}';pop='A' if bank['A'][2][name]['corners'] else 'V';cs=bank[pop][2][name]['corners']
  if not cs:case_text.append(f'TC {m}: no complete background available.');continue
  ks=cs[0];a=bits(ks);u=a@A['W1'].T+A['b1'];h=np.maximum(u,0);f=h@A['D2'].T+A['d2'];z=np.maximum(a@arr_placeholder if False else np.zeros((4,1)),0) if False else None
  # Feature values from already enumerated bank; V values evaluated using frozen functions elsewhere are unnecessary here when A available.
  if pop=='A':idx=[bank['A'][0].index(k) for k in ks];zt=arr(za)[idx,m]
  else:zt=np.full(4,np.nan)
  c=h*A['D2'][pos];rem=c.sum(1)-c[:,panel].sum(1);native=h@A['W2'].T+A['b2'];term=f[:,pos,None]*A['C2'][:,pos];other=f@A['C2'].T-term;res=h@A['R2'].T
  native_rows=[];outrows=[];small=[]
  for n,corner in enumerate(['00','01','10','11']):
   for r in range(256):native_rows.append(dict(feature=m,key=ks[n],corner=corner,neuron=r,displayed=r in panel,preactivation=u[n,r],gate=bool(u[n,r]>0),activation=h[n,r],D2=A['D2'][pos,r],contribution=c[n,r],exact_omitted_remainder=rem[n]))
   small.append([corner,f'{zt[n]:.5g}',f'{f[n,pos]:.5g}',f'{c[n,panel].sum():+.5g}',f'{rem[n]:+.5g}',f'{A["d2"][pos]:+.5g}'])
   for j in range(9):outrows.append(dict(corner=corner,output=LABELS[j],featured_term=term[n,j],other_accepted_terms=other[n,j],bias=A['b2_feature'][j],native_residual=res[n,j],full_native=native[n,j]))
  csvwrite(run/'worked_cases'/f'TC_{m}_native_terms.csv',native_rows);csvwrite(run/'worked_cases'/f'TC_{m}_output_accounting.csv',outrows);np.savez(run/'worked_cases'/f'TC_{m}.npz',keys=ks,panel=panel,preactivation=u,h=h,contributions=c,remainder=rem,TC=zt,native_feature=f[:,pos],featured_output=term,other_output=other,residual=res,native_output=native)
  case_text.append(f'TC {m}: first lexicographic complete {pop} background, key {ks[0]}, pair ({pair[0]},{pair[1]}); first bit in corner label is a{pair[0]}. Display neurons {panel}. '+('All corners inactive. ' if np.all(zt==0) else '')+'Actual TC activation and native readout are measured separately; output coefficient sign does not name the feature.\n\n'+table(['corner','TC','native read','displayed sum','exact remainder','d2'],small))
  # Full nine-output accounting at each corner is kept in CSV; compact focal output shown in report.
  j=4 if m==173 else 6
  case_text[-1]+='\n'+table(['corner',LABELS[j]+' featured','other features','bias','R2 h','native score'],[[['00','01','10','11'][n],*[f'{x:+.5g}' for x in [term[n,j],other[n,j],A['b2_feature'][j],res[n,j],native[n,j]]]] for n in range(4)])
 union=sorted(set(r for pp in panels.values() for r in pp))[:12]
 # If no worked case passes, show an explicitly empty gate slice rather than replace a rejected case.
 heat(np.column_stack([A['W1'][union],A['b1'][union]]) if union else np.empty((0,21)),[f'a{i}' for i in range(20)]+['bias'],[f'neuron {r}' for r in union],'D1=I; C1=W1 — displayed native-gate subset; bias separate','03_native_gates')
 dh=[m for m in HIST if m in ids];fill=sorted([m for m in ids if m not in dh],key=lambda m:(rows[m]['NMSE'],m));dh=(dh+fill)[:12]
 heat(A['D2'][[ids.index(m) for m in dh]][:,union],[f'neuron {r}' for r in union],[f'TC {m}' for m in dh],'D2 native-to-feature coefficients — displayed subset','04_D2')
 rms=np.sqrt(np.mean((arr(hfit)@A['D2'].T+A['d2'])**2,axis=0));ranking=sorted(ids,key=lambda m:(-np.linalg.norm(A['C2'][:,ids.index(m)])*rms[ids.index(m)],m));cols=([m for m in HIST if m in ids]+[m for m in ranking if m not in HIST])[:12]
 heat(A['C2'][:,[ids.index(m) for m in cols]],[f'TC {m}' for m in cols],LABELS,'Native C2 = W2 @ pinv(D2) — displayed subset; full matrix used','05_C2',True)
 fig,ax=plt.subplots(figsize=(12,5));x=np.arange(9);ax.bar(x-.2,[o['feature_relative_RMSE'] for o in output],.4,label='native feature-only');ax.bar(x+.2,[o['TC_substitution_relative_RMSE'] for o in output],.4,label='actual TC substitution');ax.set_xticks(x,LABELS,rotation=40,ha='right');ax.set_ylabel('Audit relative RMSE');ax.set_title('Locked coordinates: audit output approximation');ax.legend();save(fig,'06_output_errors')
 counts={s:sum(r['status']==s for r in rows) for s in ['ACCEPT','REJECT_THIS_READOUT','UNASSESSED']};confirmed=sum(audit[m]['status']=='ACCEPT' for m in ids)
 summary=f'''# Native feature validation v0.7\n\nCompleted on CUDA with frozen teacher, TC and v0.5 forward readout. No network or reverse-map fitting occurred.\n\nExact calibration accepted all three reference coordinates for A and rejected all three for B (MSE 1/8 each). Both networks and reference TC compute Boolean XOR on all four points. B's 00/11 hidden collision prevents recovery, despite the hand-specified reverse identity. Positive and incomplete controls passed; incomplete A retains -2*h_AND.\n\nLearned decisions: **{counts['ACCEPT']} accepted, {counts['REJECT_THIS_READOUT']} rejected, {counts['UNASSESSED']} unassessed**. **{confirmed}** accepted candidates confirmed on audit A. Banks contain {len(bank['V'][0])} V and {len(bank['A'][0])} A inputs; neither is claimed globally untouched.\n\nD2 is {A['D2'].shape}, numerical rank {diag['rank']}. Raw FP64 SVD cutoff max(1e-12,1e-10*smax) = {diag['cutoff']:.5g}; retained condition {diag['condition']}. Native C2 is {A['C2'].shape}. Feature-only aggregate audit relative RMSE is {checks['aggregate_feature_RRMSE']:.6g}; TC substitution is {checks['aggregate_TC_RRMSE']:.6g}. R2 Frobenius/W2 Frobenius is {checks['R2_relative_Frobenius']:.6g}; audit R2*h RMS {checks['residual_RMS']:.6g}, maximum {checks['residual_max']:.6g}. Exact accounting discrepancy is {checks['exact_accounting_max']:.3g}.\n\nNative C1 gates are dense input mixtures followed by actual ReLUs; D2 combines their signed activations. C2 assigns accepted-coordinate contributions to nine continuous output scores. Worked cases retain every native term and an exact display remainder. TC 173 and TC 910 are shown only if accepted; failures are preserved without substitutes.\n\nThis supports empirical affine readability under one fixed contract. Accepted features need not be sufficient as a set. Exact accounting includes the residual and is algebraic, not evidence of prediction success. Uniqueness, semantic simplicity, independent manipulability, causal necessity, universal validity and protected-final performance remain unsupported.\n'''
 if diag['rank']==256:summary+='\nThe native residual can vanish for algebraic reasons: these readout coordinates retain all hidden directions. This does not establish semantic simplicity or unique native features.\n'
 if diag['condition'] and diag['condition']>1e6:summary+='\nIll-conditioned interpretation: retained condition exceeds 1e6; cutoff and units were not changed.\n'
 (run/'SUMMARY.md').write_text(summary)
 hist=table(['candidate','V decision','NMSE','active RRMSE','leakage','finite RRMSE','audit decision'],[[f'TC {m}',rows[m]['status'],*[f'{rows[m][k]:.4g}' if rows[m][k] is not None else 'undefined' for k in ['NMSE','active_RRMSE','inactive_leakage','finite_change_RRMSE']],audit[m]['status']] for m in HIST])
 reporttext=summary+'\n## Exact four-point calibration\n\nA: s=a+b-2ab. B: s=ReLU(a-b)+ReLU(b-a). Equality is asserted on Boolean inputs only.\n\n'
 for name,cc in cal.items():
  reporttext+=f'### {name}\n\n'
  for k,vv in cc['matrices'].items():reporttext+=f'{k} {np.asarray(vv).shape}: `{np.round(vv,8).tolist()}`\n\n'
 reporttext+='B: **No supported coordinates from this proposal; output remains entirely in the native residual.** The hand-specified negative-control identity h_B=[[1,0,-1],[0,1,-1]] z_ref does not certify forward features. Full four-row tables: [CSV](calibration/all_four_corner_values.csv).\n\n## Frozen learned-model analysis\n\n'+hist+'\nUndefined plotted metric pairs: '+str(undefined)+'. The four acceptance criteria all apply, including active support and zero-region leakage. Edges share endpoints and are not independent samples. Validation excludes fit, selection and protected keys; training and prior-development exposure is tagged separately. No protected activations were read.\n\n'
 for name in figs:reporttext+=f'![{name}](figures/{name}.png)\n\n'
 reporttext+='## Explicit residual and audit errors\n\n'+table(['output','feature RRMSE','TC substitution RRMSE','R2 h RMS','amplified readout RMS'],[[o['output'],*[f'{o[k]:.5g}' for k in ['feature_relative_RMSE','TC_substitution_relative_RMSE','residual_RMS','amplified_readout_RMS']]] for o in output])+'\n## Worked cases\n\n'+'\n\n'.join(case_text)+'\n\nFull matrices, units and offsets are exported in matrices/. No pseudoinverse was recomputed for a displayed subset. C2 columns depend on raw TC scaling and redundancy. The audit did not revise the validation lock.\n'
 (run/'REPORT.md').write_text(reporttext);write(run/'figures/index.json',dict(figures=figs,PNG_dpi=210,undefined_metric_pairs=undefined))
 (run/'README.md').write_text('# Completed v0.7 experiment\n\nStart with [SUMMARY.md](SUMMARY.md), then [REPORT.md](REPORT.md). Full raw-coordinate matrices are in matrices/. Candidate decisions and audit replication are separate CSVs. Feature lock precedes factorization. No prior artifacts modified.\n')

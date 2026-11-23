"""Deterministic report-only display corrections from saved Phase I arrays."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import read,HIST,LABELS
from .reports import save

def repair(root):
 p=root/'phase_I';sl=read(p/'paths/guided/sequence_lock.json');old=sl['sequence'][0];new=sl['sequence'][sl['representative']];a=np.load(p/'paths/guided/checkpoints/add_000/C2.npy');b=np.load(p/f'paths/guided/checkpoints/add_{sl["representative"]:03}/C2.npy');ids=[m for m in HIST if m in old and m in new];v=np.column_stack([b[:,new.index(m)]-a[:,old.index(m)] for m in ids]);fig,ax=plt.subplots(figsize=(10,5));lim=max(1e-12,abs(v).max());im=ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(len(ids)),[f'TC {m}' for m in ids],rotation=45,ha='right');ax.set_yticks(range(9),LABELS);ax.set_title('C2 after minus before: identical shared historical IDs; displayed subset');fig.colorbar(im,ax=ax,label='raw signed coefficient change');save(fig,p/'figures','05_C2_changes')
 if len(sl['sequence'])>1:
  m=sl['sequence'][1][-1];a=np.load(p/'worked_cases'/f'TC_{m}.npz');panel=a['panel'];v=np.column_stack([a['W1'][panel],a['b1'][panel],a['M_row'][panel]]);fig,ax=plt.subplots(figsize=(12,4));lim=abs(v).max();im=ax.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_xticks(range(22),[f'a{i}' for i in range(20)]+['bias','D2 row'],rotation=50);ax.set_yticks(range(len(panel)),[f'neuron {r}' for r in panel]);ax.axvline(19.5,color='black');ax.axvline(20.5,color='black');ax.set_title(f'TC {m}: first provisional addition; W1=C1 gates and unchanged D2 row')
  for i,r in enumerate(panel):ax.text(21,i,f'{a["M_row"][r]:+.3g}',ha='center',va='center',fontsize=8)
  fig.colorbar(im,ax=ax,label='raw signed coefficient');save(fig,p/'figures','06_native_gates')
 paths={f.name:list(csv.DictReader(open(f/'reused_audit_metrics.csv'))) for f in (p/'paths').iterdir()};fig,ax=plt.subplots(figsize=(10,4.5));colors={'guided':'#126f9a','reliability':'#d07e21'}
 for n in ['guided','reliability']:
  rr=paths[n]
  for k,style in [('rho_L','-o'),('rho_T','--s')]:ax.plot(range(len(rr)),[float(r[k]) for r in rr],style,color=colors[n],label=f'{n} {k}')
 rr=[v for k,v in paths.items() if k.startswith('random')]
 for k,style in [('rho_L','-^'),('rho_T','--v')]:
  xs=range(max(map(len,rr)));vals=[[float(v[i][k]) for v in rr if len(v)>i] for i in xs];ax.plot(xs,[np.median(v) for v in vals],style,color='gray',label=f'random median {k}');ax.fill_between(xs,[min(v) for v in vals],[max(v) for v in vals],alpha=.14,color='gray')
 ref=read(p/'static_references/all_candidates/A_metrics.json');num=1024-len(old);ax.scatter([num],[ref['rho_L']],marker='X',s=80,label='all 1024, unvalidated algebraic reference');ax.set_xscale('symlog',linthresh=3);ax.set_xticks([0,1,2,3,num],[0,1,2,3,num]);ax.set_xlabel('Added rows; path additions provisional, all-1024 point unvalidated');ax.set_ylabel('Error ratio');ax.set_title('Reused audit A: correspondence rho_L and substitution rho_T');ax.legend(fontsize=8);save(fig,p/'figures','02_amplification')

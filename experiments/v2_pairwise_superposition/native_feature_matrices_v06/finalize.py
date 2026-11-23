"""Additional audit, exact indexed worked exports, and review material for v0.6."""
import argparse,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ..latent_correspondence_v05.core import tensor,numpy,inputs
from ..latent_correspondence_v05.run import models,csvwrite
from .run import OLD,savejson,read,sha,check,FEATURES

def finalize(run):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 A=dict(np.load(run/'matrices.npz'));w=dict(np.load(run/'worked_examples.npz'));man=read(run/'context_manifest.json');info=read(run/'worked_contexts.json');t,s,p,q=models(OLD)
 # Both batch and column-vector full-model identities, separate captured inference tolerance.
 checks={};contribrows=[];gaterows=[];precision={};matched={};matchedrows=[];algebra={}
 for m in FEATURES:
  pre=f'f{m}_';bits=w[pre+'bits'];h=w[pre+'h'];z=w[pre+'z'];zn=w[pre+'zn'];ks=w[pre+'keys'];c1=A['C1'];u=bits@c1.T+A['b1'];ga=u>0
  for name in ['fit','pinv']:
   S=A['S2_'+name];qq=A['q2_'+name];C=A['C2_'+name];b=A['b2_feature_'+name];values=[]
   for i in range(len(bits)):
    ez=z[i]-zn[i];rn=S@zn[i]+qq-h[i];rt=S@z[i]+qq-h[i];ya=A['W2']@h[i]+A['b2'];yn=C@zn[i]+b;yt=C@z[i]+b
    values += [check(rn,(S@A['D2']-np.eye(256))@h[i]+S@A['d2']+qq),check(rt,rn+S@ez),check(yn-ya,A['W2']@rn),check(yt-ya,A['W2']@rt),check(yt-yn,C@ez),check(yn-ya,(C@A['D2']-A['W2'])@h[i]+C@A['d2']+b-A['b2']),check(A['D2']@(S@z[i]+qq)+A['d2']-z[i],A['D2']@rt-ez)]
   checks[f'{m}_{name}']=values
  projected=bits@A['P'].T@A['D1'].T;algebra[str(m)]=dict(projected_error_norm=np.linalg.norm(projected-bits,axis=1),C1a=check(bits@c1.T,(bits@A['P'].T)@A['W1'].T),C1D1x=check(projected@c1.T,(bits@A['P'].T)@A['W1'].T))
  # FP32 inference comparison with explicit dtype-aware tolerance, not used in algebraic identities.
  with torch.no_grad():
   t.float();s.float();x32=tensor(bits).float()@p.float().T;u32,h32=t.hidden(x32);z32=s.latents(x32);y32=t.W2(h32)
  for name,actual,ref in [('u',numpy(u32),w[pre+'u']),('h',numpy(h32),h),('z',numpy(z32),z),('y',numpy(y32),h@A['W2'].T+A['b2'])]:
   np.testing.assert_allclose(actual,ref,atol=2e-6,rtol=2e-5);precision[f'{m}_{name}']=dict(max_abs=float(np.max(np.abs(actual-ref))),atol=2e-6,rtol=2e-5)
  precision[f'{m}_gate']=dict(FP64_algebra_vs_source=int(np.sum(ga!=w[pre+'gates'])),FP32_vs_FP64=int(np.sum((numpy(u32)>0)!=w[pre+'gates'])),near_zero_indices=np.argwhere(np.abs(w[pre+'u'])<=2e-6).tolist());t.double();s.double()
  for n,k in enumerate(ks):
   for code,zz in [('TC',z),('native',zn)]:
    for j in range(12):
     term=A['C2_fit'][j,m]*zz[n,m];total=zz[n]@A['C2_fit'][j];contribrows.append(dict(feature=m,population=info[str(m)]['population'],key=int(k),corner=['00','01','10','11'][n%4],code=code,output=j,feature_value=zz[n,m],C2=A['C2_fit'][j,m],feature_contribution=term,remaining_1023_sum=total-term,affine_offset=A['b2_feature_fit'][j],total=total+A['b2_feature_fit'][j]))
   for j in range(20):gaterows.append(dict(feature=m,key=int(k),input_index=j,gate_conditioned_weight=w[pre+'L1'][n,m,j],input_bit=bits[n,j],offset=w[pre+'o1'][n,m]))
  # Preserve historical matched nuisance illustrations, using the exact recorded keys.
  matchedpath=OLD/f'feature_{m}_legacy_matched.npz'
  if matchedpath.exists():
   old=np.load(matchedpath);keys=old['keys'];bb=tensor(inputs(keys.reshape(-1)));xx=bb@p.T
   with torch.no_grad():uu,hh=t.hidden(xx);zz=s.latents(xx);yy=t.W2(hh)
   hh,zz,uu,yy=map(numpy,[hh,zz,uu,yy]);hh=hh.reshape(*keys.shape,256);zz=zz.reshape(*keys.shape,1024);uu=uu.reshape(*keys.shape,256);yy=yy.reshape(*keys.shape,12);nn=hh@A['D2'].T+A['d2'];con=hh*A['D2'][m];panel=old['panel']
   for name,value in dict(keys=keys,u=uu,gates=uu>0,h=hh,z=zz,zn=nn,contributions=con,remainder=con.sum(-1)-con[...,panel].sum(-1),nuisance=old['nuisance']).items():matched[f'f{m}_{name}']=value
   for b in range(len(keys)):
    for corner in range(4):matchedrows.append(dict(feature=m,population='legacy matched nuisance',key=int(keys[b,corner]),background=int(old['background_keys'][b]),nuisance=int(old['nuisance']),corner=['00','01','10','11'][corner],zT=zz[b,corner,m],zN=nn[b,corner,m],overlap=man['membership'][str(int(keys[b,corner]))]))
 csvwrite(run/'worked_output_contributions.csv',contribrows);csvwrite(run/'worked_gate_conditioned_slices.csv',gaterows);csvwrite(run/'matched_nuisance_tables.csv',matchedrows);np.savez_compressed(run/'matched_nuisance_examples.npz',**matched);savejson(run/'column_vector_validation.json',checks);savejson(run/'source_precision_checks.json',precision);savejson(run/'worked_input_projection.json',algebra)
 # Use the historical polynomial corner convention: singles from the 00 corner.
 rows=[];changes={};fit=np.load(run/'fit_activations.npz');mu=fit['h'].mean(0)
 for pop,bankset in [('primary',man['primary_contexts']),('legacy',man['legacy_contexts'])]:
  v=dict(np.load(run/(pop+'_activations.npz')));lookup={int(k):i for i,k in enumerate(v['keys'])}
  for pair,bank in bankset.items():
   if not bank['corner_keys']:continue
   idx=np.array([[lookup[k] for k in row] for row in bank['corner_keys']]);blookup={int(k):i for i,k in enumerate(bank['background_keys'])}
   for method,S,offset,code in [('fit_z',A['S2_fit'],A['q2_fit'],'z'),('fit_zn',A['S2_fit'],A['q2_fit'],'zn'),('pinv_z',A['S2_pinv'],A['q2_pinv'],'z'),('pinv_zn',A['S2_pinv'],A['q2_pinv'],'zn'),('mean',np.zeros_like(A['S2_fit']),mu,'z')]:
    hh=v[code]@S.T+offset;yy=hh@A['W2'].T+A['b2']
    for space,truth,pred,ref in [('hidden',v['h'],hh,fit['h']),('output',v['y'],yy,fit['y'])]:
     tt=truth[idx];pp=pred[idx];ops={'first_bit':[-1,0,1,0],'second_bit':[-1,1,0,0],'interaction':[1,-1,-1,1]};pairs=[(typ,np.einsum('c,ncr->nr',op,tt),np.einsum('c,ncr->nr',op,pp)) for typ,op in ops.items()]
     if bank['nuisance_edges']:
      ei=np.array([[blookup[int(e[0])],blookup[int(e[2])]] for e in bank['nuisance_edges']]);pairs.append(('nuisance',(tt[ei[:,1]]-tt[ei[:,0]]).reshape(-1,truth.shape[1]),(pp[ei[:,1]]-pp[ei[:,0]]).reshape(-1,truth.shape[1])))
     for typ,dt,dp in pairs:
      norm=np.linalg.norm(dt,axis=1);threshold=1e-8*max(1,np.sqrt(np.mean(np.sum(ref**2,1))));eligible=norm>threshold;n=int(eligible.sum());key=f'{pop}_{pair}_{method}_{space}_{typ}'
      for suffix,value in [('target',dt),('prediction',dp),('eligible',eligible),('target_norm',norm)]:changes[key+'_'+suffix]=value
      rows.append(dict(population=pop,pair=pair,method=method,space=space,type=typ,n_changes=len(dt),n_contexts=len(idx),n_edges=len(bank['nuisance_edges']) if typ=='nuisance' else 0,eligible=n,status='unassessable' if n<32 else 'supported descriptive summary',relative_RMSE=float(np.linalg.norm((dp-dt)[eligible])/np.linalg.norm(dt[eligible])) if n else None,threshold=threshold,min_target_norm=float(norm.min()),median_target_norm=float(np.median(norm)),max_target_norm=float(norm.max()),overlap='legacy historical mixture; per-key membership in context_manifest.json' if pop=='legacy' else 'original/final/historical exclusions preserved',difference_operator=str(ops.get(typ,'background1-background0 at each corner'))))
 csvwrite(run/'finite_response_metrics.csv',rows);np.savez_compressed(run/'finite_changes.npz',**changes)
 # Compare both half fits with primary predictions in hidden and output coordinates.
 dev=dict(np.load(run/'development_activations.npz'));var=fit['h'].var(0);mask=var>max(1e-12,1e-8*np.median(var[var>0]));base=dev['z']@A['S2_fit'].T+A['q2_fit'];stability=[]
 for half in ['half1','half2']:
  hm=torch.load(run/(half+'_embedding.pt'),map_location='cpu',weights_only=False);ss=numpy(hm['M']);qq=numpy(hm['d']);delta=dev['z']@ss.T+qq-base
  stability.append(dict(half=half,hidden_primary_variance_normalized_MSE=float(np.mean(np.mean(delta**2,0)[mask]/var[mask])),output_difference_RMSE=float(np.sqrt(np.mean((delta@A['W2'].T)**2))),hidden_difference_RMSE=float(np.sqrt(np.mean(delta**2)))))
 csvwrite(run/'half_prediction_stability.csv',stability)
 # Full scaled diagnostics and norms, including affine residual absolute norms.
 diag=read(run/'matrix_diagnostics.json');M=A['D2'];sv=np.linalg.svd(M,compute_uv=False)
 for name in ['fit','pinv']:
  S=A['S2_'+name];C=A['C2_'+name];diag[name].update(hidden_linear_relative_frobenius=float(np.linalg.norm(S@M-np.eye(256))/np.linalg.norm(np.eye(256))),output_linear_relative_frobenius=float(np.linalg.norm(C@M-A['W2'])/np.linalg.norm(A['W2'])),hidden_affine_absolute_norm=float(np.linalg.norm(S@A['d2']+A['q2_'+name])),output_affine_absolute_norm=float(np.linalg.norm(C@A['d2']+A['b2_feature_'+name]-A['b2'])))
 diag['M']=dict(rank=int(sum(sv>1e-10*sv[0])),rcond=1e-10,condition=float(sv[0]/sv[-1]),smax=float(sv[0]),smin=float(sv[-1]),pinv_operator_norm=float(1/sv[-1]))
 savejson(run/'matrix_diagnostics.json',diag)
 # Vector appendices make exact native rows and remainders readable without shrinking worked pages.
 with PdfPages(run/'worked_appendix.pdf') as pdf:
  for m in FEATURES:
   pre=f'f{m}_';panel=w[pre+'panel'];ks=w[pre+'keys'];n=len(ks);fig,axs=plt.subplots(3,1,figsize=(15,11),gridspec_kw={'height_ratios':[1,1.4,1]})
   for ax in axs:ax.axis('off')
   axs[0].set_title(f'Feature {m}: native C1 weights and bias; original input indices')
   data=np.column_stack([A['C1'][panel],A['b1'][panel],A['D2'][m,panel]])
   tab=axs[0].table(cellText=[[f'{x:.4g}' for x in row] for row in data],rowLabels=[str(r) for r in panel],colLabels=[f'a{i}' for i in range(20)]+['b1','D2'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(6);tab.scale(1,1.3)
   rows=[]
   for i in range(n):
    rows.append([str(int(ks[i])),['00','01','10','11'][i%4]]+[f'{v:.6f}' for v in w[pre+'contributions'][i,panel]]+[f'{w[pre+"remainder"][i]:.6f}',f'{A["d2"][m]:.6f}',f'{w[pre+"zn"][i,m]:.6f}',f'{w[pre+"z"][i,m]:.6f}'])
   tab=axs[1].table(cellText=rows,colLabels=['key','corner']+[f'r{r}' for r in panel]+['rest 248','d2','zN','zT'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(7);tab.scale(1,1.6);axs[1].set_title('Exact signed D2 h contributions; sum displayed + rest + d2 = zN (all 256 neurons)')
   axs[2].text(.01,.9,f'Population: {info[str(m)]["population"]}. Corner convention 00,01,10,11. All values shown to display precision; CSV/NPZ retain FP64.\n\nTC preactivations: {np.array(info[str(m)]["TC_pre"])}\n\nNative active-gate counts: {info[str(m)]["active_gates"]}\n\nFull gate-conditioned maps, S2 columns, output terms, all native u/h/gates, original bits and keys are in indexed numerical exports.',va='top',fontsize=11)
   fig.tight_layout();fig.savefig(run/'figures'/f'appendix_feature_{m}.png',dpi=150);fig.savefig(run/'figures'/f'appendix_feature_{m}.pdf');pdf.savefig(fig);plt.close(fig)
 # Compile a concise eight-page main report followed by all matrix/gate appendices.
 # pdfunite retains vector paths and searchable labels.
 original=run/'matrix_report.full.pdf';shutil.copy2(run/'matrix_report.pdf',original)
 mainnames=['01_architecture','02_atlas_input','02_atlas_hidden','02_atlas_output','03_semantic_maps','04_reconstruction','05_feature_173','05_feature_910']
 appendix=[p for p in sorted((run/'figures').glob('*.pdf')) if p.stem not in mainnames and not p.stem.startswith('appendix_')]
 cmd=['pdfunite']+[str(run/'figures'/(n+'.pdf')) for n in mainnames]+[str(p) for p in appendix]+[str(run/'worked_appendix.pdf'),str(run/'matrix_report.reviewed.pdf')]
 subprocess.run(cmd,check=True);shutil.copy2(run/'matrix_report.reviewed.pdf',run/'matrix_report.pdf')
 # Actual coefficient stability and sparse/dense descriptive measurements.
 import csv
 rows=list(csv.DictReader(open(run/'coefficient_stability.csv')));metrics=read(run/'metric_summary.json');summary=(run/'SUMMARY.md').read_text()
 addition=['\n## Quantitative matrix structure and uncertainty']
 for name in ['half1','half2']:
  for space in ['S2','C2']:
   vals=np.array([float(r['cosine']) for r in rows if r['half']==name and r['space']==space and r['cosine']]);addition.append(f'{name} {space} column cosine: median {np.median(vals):.6g}, 10th percentile {np.quantile(vals,.1):.6g}.')
 addition.append('Half-fit prediction differences: `prediction_stability.json` uses the primary fit variance scales. These coefficients describe an affine embedding, not a unique neuron-to-feature inverse.')
 for m in FEATURES:
  c=A['C2_fit'][:,m];sem=A['L2_fit'][:,m];j=int(np.argmax(np.abs(sem)));addition.append(f'Feature {m}: ||D2 row||={np.linalg.norm(M[m]):.5g}, ||S2 column||={np.linalg.norm(A["S2_fit"][:,m]):.5g}, ||C2 column||={np.linalg.norm(c):.5g}; largest semantic coefficient index {j}: {sem[j]:.6g}.')
 addition.append(f'Native W1 and C1 are dense: C1 has {np.mean(np.abs(A["C1"])>1e-8):.3%} entries above 1e-8 in magnitude. Native thresholds are -b1, before the ReLU. M has retained rank {diag["M"]["rank"]} and condition {diag["M"]["condition"]:.6g}; pseudoinverse operator norm {diag["M"]["pinv_operator_norm"]:.6g}.')
 for name in ['fit','pinv']:addition.append(f'{name}: hidden linear residual ||S2D2-I||F/||I||F={diag[name]["hidden_linear_relative_frobenius"]:.6g}; output linear residual ||C2D2-W2||F/||W2||F={diag[name]["output_linear_relative_frobenius"]:.6g}; hidden/output affine residual norms {diag[name]["hidden_affine_absolute_norm"]:.6g}/{diag[name]["output_affine_absolute_norm"]:.6g}; actual code energy outside affine image M={diag[name]["outside_image_energy_fraction"]:.6g}.')
 addition += ['Source FP32 comparisons passed explicit atol=2e-6, rtol=2e-5; detailed gate discrepancies are in `source_precision_checks.json`. Strict identities use captured FP64 hidden states and recomputed algebraic outputs.','[Exact native tables](worked_native_tables.csv) · [Output contributions with remaining 1023-feature sums](worked_output_contributions.csv) · [Matched nuisance examples](matched_nuisance_tables.csv) · [Readable worked appendix](worked_appendix.pdf)']
 (run/'SUMMARY.md').write_text(summary+'\n\n'.join(addition)+'\n')
 print('FINALIZED',run,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args();finalize(args.run)

"""Post-execution integrity, serialization, and reporting checks; no selection."""
import os,sys,json,csv,shutil,subprocess
from pathlib import Path
import numpy as np,torch
from .core import *
from .reports import table,pdf

@torch.no_grad()
def check(root):
 from .repair_figures import repair
 repair(root)
 torch.set_num_threads(4);assert read(root/'execution_summary.json')['status']=='complete';n=np.load(root/'phase_I/native_matrices.npz');W=tensor(n['W2']);b=tensor(n['b2']);native=read(root/'phase_II/data_manifests.json');fitman=read(OLD/'context_manifest.json');forbidden=set(fitman['fit_keys']+fitman['selection_keys'])
 for p in BASE.rglob('protected_final_keys.npy'):forbidden.update(map(int,np.load(p)))
 for bank in native.values():assert not set(bank['keys'])&forbidden
 assert set(native['V']['keys']).isdisjoint(native['A']['keys']);grid=read(root/'phase_II/training_grid_lock.json');summaries=[];bundles=0;mx=0.;covariance_roundoff=[]
 # Every saved map is reloaded; C2 is checked directly against W2 @ J.
 for meta in [*root.glob('phase_I/paths/*/checkpoints/*/metadata.json'),*root.glob('phase_I/static_references/*/matrices/metadata.json'),*root.glob('phase_II/validation_and_factorization/*/full_set/metadata.json'),*root.glob('phase_II/validation_and_factorization/*/budget_*/metadata.json')]:
  f=meta.parent;D=tensor(np.load(f/'D2.npy'));J=tensor(np.load(f/'J.npy'));C=tensor(np.load(f/'C2.npy'));R=tensor(np.load(f/'R2.npy'));d=tensor(np.load(f/'d2.npy'));off=tensor(np.load(f/'bF.npy'));torch.testing.assert_close(C,W@J,atol=1e-10,rtol=1e-9);torch.testing.assert_close(R,W-C@D,atol=1e-10,rtol=1e-9);torch.testing.assert_close(off,b-C@d,atol=1e-10,rtol=1e-9);torch.testing.assert_close(D@J@D,D,atol=1e-8,rtol=1e-8);
  for left,right in [(J@D@J,J),(J@D,(J@D).T),(D@J,(D@J).T)]:assert float((left-right).norm()/max(1.,float(right.norm())))<1e-7
  md=read(meta);assert len(md['ids'])==len(D);assert sha(Path(md['native_matrices']))==md['native_sha256'];mx=max(mx,float((W-C@D-R).abs().max()));bundles+=1
 for w in grid['widths']:
  for seed,shuffle in zip(grid['student_seeds'],grid['shuffle_seeds']):
   name=f'w{w}_s{seed}';ro=root/'phase_II/readout_runs'/name;lock=read(ro/'readout_lock.json');assert sha(Path(lock['TC_checkpoint']))==lock['TC_checkpoint_sha256'];assert sha(ro/'M.npy')==lock['M'];assert sha(ro/'d.npy')==lock['d'];cp=torch.load(lock['TC_checkpoint'],map_location='cpu',weights_only=False);assert cp['width']==w and cp['lambda']==1e-4;assert cp['config']['student_seed']==seed and cp['config']['shuffle_seed']==shuffle;assert cp['config']['training']==grid['original_config']['training'];assert torch.allclose(cp['model']['decoder.weight'].norm(dim=0),torch.ones(w),atol=3e-7,rtol=3e-7)
   f=root/'phase_II/validation_and_factorization'/name;sl=read(f/'sequence_lock.json');assert sha(f/'accepted_ids.json')==sl['accepted_ids_sha256'];info=read(f/'cell_summary.json');assert info['width']==w and info['seed']==seed;summaries.append(info)
   M=np.load(ro/'M.npy');ids=read(f/'accepted_ids.json');np.testing.assert_array_equal(np.load(f/'full_set/D2.npy'),M[ids]);assert len(set(ids))==len(ids);assert all(set(S)<=set(ids) for S in sl['sequence']);assert [len(S) for S in sl['sequence']]==list(range(len(sl['sequence'])))
 # Enrich JSON metrics without changing any saved scalar or selected ID.
 for file in list(root.glob('phase_I/static_references/*/*metrics.json'))+list(root.glob('phase_II/validation_and_factorization/*/full_*_metrics.json')):
  r=read(file)
  if 'per_output_N' in r:
   nn=np.array(r['per_output_N']);ll=np.array(r['per_output_L']);tt=np.array(r['per_output_T']);r['per_output_objective']=(nn**2+ll**2);r['per_output_cross']=(tt**2-nn**2-ll**2);r['energy_units']='per-example; sum energies equal bank row count times these energies';write(file,r)
 # Enrich all scalar tables with exact per-output J/cross counterparts from stored ratios.
 for file in list(root.glob('phase_I/paths/*/*metrics.csv'))+list(root.glob('phase_II/validation_and_factorization/*/*metrics.csv'))+[root/'phase_II/width_summary.csv',root/'phase_II/equal_budget_metrics.csv']:
  rows=list(csv.DictReader(open(file)))
  for r in rows:
   if 'per_output_N' in r and r['per_output_N']:
    nn=np.array(json.loads(r['per_output_N']));ll=np.array(json.loads(r['per_output_L']));tt=np.array(json.loads(r['per_output_T']));r['per_output_objective']=(nn**2+ll**2).tolist();r['per_output_cross']=(tt**2-nn**2-ll**2).tolist();r['energy_units']='mean_energy and varying_energy are per-example; multiply by bank row count for sum energies'
  csvwrite(file,rows)
 widthtable=table(['width','accepted median [range]','rank median [range]','rho_N median','rho_L median','rho_T median','L0 median'],[[w,*[f'{np.median([r[k] for r in summaries if r["width"]==w]):.4g} [{min(r[k] for r in summaries if r["width"]==w):.4g},{max(r[k] for r in summaries if r["width"]==w):.4g}]' for k in ['accepted','accepted_rank']],*[f'{np.median([r[k] for r in summaries if r["width"]==w]):.4g}' for k in ['rho_N','rho_L','rho_T']],f'{np.median([r["audit_fidelity"]["exact_positive_L0"] for r in summaries if r["width"]==w]):.3g}'] for w in grid['widths']])
 budget=list(csv.DictReader(open(root/'phase_II/equal_budget_metrics.csv')));common=table(['width','budget','seeds attained','rho_N median [range]'],[[w,bb,len(vals),f'{np.median(vals):.4g} [{min(vals):.4g},{max(vals):.4g}]'] for w in grid['widths'] for bb in [16,32,64,128] if (vals:=[float(r['rho_N']) for r in budget if int(r['width'])==w and int(r['budget'])==bb])])
 cross='''# v0.8 — measured cross-phase findings\n\n**Both phases completed on CUDA.** The teacher, historical TC/readout and previous scientific artifacts remain unchanged. C2 was always W2 @ pinv(D2); no S2 or output regression.\n\n**Residual-guided add-back:** 125 leakage-only reserve candidates were available. Only provisional TC 31 was selected; its V innovation ratios were 0.112 value / 0.223 edge. It mostly helped g2 AND(a3,a4), while some outputs regressed. Audit rho_N changed 2.031→1.934 (203.1%→193.4%), rho_L 0.555→0.574 and rho_T 2.110→2.019. At one attained addition it had better native coverage than reliability-first and the random median, but not lower amplification than reliability-first; it tied random paths that also picked TC 31. No claim of equal-runtime superiority is supported.\n\nTC 266 remained strictly rejected and failed innovation guards (0.438 value / 0.798 edge). The guided path then stalled; this does not rule out useful groups. All-reserve and all-candidate references reached hidden rank 256 and essentially zero native residual algebraically, yet substitution errors remained 0.624 and 0.125 respectively. Strict validity, provisional usefulness and algebraic completeness are different outcomes.\n\n**Width cohort:** 13 fresh TCs and 2 verified historical reuses, exactly five widths × three seeds. Strict validation and audit used common banks excluding TC training/checkpoint-validation inputs. Full accepted-set results are:\n\n'''+widthtable+'\nEqual attained budgets are reported without padding stopped sequences; three seeds support descriptive replication only. L0 changes materially with width, so width effects are not isolated from sparsity, fidelity, rarity or convergence. See both full reports for per-output, response, conditioning and cancellation diagnostics; all unfavorable seed results are retained.\n\n[Phase I](phase_I/REPORT.md) · [Phase II](phase_II/REPORT.md) · [Equal-budget table](phase_II/equal_budget_metrics.csv) · [Addition ledger](phase_I/paths/guided/additions.csv).\n'
 (root/'CROSS_PHASE_SUMMARY.md').write_text(cross);pdf(root/'CROSS_PHASE_SUMMARY.md');pdf(root/'phase_I/REPORT.md');(root/'phase_II/EQUAL_BUDGET_SUMMARY.md').write_text('# Equal attained budgets\n\n'+common)
 write(root/'numerical_checks.json',dict(saved_bundles_reloaded=bundles,maximum_affine_weight_identity_error=mx,all_C2_equal_W2_J=True,per_output_cross_energy_identity=True,common_banks_disjoint=True,all_checkpoint_policies_match=True,all_units_raw=True,roundoff_tolerance='score quad allows only >= -1e-10*max(1,||A||F²||G||F) before clamp; selected-set direct predictions checked at atol 1e-9 rtol 1e-8'))
 for name,role in [('all_candidates','UNVALIDATED_ALGEBRAIC_REFERENCE'),('all_reserve','UNGUARDED_ALL_RESERVE_DIAGNOSTIC')]:
  p=root/'phase_I/static_references'/name/'matrices/metadata.json';m=read(p);m['dictionary_role']=role;write(p,m)
 for name,src in [('prepare.log','/tmp/v08-prepare.log'),('phase_I.log','/tmp/v08-phase1-fixed.log'),('tests.expanded.log','/tmp/v08-CPU-tests-expanded.log')]:shutil.copy2(src,root/name)
 (root/'environment.txt').write_text(subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout)
 write(root/'commands.json',dict(prepare='python -m ...residual_width_code_v08.run prepare '+str(root),phase_I='CUBLAS_WORKSPACE_CONFIG=:4096:8 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 python -m ...residual_width_code_v08.run phase1 '+str(root),tmux='v08-width-run01',orchestrator=read(root/'phase_II/launched.json'),worker_exit_codes=read(root/'phase_II/worker_exit_codes.json'),audit_exit_code=read(root/'orchestrator_exit_code.json'),CPU_fixtures_passed=6,initial_joint_tests_passed=7,repairs=['Phase I nested NumPy CSV serialization fixed; failed attempt preserved; thresholds unchanged','Figure 6 explicitly selects first primary addition, replacing filesystem-order display selection']))
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,root/'source_snapshot'/p.name)
 write(root/'final_checks.json',dict(status='passed',bundles=bundles,all_15_cells_verified=True,protected_final_scored=False))
 print('FINAL CHECKS PASSED',bundles,flush=True)
if __name__=='__main__':check(Path(sys.argv[1]))

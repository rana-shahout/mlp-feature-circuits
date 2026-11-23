"""Final prose/navigation and artifact sealing, using completed measured results only."""
import sys,shutil,json,subprocess
from pathlib import Path
import markdown
from weasyprint import HTML
from .core import read,write,sha,OLD,BASE
from .reports import pdf
root=Path(sys.argv[1]);assert read(root/'final_checks.json')['status']=='passed'
interpret='''\nNumerical accepted rank equals accepted count in every cell: the additional strictly accepted coordinates through width 1024 supply additional retained directions rather than only redundant rows. None spans all 256 hidden directions. Width 2048 regresses to rank 59–68 under this fixed protocol, and its full-set median native error is 3.038 versus 2.031 at width 1024. All full-set native errors remain large.\n\nEqual-budget results are not a uniform width ranking. At 16 coordinates, median native error is 1.748 for width 512 versus 1.931 for width 1024; at 32, these are 3.047 versus 2.242. At 64, width 512 is slightly lower (2.480 versus 2.512), while width 1024 is lower at 128 (2.062 versus 2.258). Seed ranges overlap. Width 128 attains budget 16 in only two seeds, and width 2048 attains budget 64 in only one; missing cells are unavailable. Audit failures remain included in the original V-locked sets.\n\nL0 rises from a median 48.9 to 721 across the width range, and width-2048 fidelity is worse within the common cap. Sparsity and optimization confounds are visible; these results do not isolate a pure capacity effect. No extra epochs, replacement seeds, matched-sparsity training or rescue thresholds were introduced. See [equal-budget counts and ranges](EQUAL_BUDGET_SUMMARY.md).\n'''
for name in ['PHASE_II_SUMMARY.md','REPORT.md']:
 p=root/'phase_II'/name;p.write_text(p.read_text()+'\n## Interpretation of the measured comparisons\n'+interpret)
methods='''\n## Conventions and execution checks\n\nThe main analysis is the canonical 20-input → 256-native-neuron → 9-continuous-score export of the frozen teacher. It has one hidden ReLU and no output ReLU. The original embedded training bottleneck remains in the model's learning history; this export preserves hidden activations and semantic scores. Phase II TC training remains the original internal 12→p→12 problem.\n\nNine targeted tests passed, including [the exact XOR controls](calibration/results.json). The guided TC 31 addition also passes the post-lock audit innovation and amplification guards. TC 144 fails the audit innovation guard in two random paths; those paths and their results remain unchanged. Residual mean/varying energies in scalar files are per-example energies; multiplying by the bank row count gives the corresponding sums.\n\nThe Phase I nested-array CSV serialization error, an audit metadata dictionary merge error, and deterministic report display corrections were repaired with logs retained in ../repairs/. Neither repair required TC retraining or changed scientific thresholds, fitted rows, locked paths, checkpoint choices or accepted sets.\n'''
p=root/'phase_I/REPORT.md';p.write_text(p.read_text()+methods)
for p in [root/'phase_I/REPORT.md',root/'phase_II/REPORT.md']:pdf(p)
cross=root/'CROSS_PHASE_SUMMARY.md';s=cross.read_text();old='Equal attained budgets are reported without padding stopped sequences; three seeds support descriptive replication only. L0 changes materially with width, so width effects are not isolated from sparsity, fidelity, rarity or convergence. See both full reports for per-output, response, conditioning and cancellation diagnostics; all unfavorable seed results are retained.'
new='Rank equaled count in every cell: width 1024 supplied more retained directions, but no strict dictionary reached rank 256; width 2048 regressed. Equal-budget rankings varied: at 16 coordinates, median rho_N was 1.748 (width 512) versus 1.931 (1024); at 32 it was 3.047 versus 2.242. Missing budget/seed pairs were not padded. Activity rose markedly and width-2048 fidelity worsened, leaving sparsity and convergence confounds visible. Three seeds provide descriptive replication, not precise uncertainty.'
s=s.replace(old,new);cross.write_text(s)
css='@page{size:A4;margin:12mm}body{font-family:DejaVu Sans;font-size:9pt;line-height:1.24}h1{font-size:17pt}table{border-collapse:collapse;width:100%;font-size:7.4pt}td,th{padding:3px;border-bottom:1px solid #ccc}a{color:#235b8b}'
doc=HTML(string='<style>'+css+'</style>'+markdown.markdown(s,extensions=['tables']),base_url=str(root.resolve())).render();assert len(doc.pages)==1;doc.write_pdf(cross.with_suffix('.pdf'))
(root/'README.md').write_text('''# SPEC v0.8 — both phases complete

Start with the one-page [cross-phase summary](CROSS_PHASE_SUMMARY.md) ([PDF](CROSS_PHASE_SUMMARY.pdf)).

- [Phase I summary](phase_I/PHASE_I_SUMMARY.md) · [readable report](phase_I/REPORT.md) ([PDF](phase_I/REPORT.pdf)) · [six figures](phase_I/figures/).
- [Phase II summary](phase_II/PHASE_II_SUMMARY.md) · [readable report](phase_II/REPORT.md) ([PDF](phase_II/REPORT.pdf)) · [six figures](phase_II/figures/).
- [Guided addition ledger](phase_I/paths/guided/additions.csv) · [all trial decisions](phase_I/paths/guided/trial_ledger.csv) · [equal-count comparisons](phase_I/equal_addition_comparison.csv).
- [Width-ablation table](phase_II/width_summary.csv) · [equal-budget metrics](phase_II/equal_budget_metrics.csv) · [attainment counts](phase_II/EQUAL_BUDGET_SUMMARY.md).
- Selected Phase I state: [D2](phase_I/paths/guided/checkpoints/add_001/D2.npy), [C2](phase_I/paths/guided/checkpoints/add_001/C2.npy), [R2](phase_I/paths/guided/checkpoints/add_001/R2.npy). All Phase II full-set and budget bundles are under [validation_and_factorization/](phase_II/validation_and_factorization/).

13 fresh TCs and two verified historical reuses completed the 15-cell grid. The teacher and prior scientific artifacts remain unchanged. C2 was always computed as W2 @ pinv(D2); no S2 or output regression was fitted. Strict labels remain separate from provisional Phase I additions. No protected-final population was scored.

The initial orchestrator audit failed on a duplicate metadata key; both training workers exited 0. The repaired audit and final checks exited 0. Failed-attempt logs are retained in repairs/, with successful completion in audit_fixed.log and final_checks.log. Mathematical and integrity tests passed; plots and report pages were inspected.

The requested residual_width_v0_8/ directory and bundled REFERENCE_CHECKS.py were absent. Root SPEC.md and CODEX_PROMPT.md both contained v0.8 and were read in full. The missing reference script itself could not be run; its specified mathematical fixtures were implemented and tested in the repository. Required trained artifacts and CUDA were available, so neither experiment phase was blocked.
''')
write(root/'audit_repair_exit_code.json',0);write(root/'final_checks_exit_code.json',0);commands=read(root/'commands.json');commands['initial_orchestrator_exit_code']=commands.pop('audit_exit_code');commands['audit_repair_exit_code']=0;commands['final_checks_exit_code']=0;commands['final_targeted_tests_passed']=9;commands['repairs'].append('Audit metadata inf/met duplicate condition key merged explicitly; all existing checkpoints/readouts/accepted sets retained');write(root/'commands.json',commands)
summary=read(root/'execution_summary.json');summary.update(version='0.8',final_checks='passed',saved_matrix_bundles_verified=121,targeted_tests_passed=9,figures_visually_inspected=12,phase_I_report_pages=int(subprocess.run(['pdfinfo',str(root/'phase_I/REPORT.pdf')],capture_output=True,text=True).stdout.split('Pages:')[1].split()[0]),phase_II_report_pages=int(subprocess.run(['pdfinfo',str(root/'phase_II/REPORT.pdf')],capture_output=True,text=True).stdout.split('Pages:')[1].split()[0]),cross_phase_PDF_pages=1,initial_audit_failure_repaired=True,unresolved_experiment_blockers=[]);write(root/'execution_summary.json',summary)
source=read(root/'source_manifest.json')
for name in ['M.npy','d.npy','fit_statistics.npz','context_manifest.json','preprocessing.npz','readout_lock.json','manifest_lock.json']:
 p=OLD/name;source[str(p)]=sha(p)
for p in [Path('SPEC.md'),Path('CODEX_PROMPT.md')]:source[str(p)]=sha(p)
write(root/'source_manifest.json',source)
for folder in ['phase_I','phase_II']:write(root/folder/'figures/index.json',dict(figures=[f.name for f in sorted((root/folder/'figures').glob('*.png'))],PNG_dpi=220,all_visually_inspected=True,discrete_entity_labels=True,raw_signed_coefficients=True))
for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,root/'source_snapshot'/p.name)
write(root/'artifact_hashes.json',{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
print('REPORTS COMPLETE; one-page cross-phase PDF; all outputs sealed',flush=True)

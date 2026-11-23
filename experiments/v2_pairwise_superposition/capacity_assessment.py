"""Research interpretation and final audit, without additional training or inference."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .capacity import BASE,ROOT,OUT,RUN,CK,NEW,OLD,load_frozen_data
from .run import digest
from src.runtime import write_json,load_checkpoint

ASSESSMENT='''**The width-128 teacher is a better original-loss predictor, but it does not meet the unchanged gate for starting the intended all-primitive transcoder experiment.** Validation exact-vector accuracy increases 90.70%→94.50% (+3.80 percentage points), training exact-vector accuracy 91.15%→95.02%, and validation full-output MSE falls 0.0126542→0.0097205 (23.18% reduction). Validation semantic MSE falls 0.0168719→0.0129601. Width128 is preferable to width64 under the preserved validation-MSE selection metric, but that preference is not scientific acceptance of the teacher.

All five previously weak ANDs improve in both training and validation. Validation recall changes: AND(0,1) 88.04%→97.61%; AND(0,2) 78.00%→90.50%; AND(3,4) 75.62%→81.59%; AND(5,6) 87.11%→91.24%; AND(9,10) 71.21%→82.32%. Their validation FPRs stay low (0–0.143%), unlike the balanced-loss trial. AND(7,8) remains perfectly classified, although its continuous semantic MSE and interaction approximation worsen. XOR recall improves 97.30%→98.74% and FPR falls 1.694%→1.271%. Inhibition regresses: recall 99.832%→99.496% and FPR 0→0.249%, although it still passes the 99% balanced-accuracy gate. The secondary DNF improves: balanced accuracy 95.39%→96.83%, recall 91.55%→94.04%, FPR 0.770%→0.385%; it is not counted as primitive recovery.

**Only 2/8 primitives meet at least 99% validation balanced accuracy:** AND(7,8) (100%) and inhibition (99.62%). AND(0,1) (98.73%) and XOR (98.73%) are close but do not pass. AND(0,2), AND(3,4), AND(5,6), AND(9,10) achieve 95.20%, 90.78%, 95.62%, 91.16%. The criterion has not been rounded up, averaged across outputs, or relaxed. Validation FN counts remain 5, 19, 37, 17, 0, 35, 14, 3 across g0..g7. In particular, training recall is still only 82.78% for AND(3,4), 85.01% for AND(9,10), and 89.65% for AND(0,2); remaining errors are not explained solely by unseen-example generalization.

**Geometry is a contributing association, not a complete explanation.** Before retraining, baseline validation XOR FNs had median opposite-label training distance 0.8149, versus 0.9830 for correctly recalled positives; AND(5,6) was 0.8352 versus 0.9979. AND(3,4) had a smaller gap (0.7750 versus 0.7996). AND(0,1) medians were equal (0.8256), and AND(9,10) FNs were not closer by median absolute distance (0.7779 versus 0.7750), although opposite/same distance ratios differ. Training associations are also reported. Thus some difficult examples occupy more locally conflicting regions, but nearest opposite-label distance alone cannot account for all weak ANDs. Ratios are affected by class prevalence and nearest-neighbor sampling. These finite train/validation comparisons neither establish irreducible input collisions nor prove a causal interference mechanism. Improvement with unchanged P/Q and data shows that the original results were not fixed by embedding geometry alone.

**Most studied interactions become more distributed, not dominated by one neuron.** Across the eligible fixed contexts, effective unit counts increase for AND(0,1) 4.69→6.49, AND(0,2) 1.70→8.37, AND(3,4) 3.31→5.36, AND(5,6) 4.16→6.06, AND(9,10) 3.39→5.50, XOR 4.79→11.44, and inhibition 2.14→8.18. These effective counts are descriptive and much smaller than the available width, not a literal count of necessary neurons. AND(7,8) is an exception: effective units 1.71→1.52 and top-one absolute share 70.45%→78.72%, but that unit alone is not an exact explanation. Its average positive sum is +1.3465 and negative sum −0.3633, with net +0.9832 and top-one signed residual MAE 0.3489. XOR has +1.9328 positive and −3.6551 negative contributions (net −1.7223), 64.65% cancellation and only 17.57% top-one absolute share. Inhibition has +0.6984 and −1.5003 (net −0.8019). Signed cancellation remains consequential; neither a large absolute contribution nor good classification identifies a standalone Boolean neuron.

Interaction fidelity improves unevenly. Pooled primitive target-interaction MAE improves for g0, g2, g3, g5 and XOR; it worsens for g1, AND(7,8), and inhibition. XOR MAE improves 0.4230→0.3405 and AND(0,1) 0.3465→0.2550; AND(7,8) worsens 0.01489→0.06568 despite unchanged perfect ordinary accuracy. These summaries include OOD corners; the per-regime table reports the much smaller subset of all-corners-density-supported contexts separately. There are only 3–5 such contexts per primitive, so they are diagnostic comparisons, not robust population estimates. Full signed values for every context/output/neuron and residual checks are retained.

Matched distractor flips remain nonzero. Within density support, the observed classification-flip fraction decreases 0.813%→0%, but mean absolute semantic-score change slightly increases 0.03181→0.03405 and the maximum remains 0.31055. This supports improved threshold robustness on this small historical probe bank, not complete distractor invariance. Counts are repeated rule-context memberships and are not independent samples. The ordinary distractor-group tables and full matched rows allow inspection beyond these averages.

**Interpretation:** increasing width helps under the exact original loss and training policy, consistent with a capacity-related and/or width-dependent optimization limitation in the first teacher. One seed with different parameter dimensions cannot cleanly separate representational capacity from optimization dynamics or initialization effects. Persistent training errors, local label conflicts and remaining nuisance dependence leave objective allocation, optimization and feature interference as plausible contributors. The new model reaches the 2,000-epoch budget with small continuing validation improvement; that is not evidence that more epochs or another width would solve the remaining rules. No further variant was launched.

**Decision:** do not start the transcoder under the stated all-primitive acceptance criterion. Preserve this improved teacher as the capacity-trial candidate and review the remaining AND failures before any new scientific decision. The original test remains historical only; the independently reserved final population was not opened or scored. No independent final-generalization claim or causal mechanism claim is made.'''


def finalize():
    cp=load_checkpoint(CK/'last.pt');initial=load_checkpoint(CK/'initial.pt')
    baseline=load_checkpoint(BASE/'checkpoints'/OLD/'last.pt')
    generator=torch.Generator().manual_seed(3105)
    for _ in range(cp['epoch']):torch.randperm(14000,generator=generator)
    assert torch.equal(generator.get_state(),cp['shuffle_rng'])
    if cp['epoch']==baseline['epoch']:assert torch.equal(cp['shuffle_rng'],baseline['shuffle_rng'])
    assert all(a['learning_rate']==b['learning_rate'] for a,b in zip(cp['history'],baseline['history']))
    assert cp['config']['training']==baseline['config']['training']
    torch.manual_seed(3105)
    from src.models import OriginalMLP
    reconstructed=OriginalMLP(12,128,12)
    assert all(torch.equal(v.cpu(),reconstructed.state_dict()[k]) for k,v in initial['model'].items())
    protected=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(p))==h for p,h in protected.items())
    report=OUT/'report.md';s=report.read_text();assert 'ASSESSMENT_PENDING' in s
    s=s.replace('ASSESSMENT_PENDING',ASSESSMENT)
    s+='''
## Final verification and executed commands

All 19 V2 tests passed before training (`19 passed in 2.89s`, full log in tests_final_pretraining.log). Tests cover the width-only config override, CUDA gradients/checkpoint roundtrip, exact pair/triple signed sums at width128, label-specific nearest-neighbor distances and self-exclusion, and contribution cancellation. The initial test invocation failed because an earlier test initialized CUDA before deterministic CuBLAS configuration was set. The suite passed after setting `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process start; deterministic checks were not disabled. Both attempt logs are retained. The actual experiment includes the same environment setting and exited 0 without training/postprocessing retries.

Executed in named tmux session `v2-capacity128`:

```bash
bash experiments/v2_pairwise_superposition/execution_logs/capacity128-20261027/execute.sh
```

The script sets `CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and `PYTHONUNBUFFERED=1`, then runs `.venv/bin/python -m experiments.v2_pairwise_superposition.capacity`, piping stdout/stderr to its own execution.log and recording the Python exit status. No session remains after successful completion. Test command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest experiments/v2_pairwise_superposition/tests -q
```

Report-only scientific interpretation/final checks:

```bash
.venv/bin/python -m experiments.v2_pairwise_superposition.capacity_assessment
```

Final control checks confirm the new saved initialization matches deterministic reconstruction at seed 3105, all 2,000 learning-rate entries equal the baseline, and final CPU shuffle-generator state equals both deterministic reconstruction and the original baseline. Optimizer/training settings match exactly. The new width has 3,212 parameters versus 1,612 for width64. All 143 previously protected files verified unchanged, including prior V1/V2 checkpoints and earlier V2 logs/reports, original dataset/manifest, SPEC and prompt. The reserved final data file was not accessed. Final code/test snapshots and report hashes are recorded in the run directory. No packages were installed.
'''
    # Test duration is taken from actual recorded output rather than inferred.
    testlog=(BASE/'execution_logs'/NEW/'tests_final_pretraining.log').read_text().strip().splitlines()[-1]
    s=s.replace('19 passed in 2.89s',testlog)
    report.write_text(s)
    for name in ('capacity_assessment.py','capacity_README.md'):
        shutil.copyfile(BASE/name,RUN/(name+'.snapshot'))
    for p in (BASE/'tests').glob('*.py'):shutil.copyfile(p,RUN/'source_snapshot'/('tests__'+p.name))
    write_json(RUN/'final_verification.json',{'report_sha256':digest(report),'assessment_sha256':digest(BASE/'capacity_assessment.py'),'tests':testlog,'protected_files_verified':len(protected),'initialization_reconstruction_matches':True,'shuffle_state_matches_baseline':True,'all_epoch_LRs_match_baseline':True,'old_test_scored':False,'reserved_final_accessed':False,'experiment_exit_status':int((BASE/'execution_logs'/NEW/'exit_status').read_text()),'additional_training_launched':False})
    print(report)

if __name__=='__main__':finalize()

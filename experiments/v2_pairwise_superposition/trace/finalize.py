"""Read and explain the saved eight-input trace; no additional model evaluation."""
import json,shutil
from pathlib import Path
import numpy as np
from src.runtime import load_checkpoint,write_json
from ..run import BASE,ROOT,digest
from .run import DIRECTORY
from .report import produce

INTERPRETATION='''**Exact algebra — how the teacher produces all four values.** In both backgrounds, the three displayed original neurons are inactive at00,01,10 and turn on at11. Their single-input coefficients and baseline are therefore zero; they supply signed joint terms. In B0, the11 response from them is `+1.040799 − 0.457968 + 0.378106 = 0.960937`. Neuron192 subtracts from the output sum at11; here it calibrates the joint amplitude rather than suppressing a single-input response. It does not suppress another hidden neuron. Their combined11 contribution is0.956933 in B1.

**The original remainder is small in joint interaction, but essential to the complete function.** In B0 it contributes the polynomial `[0.0909705, −0.00180382, −0.000479581, 0.0115101]`, while the actual output bias contributes`[−0.0867383,0,0,0]`. Their baseline cancellation leaves the observed s00=0.00423226. Together with the trio they produce`[0.00423226, −0.00180382, −0.000479581, 0.972447]` for the complete teacher polynomial. Thus a7-only/a8-only outputs, baseline cancellation and a small additional joint term lie outside the displayed trio. In B1 the remainder is`[0.0858015,0.00442928,0.00286153,0.00831141]` before the same output bias; omitting it would give the wrong baseline, both false-corner values and the11 value. No offset has been fitted to conceal this.

**Additive components really matter.** The91 always-on native units in B0 supply baseline0.0943279, a7 response−0.00713535, and a8 response+0.0111573 with zero joint coefficient. Other remainder components counter these single-input effects, leaving small total false-corner responses. The always-on set is defined separately in each background; its changing membership is not a fixed neuron group. A rank based only on the joint coefficient would miss these additive contributions. This is why the complete corner polynomial and explicit remainder are needed even when a compact group explains most of the joint response.

**Observed transcoder description — similar local gates, incomplete selected pair.** Features173 and266 are also11-only here, with positive and negative g4 writes, but no feature-to-neuron identity is inferred. In B0 their signed sum is only`0.498262 − 0.133236 = 0.365027`. The remaining1022 features supply a joint coefficient0.597654, compared with the complete transcoder joint coefficient0.962680: most of that joint computation is outside the displayed pair. Their remainder also supplies the baseline and single-input responses. The same pattern holds in B1. These features suggest a similar local threshold-and-opposing-write motif, while the full transcoder is an incomplete approximation to the teacher, not an exactly equivalent construction on all eight inputs. Its maximum absolute g4 residual here is0.0102892, at B1/10.

**Observed nuisance change — small net effect, different compensation.** Adding a15 shifts every displayed unit's q by exactly its a15 input coefficient, but none leaves its11-only interval. At11, native224/192/30 change by`+0.0119418, −0.0199434, +0.00399714`: the stronger negative write offsets the positive increases. Their sum changes−0.00400446, and the remainder adds+0.00120647, so the full teacher changes only−0.00279799. All teacher corner changes are small (maximum absolute change0.00516904). No replacement background was sought. Gates do switch among other units, but switching counts alone do not measure g4 impact.

The transcoder pair similarly changes−0.00339846 at11, but its remainder adds+0.00929904, so the full transcoder changes **+0.00590058**, opposite to the teacher. Its near-exact B1/11 match (residual−8.85e-6) therefore does not show that it reproduces the teacher's nuisance response. The full polynomials also reveal different single-input effects, particularly in B1. Matching an11 value or a joint coefficient alone would hide these distinctions.

**Hypotheses and one discriminating observation, not a new experiment.** All five displayed components are conjunction-like gates in these two specific backgrounds. Two backgrounds cannot establish globally invariant AND detectors, and the output-write vectors alone establish neither semantic purity nor causal effects. A further observation that would distinguish local resemblance from more general equivalence is a separately declared development background whose computed q lies on different sides of a teacher/transcoder11-only boundary; check the predicted gate changes and the *full* four-corner/remainder accounting there. Select such a comparison from coefficients and permitted membership metadata before viewing outputs. No third background, hidden-neuron intervention, fitting, correspondence model or further case study was executed here.'''


def main():
    root=DIRECTORY
    if (root/'report.finalized.json').exists():raise RuntimeError('Final report already exists; refusing overwrite')
    with np.load(root/'native_trace.npz') as f:native={k:f[k] for k in f.files}
    with np.load(root/'transcoder_trace.npz') as f:tc={k:f[k] for k in f.files}
    shutil.copyfile(root/'report.md',root/'report.initial.md')
    produce(root,native,tc)
    p=root/'report.md';s=p.read_text();assert 'INTERPRETATION_PENDING' in s;s=s.replace('INTERPRETATION_PENDING',INTERPRETATION)
    s=s.replace('These eight inputs were already training examples and Stage 3 discovery inputs, not independent evaluation data.','These eight inputs were already training examples and Stage 3 discovery inputs, not independent evaluation data. Values below use float64 arithmetic on the saved float32 weights; direct float32 outputs are retained in the exports.')
    tests=(root/'tests.log').read_text().strip().splitlines()[-1]
    before=json.loads((root/'protected_before.json').read_text());assert all(digest(Path(path))==h for path,h in before.items())
    result=json.loads((root/'complete.json').read_text());provenance=json.loads((root/'provenance.json').read_text());frozen=json.loads((root/'selection_frozen.json').read_text());assert frozen['selection_sha256']==digest(root/'selection.json')
    assert int((root/'exit_status').read_text())==0
    config_records={}
    for label in ('teacher','transcoder'):
        path=Path(provenance[label]);assert digest(path)==provenance[label+'_sha256'];checkpoint=load_checkpoint(path);config_records[label]={'configuration':checkpoint['config'],'selected_epoch':checkpoint['epoch'],'sha256':digest(path)}
    write_json(root/'frozen_configurations.json',config_records)
    helpers=[BASE/'stage2/core.py',BASE/'data.py',BASE/'run.py']
    for helper in helpers:shutil.copyfile(helper,root/'source_snapshot'/str(helper.relative_to(ROOT)).replace('/','__'))
    shutil.copyfile(Path(__file__),root/'source_snapshot/trace_finalize.py')
    s+='\nExecuted status: **exit0; '+tests+'**. Tests were targeted to this trace, not the earlier intervention suite. CUDA hardware and exact checkpoint hashes are recorded in provenance.json. All '+str(len(before))+' protected prior artifacts retain their original SHA-256 values. The metadata-frozen pair remained unchanged. No packages were installed. The analysis was small enough to run directly without tmux or background jobs.\n'
    s+='\nReport-only finalization command: `.venv/bin/python -m experiments.v2_pairwise_superposition.trace.finalize`. Additional model/helper source snapshots and both frozen configurations are archived. The only reservation material read was its existing exclusion manifest; the reserved population file was not accessed.\n'
    p.write_text(s)
    result['report_sha256']=digest(p);result['tests']=tests;result['helper_source_hashes']={str(path):digest(path) for path in helpers};result['final_report_sources']={str(path):digest(path) for path in (Path(__file__),Path(__file__).parent/'report.py')};write_json(root/'complete.json',result)
    shutil.copyfile(Path(__file__).parent/'report.py',root/'source_snapshot/trace_report_final.py')
    write_json(root/'report.finalized.json',{'report_sha256':digest(p),'tests':tests,'finalize_source_sha256':digest(__file__),'model_outputs_recomputed':False,'hidden_interventions':False});print(p)

if __name__=='__main__':main()

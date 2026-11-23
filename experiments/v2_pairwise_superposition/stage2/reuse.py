"""Offline first differences distinguish input sharing from pairwise interaction sharing."""
import json,shutil
from pathlib import Path
import numpy as np
from src.runtime import write_json
from ..data import keys
from ..run import BASE,digest,write_csv
from .run import C,ID,RUN,OUT
from .core import name


def main():
    summary=[];candidates=[];maxres=0
    for width in C['widths']:
        for coefficient in C['lambdas']:
            key=name(width,coefficient)
            with np.load(OUT/key/'coordinates.npz') as f:coord={k:f[k] for k in f.files}
            with np.load(OUT/key/'signed_interactions.npz') as f:
                k0=keys(f['p0_corners'][:,0,:].astype(np.uint8));k1=keys(f['p1_corners'][:,0,:].astype(np.uint8));d0={int(k):i for i,k in enumerate(k0)};d1={int(k):i for i,k in enumerate(k1)};common=sorted(set(d0)&set(d1));i0=[d0[k] for k in common];i1=[d1[k] for k in common]
                z0,z1=f['p0_z'][i0],f['p1_z'][i1];s0,s1=f['p0_student_corner_scores'][i0],f['p1_student_corner_scores'][i1]
                effects={}
                for label,z,j,s in [('g0',z0,0,s0),('g1',z1,1,s1)]:
                    for condition,hi,lo in [('partner_off',2,0),('partner_on',3,1)]:
                        effect=(z[:,hi,:]-z[:,lo,:])*coord['A'][j,:]
                        direct=s[:,hi,j]-s[:,lo,j];np.testing.assert_allclose(effect.sum(1),direct,atol=1e-9,rtol=1e-8);maxres=max(maxres,float(np.abs(effect.sum(1)-direct).max()));effects[label+'_'+condition]=effect
                v0=np.abs(effects['g0_partner_on']).mean(0);v1=np.abs(effects['g1_partner_on']).mean(0);profile=float(v0@v1/(np.linalg.norm(v0)*np.linalg.norm(v1)));ranking=np.argsort(-np.sqrt(v0*v1))
                top=ranking[0];summary.append({'config':key,'n_deduplicated_matched_bases':len(common),'first_a0_effect_usage_cosine_partner_on':profile,'top_shared_ranked_unit':int(top),'top_mean_abs_g0_partner_on':float(v0[top]),'top_mean_abs_g1_partner_on':float(v1[top]),'top_mean_signed_g0_partner_on':float(effects['g0_partner_on'][:,top].mean()),'top_mean_signed_g1_partner_on':float(effects['g1_partner_on'][:,top].mean())})
                for u in ranking[:10]:
                    candidates.append({'config':key,'unit':int(u),'matched_base_keys':common,'complete_K':coord['K'][u].tolist(),'complete_A':coord['A'][:,u].tolist(),'bias':float(coord['encoder_bias'][u]),'mean_signed_first_effects':{label:float(value[:,u].mean()) for label,value in effects.items()},'mean_absolute_first_effects':{label:float(np.abs(value[:,u]).mean()) for label,value in effects.items()},'complete_context_first_effects':{label:value[:,u].tolist() for label,value in effects.items()},'g0_interaction_from_on_minus_off':float((effects['g0_partner_on'][:,u]-effects['g0_partner_off'][:,u]).mean()),'g1_interaction_from_on_minus_off':float((effects['g1_partner_on'][:,u]-effects['g1_partner_off'][:,u]).mean()),'interpretation':'Shared first-input response may be additive or compensatory; compare partner-on/off differences and output signs before proposing a reusable conjunction component.'})
    write_csv(OUT/'shared_input_first_differences.csv',summary);write_json(OUT/'shared_input_first_difference_candidates.json',candidates)
    report=OUT/'report.md';s=report.read_text();s=s.replace('and a11-only activation gate','and an 11-only activation gate')
    s=s.replace('Normalized counts are lower for the transcoders because their dictionaries are larger','For the fidelity winner and low-L0 Pareto endpoint, normalized counts are lower because their dictionaries are larger')
    s=s.replace('The shared-input analysis does not reveal a clean reusablea0 computation for both conjunctions.','The shared pairwise-interaction analysis does not reveal a clean reusable nonlinear a0 component for both conjunctions. Additive a0 responses have zero pairwise differences, so this test alone does not rule out input reuse.')
    lines=['','## Supplement: shared first-input responses versus nonlinear reuse','','A low pair-interaction usage cosine does not rule out shared additive input responses. Using the already saved feature activations, we computed each unit’s signed first difference for toggling a0 with its partner off/on: Δ0z×A_g0 for(a0,a1), and Δ0z×A_g1 for(a0,a2). The12 base contexts are identical and deduplicated across both pairs. Every full signed first-difference sum was checked against the saved direct semantic output difference; no model was fitted or newly inferred. The maximum identity residual was '+f'{maxres:.3g}'+'.','','| Config | First-a0-response usage cosine (partner on) | Top shared-ranked unit | Mean signed effect g0 | Mean signed effect g1 |','|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['config']} | {r['first_a0_effect_usage_cosine_partner_on']:.6f} | {r['top_shared_ranked_unit']} | {r['top_mean_signed_g0_partner_on']:.6f} | {r['top_mean_signed_g1_partner_on']:.6f} |")
    lines+=['','First-response overlap is larger than pairwise-interaction overlap. In the fidelity winner, feature922 contributes about−0.02928 to the a0 effect on g0 both with a1 off and on (zero mean pair interaction), while its g1 first effect changes from+0.20768 with a2 off to+0.16998 with a2 on. This is shared but partly additive/oppositely signed response, not a common helpful conjunction detector. In the low-L0 endpoint, feature404 contributes positive first effects to both outputs (about+0.06302/+0.06442 with partners on), with small negative pair interactions (−0.00550/−0.00202). Conversely, intermediate Pareto model512/1e-4 feature181 contributes positive nonlinear pair terms to both relationships (+0.03633/+0.01382). There is evidence of limited shared input-response components, but not a dominant reusable nonlinear a0 circuit from these bank measurements alone.','', 'Full K/A/bias and all context-specific partner-on/off effects for the top10 shared-response candidates per model are in shared_input_first_difference_candidates.json. Shared first responses can be linear, compensatory, or oppositely signed; they are not automatically a common helpful AND circuit. Input representation reuse and shared nonlinear interaction implementation must remain separate claims. This supplementary analysis is diagnostic on the same historical bank and does not change the validation-selected model.','', 'This final report-only supplement was executed with `.venv/bin/python -m experiments.v2_pairwise_superposition.stage2.reuse`. It also clarifies the 11-only Boolean-corner wording and scopes the normalized-concentration claim to the named Pareto endpoints. No training/evaluation source or checkpoint was modified.']
    report.write_text(s+'\n'.join(lines)+'\n')
    protected=json.loads((RUN/'protected_before.json').read_text());assert all(digest(Path(p))==h for p,h in protected.items())
    shutil.copyfile(Path(__file__),RUN/'source_snapshot/stage2_reuse.final.py')
    result=json.loads((RUN/'complete.json').read_text());result['report_sha256']=digest(report);result['shared_first_difference_max_identity_residual']=maxres;result['reuse_source_sha256']=digest(__file__);write_json(RUN/'complete.json',result)
    print(report)
    print(summary)

if __name__=='__main__':main()

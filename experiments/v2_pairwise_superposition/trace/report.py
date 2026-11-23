"""Human-readable complete eight-input accounting; no component selection or fitting."""
import json
import numpy as np
from ..run import write_csv
from .core import polynomial


def fmt(x):return f'{float(x):.12g}'
def vector(v):return '['+', '.join(fmt(x) for x in v)+']'


def accounting(result,b):
    rows=[]
    for unit in result['displayed']:
        rows.append((f'Unit {int(unit)}',result['semantic_contributions'][b,:,4,unit],result['g4_alpha'][b,unit]))
    rows += [('All remaining units',result['semantic_remainder'][b,:,4],result['g4_remainder_alpha'][b]),('Semantic output bias',np.full(4,result['semantic_bias'][4]),result['g4_bias_alpha'][b]),('Complete output',result['semantic_total'][b,:,4],result['g4_total_alpha'][b])]
    return rows


def produce(root,native,tc):
    selection=json.loads((root/'selection.json').read_text());checks=json.loads((root/'checks.json').read_text());backgrounds=selection['background_active_coordinates'];flip=selection['flipped_input'];corners=['00','01','10','11']
    lines=['# AND(a7, a8): a complete two-background computational trace','',f"**B0:** active background coordinates {backgrounds[0]}. **B1:** {backgrounds[1]}; only a{flip} changes 0→1. All nine Boolean targets are unchanged between corresponding corners. These eight inputs were already training examples and Stage 3 discovery inputs, not independent evaluation data.",'','| Background | a7a8 | Active count | Full teacher g4 | Full transcoder g4 | TC − teacher |','|---|---|---:|---:|---:|---:|']
    for b in range(2):
        for c,corner in enumerate(corners):
            y=native['semantic_total'][b,c,4];z=tc['semantic_total'][b,c,4]
            lines.append(f"| B{b} | {corner} | {selection['corner_densities'][b][c]} | {fmt(y)} | {fmt(z)} | {fmt(z-y)} |")
    lines+=['','## Selection and exact algebra','','The first qualifying discovery pair was selected by ascending B0 integer key, then ascending added nuisance coordinate, using input/membership metadata only. Conditions: three active background bits, B1 in the same saved bank, no historical-test corners, and every corner documented as excluded from the reserved population. There were '+str(selection['qualified_pair_count_in_partition'])+' qualifying pairs; this one was frozen before any checkpoint outputs were read. Its nuisance effect was not used to choose it.',f"\nStage 3 discovery indices: {selection['background_development_indices']}; background keys {selection['background_keys']}. Original corner memberships: {selection['original_membership']}. Input keys and exact memberships are in selection.json. The original-data exclusion rule in the reservation manifest certifies all eight as excluded; the reserved dataset itself was never opened. No new population was generated.",'','Column-vector convention: `x=P a`, `C=W1 P`, `D=Qᵀ W2`, `beta=Qᵀ b2`; teacher `s=D ReLU(Ca+b1)+beta`. Transcoder `K=EP`, `A=QᵀB`, `gamma=Qᵀc`; `s_TC=A ReLU(Ka+e)+gamma`. Row batches implement `x=a@P.T`, hidden inputs `x@W1.T+b1`, semantic outputs `y@Q`. D is the **known effective readout**, not a newly discovered intrinsic feature decoder. All feature indices are zero-based; g4 is AND(a7,a8).','','For each unit the background offset is `q=b+sum_l coefficient[l]*a_l` excluding a7/a8. Preactivations in corner order00,01,10,11 are `[q, q+v, q+u, q+u+v]`. Gates are strictly `p>0`; ReLU and the actual signed output weight are retained. No neurons/features were altered, ablated or matched.','','## Detailed original-neuron trace']
    for label,result in [('Original neuron',native),('Transcoder feature',tc)]:
        if label=='Transcoder feature':lines+=['','## Detailed transcoder-feature trace','','Features173/266 are not paired with any particular original neuron. They are two components of a1024-feature sum.']
        for unit in result['displayed']:
            unit=int(unit);u=result['u'][unit];v=result['v'][unit];weight=result['semantic_readout'][4,unit];bias=result['hidden_bias'][unit]
            lines += ['',f'### {label} {unit}',f"\n`u={fmt(u)}`, `v={fmt(v)}`, original/encoder bias `{fmt(bias)}`, signed g4 write `{fmt(weight)}`.",f"\nComplete semantic output-write vector (g0..g8): `{vector(result['semantic_readout'][:,unit])}`. This is a readout observation, not a claim of polysemanticity or a causal effect on those outputs.",'','| Background | Each active-input term in q | Total q (including hidden bias) | Preactivations [00,01,10,11] | ReLU activations | Gate pattern | Signed g4 contributions |','|---|---|---:|---|---|---|---|']
            for b in range(2):
                terms='; '.join(f"a{i}: {fmt(result['background_terms'][b,unit,i])}" for i in backgrounds[b]);gate=''.join(str(int(g)) for g in result['gate'][b,:,unit])
                lines.append(f"| B{b} | {terms} | {fmt(result['q'][b,unit])} | {vector(result['preactivation'][b,:,unit])} | {vector(result['activation'][b,:,unit])} | `{gate}` | {vector(result['semantic_contributions'][b,:,4,unit])} |")
            if u>0 and v>0:
                lines.append(f"\n11-only interval: **{fmt(-(u+v))} < q ≤ {fmt(-max(u,v))}**. B0: {'inside' if result['positive_11_only_band'][0,unit] else 'outside'}; B1: {'inside' if result['positive_11_only_band'][1,unit] else 'outside'}. This characterizes these two backgrounds, not a global detector.")
    lines+=['','## Complete corner accounting, including every remainder','','The selected IDs are starting points. Remainders sum all other253 original neurons or1022 transcoder features with their signs; no top-k truncation or fitted offset is used. The semantic output bias is the actual projected network bias.']
    exports=[]
    for label,result in [('Teacher',native),('Transcoder',tc)]:
        for b in range(2):
            lines += ['',f'### {label}, B{b}','','| Contribution | 00 | 01 | 10 | 11 |','|---|---:|---:|---:|---:|']
            for name,values,alpha in accounting(result,b):
                lines.append('| '+name+' | '+' | '.join(fmt(x) for x in values)+' |');exports.append({'model':label,'background':f'B{b}','component':name,**dict(zip(corners,map(float,values))),**dict(zip(['alpha0','alpha7','alpha8','alpha78'],map(float,alpha)))})
    write_csv(root/'corner_and_polynomial_accounting.csv',exports)
    lines+=['','## Complete Boolean-corner polynomial','','Each row is exactly `alpha0 + alpha7*a7 + alpha8*a8 + alpha78*a7*a8` on the four Boolean corners. These are finite algebraic coefficients, not a claim of bilinear behavior between corners. The ideal AND vector is[0,0,0,1]; our primary comparison remains the actual teacher against the transcoder.','','| Model / background / component | Baseline alpha0 | Single alpha7 | Single alpha8 | Joint alpha78 |','|---|---:|---:|---:|---:|']
    for row in exports:lines.append('| '+row['model']+' / '+row['background']+' / '+row['component']+' | '+' | '.join(fmt(row[k]) for k in ('alpha0','alpha7','alpha8','alpha78'))+' |')
    lines+=['','Always-on units are exactly additive over these corners; they can supply a baseline or single-input response even though their pair interaction is zero (up to roundoff). They are retained. Here their aggregate is:','','| Model / background | Always-on unit count | Baseline | Single a7 | Single a8 | Joint (roundoff only) |','|---|---:|---:|---:|---:|---:|']
    for label,result in [('Teacher',native),('Transcoder',tc)]:
        for b in range(2):
            always=result['gate'][b].all(0);values=result['semantic_contributions'][b,:,4,:][:,always].sum(-1);a=polynomial(values)
            lines.append(f'| {label} / B{b} | {int(always.sum())} | '+' | '.join(fmt(x) for x in a)+' |')
    lines+=['','### Mathematical unit-test example','','`h1=ReLU(a)`, `h2=ReLU(a-b)`, `s=h1-h2` has corner values h1=[0,0,1,1], −h2=[0,0,−1,0], sum=[0,0,0,1]. Their polynomial vectors are[0,1,0,0] and[0,−1,0,1]. The zero-interaction h1 is essential: it supplies the a-only response that the second component cancels at10, while preserving11. The targeted tests check this explicitly.','','## What the nuisance input changes','','For every one of the256/1024 units, `q(B1)-q(B0)=coefficient[a'+str(flip)+']`. The same preactivation shift applies at all four corners of a unit; ReLU may or may not change gate state. Hidden units do not feed into one another: changes are driven by the common input and their signed outputs add linearly. In particular a negative output write subtracts from the sum; it does not inhibit another hidden neuron.','','| Model / displayed unit | q change | Flipped-input coefficient | Gate changes | Change in alpha0 / alpha7 / alpha8 / alpha78 |','|---|---:|---:|---|---|']
    for label,result in [('Teacher',native),('Transcoder',tc)]:
        for unit in result['displayed']:
            changed=[corners[c] for c in range(4) if result['gate'][0,c,unit]!=result['gate'][1,c,unit]]
            lines.append(f"| {label} / {int(unit)} | {fmt(result['q'][1,unit]-result['q'][0,unit])} | {fmt(result['input_coefficients'][unit,flip])} | {changed or 'none; see magnitude change'} | {vector(result['g4_alpha'][1,unit]-result['g4_alpha'][0,unit])} |")
    lines+=['','### Full signed compensation for B1 − B0','','| Model / component | Delta00 | Delta01 | Delta10 | Delta11 |','|---|---:|---:|---:|---:|']
    gate_notes=[]
    for label,result in [('Teacher',native),('Transcoder',tc)]:
        b0=accounting(result,0);b1=accounting(result,1)
        for (name,v0,_),(_,v1,_) in zip(b0,b1):lines.append('| '+label+' / '+name+' | '+' | '.join(fmt(x) for x in v1-v0)+' |')
        changes=result['gate'][0]!=result['gate'][1]
        gate_notes.append('\n'+label+' gate switches among **all units**, by corner: '+', '.join(f'{corner}: {int(changes[c].sum())}' for c,corner in enumerate(corners))+'. Exact IDs and contribution changes are exported in '+('native' if label=='Teacher' else 'transcoder')+'_background_changes.csv.')
    lines+=gate_notes
    lines+=['','## Interpretation','','INTERPRETATION_PENDING','','## Precision, verification and files','','All component arrays retain float64 precision after casting the actual saved float32 weights and embeddings. Direct float32 execution on the same8 inputs is exported for comparison; this is not another population. Human tables round to12 significant digits; CSV/NPZ keep full unrounded float64 values. Sum checks, q-shift identities, polynomial reconstruction and 11-only intervals were checked at atol1e-10/rtol1e-10; the actual errors are in checks.json. No components are zeroed because their alpha78 is small.']
    for label,result in [('native',native),('transcoder',tc)]:
        check=checks['models'][label];lines.append(f"\n{label}: maximum full signed-sum/direct residual {check['sum_vs_direct_max_abs']:.3g}; displayed+remainder+bias residual {check['displayed_remainder_bias_max_abs']:.3g}; maximum float64/float32 semantic-output difference {check['float64_vs_saved_dtype_float32_max_abs']:.3g}.")
    lines+=['','- native_trace.npz / transcoder_trace.npz: complete coefficient matrices, all background terms, offsets, preactivations, gates, activations, all9 signed output contributions and polynomial terms.','- *_all_components.csv / *_all_coefficients.csv: every native neuron and transcoder feature, full input/output coefficients and each corner.','- *_background_changes.csv: every gate switch, offset identity and signed nuisance effect.','- corner_and_polynomial_accounting.csv / full_output_comparison.npz: displayed components, explicit remainders, true biases, full totals and TC residuals.','- selection.json / selection_frozen.json / selected_inputs.npz: output-independent choice and exact known input membership/exclusion evidence.','- provenance.json / checks.json / source_snapshot/: checkpoint/data/source hashes, CUDA/runtime settings and numerical checks.','- tests.log / execution.log / commands.sh: actual execution and targeted mathematical/CUDA tests.','','No training, fitted readout, neuron correspondence, hidden-neuron intervention, reserved-population scoring or XOR extension is part of this trace.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')

"""Fixed-group sufficiency and actual frozen-network activation interventions."""
import json
import numpy as np
import torch
from src.runtime import write_json
from ..run import write_csv
from ..capacity256 import signed_statistics
from .methods import group_metrics
from .run import C,RUN,OUT


def threshold_analysis(bank,native,tc,path):
    path.mkdir(exist_ok=False);records=[]
    selection=json.loads((RUN/'fixed_hypotheses.json').read_text())['selection'];tcids=selection['selected_tc_units']
    for label,coord,h,contrib,indices in [('native',native,bank['h'],bank['native_contributions'],range(256)),('transcoder',tc,bank['z'],bank['tc_contributions'],tcids)]:
        for unit in indices:
            v=h[:,:,unit];patterns=[''.join(map(str,r.astype(int))) for r in (v>0)];offset=bank['a'][:,0]@coord['C1'][unit]+coord['b1'][unit];weights=coord['C1'][unit]
            signal=contrib[:,4,unit]
            records.append({'model':label,'unit':int(unit),'C1_or_K_all20':weights.tolist(),'bias':float(coord['b1'][unit]),'all9_output_writes':coord['D2_eff'][:,unit].tolist(),'input7':float(weights[7]),'input8':float(weights[8]),'background_offset_quantiles':np.quantile(offset,[0,.01,.1,.5,.9,.99,1]).tolist(),'gate_patterns':{p:patterns.count(p) for p in sorted(set(patterns))},'11_only_gate_fraction':patterns.count('0001')/len(patterns),'mean_corner_activation':v.mean(0).tolist(),'mean_corner_signed_g4_contribution':(v*coord['D2_eff'][4,unit]).mean(0).tolist(),'mean_interaction':float(signal.mean()),'mean_abs_interaction':float(np.abs(signal).mean()),'positive_interaction_mean':float(np.maximum(signal,0).mean()),'negative_interaction_mean':float(np.minimum(signal,0).mean()),'interaction_sd':float(signal.std()),'offtarget_abs_input_mass':float(np.abs(np.delete(weights,[7,8])).sum())})
    write_json(path/'units.json',records)
    # Candidate gate deviations retained with exact input contexts, not only averages.
    deviations=[]
    for unit in [173,266]:
        h=bank['z'][:,:,unit];patterns=[''.join(map(str,r.astype(int))) for r in (h>0)]
        for i,pattern in enumerate(patterns):
            if pattern!='0001':deviations.append({'unit':unit,'context':i,'active_background':np.flatnonzero(bank['a'][i,0]).astype(int).tolist(),'gate_pattern':pattern,'corner_activations':h[i].tolist(),'signed_interaction_contribution':float(bank['tc_contributions'][i,4,unit])})
    write_json(path/'tc_anchor_deviations.json',deviations)


@torch.no_grad()
def evaluate_all(split,bank,hypotheses,parameters,clamp,native,teacher,p,q,device):
    h=torch.tensor(bank['h'],device=device);a=torch.tensor(bank['a'],device=device);readout=torch.tensor(native['D2_eff'],device=device);baseline=teacher.W2(h)@q
    h_context=torch.relu(a@torch.tensor(parameters['retained_C1'].T,device=device)+torch.tensor(parameters['offset'],device=device))
    h_invariant=torch.relu(a[...,[7,8]]@torch.tensor(parameters['pair_C1'].T,device=device)+torch.tensor(parameters['invariant_offset'],device=device))
    clamp_t=torch.tensor(clamp,device=device);metrics=[];interventions=[];candidate_arrays={};maxidentity=0
    signs=torch.tensor([1.,-1.,-1.,1.],device=device)
    for group in hypotheses:
        s=group['neurons'];offset=np.array(group['discovery_intercept']);metric=group_metrics(bank['h'],bank['output'],native['D2_eff'],native['semantic_bias'],s,offset)
        perinter=metric.pop('per_context_interaction_error');perpair=metric.pop('per_context_pair_response_mse')
        contribution=bank['native_contributions'][:,4,s];stats=signed_statistics(contribution)
        row={'population':split,'name':group['name'],'method':group['method'],'k':group['k'],'control':group['control'],**metric,'mean_positive_interaction_sum':float(stats['positive'].mean()),'mean_negative_interaction_sum':float(stats['negative'].mean()),'mean_cancellation':float(stats['cancellation'].mean()),'mean_effective_units':float(stats['effective_units'].mean()),'matched_to':group.get('matched_to'),'matching_qualified':group.get('within_25pct'),'matched_overlap':group.get('overlap_fraction')};metrics.append(row)
        if not group['control']:
            candidate_arrays[group['name']+'_interaction_error']=perinter;candidate_arrays[group['name']+'_pair_mse']=perpair
        zero_delta=None
        for kind in ('zero','clamp'):
            manipulated=h.clone();manipulated[:,:,s]=0 if kind=='zero' else clamp_t[s]
            actual=teacher.W2(manipulated)@q-baseline
            exact=(-h[:,:,s])@readout[:,s].T if kind=='zero' else (clamp_t[s]-h[:,:,s])@readout[:,s].T
            torch.testing.assert_close(actual,exact,atol=1e-9,rtol=1e-8);maxidentity=max(maxidentity,float((actual-exact).abs().max()))
            delta=(actual*signs[None,:,None]).sum(1)
            if kind=='zero':zero_delta=delta
            else:torch.testing.assert_close(delta,zero_delta,atol=1e-9,rtol=1e-8)
            record={'population':split,'name':group['name'],'method':group['method'],'k':group['k'],'intervention':kind,'g4_mean_effect':float(actual[:,:,4].mean()),'g4_mean_abs_effect':float(actual[:,:,4].abs().mean()),'g4_mean_interaction_change':float(delta[:,4].mean()),'g4_interaction_change_sd':float(delta[:,4].std(unbiased=False)),'g4_class_flip_fraction':float((((baseline+actual)[:,:,4]>=.5)!=(baseline[:,:,4]>=.5)).double().mean()),'mean_abs_collateral_effect':float(actual[:,:,[0,1,2,3,5,6,7,8]].abs().mean()),'matching_qualified':group.get('within_25pct'),'matched_to':group.get('matched_to')}
            for j in range(9):record[f'g{j}_mean_abs_effect']=float(actual[:,:,j].abs().mean());record[f'g{j}_mean_interaction_change']=float(delta[:,j].mean())
            for predictor,hhat in [('invariant_gate',h_invariant),('context2_gate',h_context)]:
                predicted=(-hhat[:,:,s])@readout[:,s].T if kind=='zero' else (clamp_t[s]-hhat[:,:,s])@readout[:,s].T
                error=predicted-actual;pd=(predicted*signs[None,:,None]).sum(1)
                record[predictor+'_g4_effect_rmse']=float(error[:,:,4].square().mean().sqrt());record[predictor+'_g4_interaction_effect_mae']=float((pd[:,4]-delta[:,4]).abs().mean());record[predictor+'_all_output_effect_rmse']=float(error.square().mean().sqrt())
                if not group['control']:candidate_arrays[group['name']+'_'+kind+'_'+predictor+'_predicted_effect']=predicted.cpu().numpy()
            template=torch.tensor(group['discovery_'+kind+'_effect_mean_corners'],device=device)[None,:,:].expand(len(h),-1,-1)
            template_delta=(template*signs[None,:,None]).sum(1)
            record['discovery_mean_effect_rmse']=float((template[:,:,4]-actual[:,:,4]).square().mean().sqrt());record['discovery_mean_interaction_mae']=float((template_delta[:,4]-delta[:,4]).abs().mean())
            if not group['control']:candidate_arrays[group['name']+'_'+kind+'_actual_effect']=actual.cpu().numpy()
            interventions.append(record)
    write_csv(OUT/f'{split}_group_metrics.csv',metrics);write_csv(OUT/f'{split}_interventions.csv',interventions);np.savez_compressed(OUT/f'{split}_candidate_effects.npz',**candidate_arrays)
    write_json(OUT/f'{split}_intervention_identity.json',{'max_actual_vs_exact_signed_effect_error':maxidentity,'constant_clamp_and_zero_have_identical_pair_interaction_effects':True,'note':'Exact identity uses realized held-out activations; it is a correctness check. Gate/mean predictions use discovery-only parameters and are the generalization test.'})
    # Paired development bootstrap: primary comparison fixed in advance, not chosen by held-out rank.
    if split=='heldout':
        rng=np.random.default_rng(C['prediction']['bootstrap_seed']);comparisons=[];n=len(h)
        for k in C['budgets']:
            guided=candidate_arrays[f'tc_guided_signed_k{k}_interaction_error'];direct=candidate_arrays[f'direct_joint_k{k}_interaction_error'];difference=np.abs(guided)-np.abs(direct)
            samples=np.array([difference[rng.integers(n,size=n)].mean() for _ in range(C['prediction']['bootstrap_replicates'])])
            comparisons.append({'k':k,'guided_minus_direct_interaction_mae':float(difference.mean()),'paired_context_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'interpretation':'Positive means guided is worse; empirical development-context resampling, not independent final confirmation.'})
        write_json(OUT/'guided_vs_direct_bootstrap.json',comparisons)

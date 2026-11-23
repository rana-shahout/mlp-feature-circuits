"""Validation fidelity and complete signed finite interactions for V2 transcoders."""
import json
import numpy as np
import torch
from src.models import SparseTranscoder
from src.coordinates import transcoder_coordinates
from src.interactions import decompose_corners
from src.runtime import write_json
from ..run import BASE,write_csv
from ..data import RULES,PAIRS,targets,keys
from ..capacity256 import signed_statistics
from .core import infer,fidelity,activation_summary,name

CASES=[(4,4,'and_78'),(6,6,'xor_1112'),(0,0,'and_01')]


def interactions(model,width,report,d,device):
    m=SparseTranscoder(12,width,12).to(device).double();m.load_state_dict(model.state_dict());m.eval().requires_grad_(False)
    p=torch.tensor(d['P'],device=device,dtype=torch.float64);q=torch.tensor(d['Q'],device=device,dtype=torch.float64);coord=transcoder_coordinates(m,p,q)
    np.savez_compressed(report/'coordinates.npz',K=coord['C1'].cpu().numpy(),A=coord['D2_eff'].cpu().numpy(),encoder_bias=coord['b1'].cpu().numpy(),semantic_bias=coord['semantic_bias'].cpu().numpy())
    source=BASE/'reports/capacity256-20261027/signed_interactions_all_widths.npz';arrays={};rows=[];summary=[];maxres=0.;case_data={}
    with np.load(source) as saved:
        for pi,indices in enumerate(list(PAIRS)+[(1,2)]):
            a=saved[f'p{pi}_corners'];assert not set(map(int,keys(a.reshape(-1,20).astype(np.uint8)))) & set(map(int,keys(d['a'][d['test']])));truth=saved[f'p{pi}_ground_truth'];teacher_c=saved[f'w256_p{pi}_contributions'];teacher_v=saved[f'w256_p{pi}_value']
            corners=torch.tensor(a,device=device,dtype=torch.float64);signs=torch.tensor([1.,-1.,-1.,1.],device=device,dtype=torch.float64)
            with torch.no_grad():
                result=decompose_corners(corners,signs,coord);z=m.latents(corners@p.T);raw=m.decoder(z);semantic=raw@q
                direct=(semantic*signs[None,:,None]).sum(1)
            torch.testing.assert_close(result['value'],direct,atol=1e-9,rtol=1e-8)
            maxres=max(maxres,float((result['value']-direct).abs().max()))
            contribution=result['contributions'].cpu().numpy();student_v=direct.cpu().numpy();scores=semantic.cpu().numpy()
            # Reconstruct the teacher's corner scores from its exact frozen neuron coordinates.
            from .run import TEACHER
            from .core import frozen_teacher
            teacher,_=frozen_teacher(TEACHER,device);teacher=teacher.double()
            with torch.no_grad():teacher_scores=(teacher((corners@p.T).reshape(-1,12)).reshape(len(a),4,12)@q).cpu().numpy()
            np.testing.assert_allclose((teacher_scores*np.array([1,-1,-1,1])[None,:,None]).sum(1),teacher_v,atol=1e-9,rtol=1e-8)
            target_corners=targets(a);allcorrect=((teacher_scores>=.5)==target_corners.astype(bool)).all((1,2))
            density=a.sum(-1);inside=(density.min(1)>=3)&(density.max(1)<=6)
            arrays.update({f'p{pi}_corners':a,f'p{pi}_contributions':contribution,f'p{pi}_student_value':student_v,f'p{pi}_teacher_value':teacher_v,f'p{pi}_teacher_contributions':teacher_c,f'p{pi}_residual':student_v-teacher_v,f'p{pi}_direct_identity_residual':result['value'].cpu().numpy()-student_v,f'p{pi}_z':z.cpu().numpy(),f'p{pi}_ground_truth':truth,f'p{pi}_teacher_corner_scores':teacher_scores,f'p{pi}_student_corner_scores':scores})
            for j,rule in enumerate(RULES):
                tc=signed_statistics(teacher_c[:,j,:]);sc=signed_statistics(contribution[:,j,:]);residual=student_v[:,j]-teacher_v[:,j]
                rulecorrect=((teacher_scores[:,:,j]>=.5)==target_corners[:,:,j].astype(bool)).all(1)
                for ci in range(len(a)):
                    row={'pair':str(indices),'pair_index':pi,'output':rule,'output_index':j,'context':ci,'density_supported':bool(inside[ci]),'all_corners_teacher_correct':bool(allcorrect[ci]),'relevant_output_teacher_correct':bool(rulecorrect[ci]),'ground_truth_interaction':float(truth[ci,j]),'interaction_residual':float(residual[ci])}
                    row.update({'teacher_'+k:float(v[ci]) for k,v in tc.items() if k!='top1_neuron'});row.update({'student_'+k:float(v[ci]) for k,v in sc.items() if k!='top1_neuron'});rows.append(row)
                for regime,mask in [('all',np.ones(len(a),bool)),('density_supported',inside),('OOD_density',~inside),('teacher_correct_output_corners',rulecorrect),('teacher_incorrect_output_corners',~rulecorrect)]:
                    if not mask.any():continue
                    summary.append({'pair_index':pi,'pair':str(indices),'output_index':j,'output':rule,'regime':regime,'n_contexts':int(mask.sum()),'interaction_mae':float(np.abs(residual[mask]).mean()),'interaction_mse':float((residual[mask]**2).mean()),'teacher_interaction_mean':float(teacher_v[mask,j].mean()),'student_interaction_mean':float(student_v[mask,j].mean()),'teacher_context_sd':float(teacher_v[mask,j].std()),'student_context_sd':float(student_v[mask,j].std()),'residual_context_sd':float(residual[mask].std()),'teacher_positive_sum':float(tc['positive'][mask].mean()),'teacher_negative_sum':float(tc['negative'][mask].mean()),'student_positive_sum':float(sc['positive'][mask].mean()),'student_negative_sum':float(sc['negative'][mask].mean()),'teacher_effective_units':float(tc['effective_units'][mask].mean()),'student_effective_units':float(sc['effective_units'][mask].mean()),'teacher_effective_fraction':float(tc['effective_fraction_of_width'][mask].mean()),'student_effective_fraction':float(sc['effective_fraction_of_width'][mask].mean()),'teacher_cancellation':float(tc['cancellation'][mask].mean()),'student_cancellation':float(sc['cancellation'][mask].mean()),'teacher_top1_share':float(tc['top1_absolute_share'][mask].mean()),'student_top1_share':float(sc['top1_absolute_share'][mask].mean()),'teacher_top5_share':float(tc['top5_absolute_share'][mask].mean()),'student_top5_share':float(sc['top5_absolute_share'][mask].mean())})
            case_data[pi]={'a':a,'z':z.cpu().numpy(),'contributions':contribution,'teacher_value':teacher_v,'student_value':student_v,'teacher_scores':teacher_scores,'student_scores':scores,'inside':inside}
    np.savez_compressed(report/'signed_interactions.npz',**arrays);write_csv(report/'interaction_contexts.csv',rows);write_csv(report/'interaction_summary.csv',summary)
    write_json(report/'interaction_checks.json',{'max_signed_sum_vs_direct_output_residual':maxres,'teacher_full_signed_sums_rechecked':True,'identical_historical_contexts':True})
    candidates(model,report,coord,case_data)
    return summary


def candidates(model,report,coord,data):
    k=coord['C1'].cpu().numpy();a=coord['D2_eff'].cpu().numpy();bias=coord['b1'].cpu().numpy();records=[];topsets=[]
    for pi,j,case in CASES:
        d=data[pi];v=d['contributions'][:,j,:];positive=np.maximum(v,0).mean(0);negative=-np.minimum(v,0).mean(0);absolute=np.abs(v).mean(0)
        top=np.argsort(-absolute);chosen=sorted(set(top[:3])|set(np.argsort(-positive)[:3])|set(np.argsort(-negative)[:3]))
        chosen=[u for u in chosen if absolute[u]>1e-12]
        for count in (1,3,5,10):
            units=top[:count];signed=v[:,units].sum(1)
            topsets.append({'case':case,'k':count,'fixed_units':units.astype(int).tolist(),'subset_vs_student_interaction_mae':float(np.abs(signed-d['student_value'][:,j]).mean()),'subset_vs_teacher_interaction_mae':float(np.abs(signed-d['teacher_value'][:,j]).mean()),'note':'Fixed set chosen on this reused diagnostic bank by mean absolute contribution; descriptive, not held-out or causal evidence'})
        for unit in chosen:
            z=d['z'][:,:,unit];patterns=[''.join(str(int(x)) for x in row) for row in (z>0)]
            counts={p:patterns.count(p) for p in sorted(set(patterns))}
            record={'case':case,'unit':int(unit),'K_complete_20_inputs':k[unit].tolist(),'encoder_bias':float(bias[unit]),'A_complete_9_outputs':a[:,unit].tolist(),'corners_order':['00','01','10','11'],'mean_corner_activations':z.mean(0).tolist(),'corner_activation_sd_across_contexts':z.std(0).tolist(),'activation_gate_pattern_counts':counts,'activation_over_all_contexts':z.tolist(),'contexts_complete_input_vectors':d['a'].astype(int).tolist(),'signed_interaction_contribution_all_contexts':v[:,unit].tolist(),'mean_signed_contribution':float(v[:,unit].mean()),'mean_absolute_contribution':float(absolute[unit]),'positive_mean':float(positive[unit]),'negative_mean':-float(negative[unit]),'interpretation':'Gate patterns and coefficients describe conditional responses only; not a proven AND/XOR detector or an original-neuron match.'}
            # Conditional nuisance contrast: identical semantic corners, varying only a15..a19.
            changes=[]
            for c0 in range(len(z)):
                for c1 in range(c0+1,len(z)):
                    if np.array_equal(d['a'][c0,:,:15],d['a'][c1,:,:15]) and not np.array_equal(d['a'][c0,:,15:],d['a'][c1,:,15:]):
                        changes.append({'context0':c0,'context1':c1,'activation_changes_four_corners':(z[c1]-z[c0]).tolist(),'signed_interaction_change':float(v[c1,unit]-v[c0,unit])})
            record['matched_distractor_context_changes']=changes;records.append(record)
    reuse0=np.abs(data[0]['contributions'][:,0,:]).mean(0);reuse1=np.abs(data[1]['contributions'][:,1,:]).mean(0);shared=np.sqrt(reuse0*reuse1)
    reuse=[{'unit':int(u),'K_a0':float(k[u,0]),'A_g0':float(a[0,u]),'A_g1':float(a[1,u]),'mean_abs_and01_to_g0_contribution':float(reuse0[u]),'mean_abs_and02_to_g1_contribution':float(reuse1[u]),'complete_K':k[u].tolist(),'complete_A':a[:,u].tolist(),'bias':float(bias[u])} for u in np.argsort(-shared)[:10]]
    write_json(report/'candidate_features.json',{'cases':records,'selection':'Union of top3 positive, negative, and absolute mean contributions, threshold>1e-12; all signed contributions retained separately','reuse_candidates':reuse,'reuse_caveat':'Same transcoder unit contributing to both relationships is descriptive shared usage, not proof of an identical Boolean circuit or original-neuron mechanism'})
    write_json(report/'fixed_topk_approximation.json',topsets)


def evaluate(model,width,coefficient,report,d,teacher,xs,device):
    from .run import SOURCE
    results={}
    for split in ('train','validation'):
        y,z=infer(model,xs[split],latents=True);result=fidelity(y,teacher[split],d['g'][d[split]],d['Q']);result['activation']=activation_summary(z);results[split]=result
        np.savez_compressed(report/f'predictions_{split}.npz',output=y,semantic=y.astype(float)@d['Q'].astype(float),vector_key=keys(d['a'][d[split]]))
        np.savez_compressed(report/f'latents_{split}.npz',z=z)
    results['dead_on_train_and_validation']=int(np.sum((np.array(results['train']['activation']['per_feature_prevalence'])==0)&(np.array(results['validation']['activation']['per_feature_prevalence'])==0)))
    write_json(report/'metrics.json',results)
    with np.load(SOURCE/'controlled_predictions_all_widths.npz') as bank:a=bank['a'].copy();g=bank['g'].copy();teacher_y=bank['w256'].copy()
    assert not set(map(int,keys(a))) & set(map(int,keys(d['a'][d['test']])))
    inputs=torch.tensor(a,device=device,dtype=torch.float32)@torch.tensor(d['P'].T,device=device);student,z=infer(model,inputs,latents=True);density=a.sum(1);contrasts={}
    for label,mask in [('density_supported',(density>=3)&(density<=6)),('OOD_density',(density<3)|(density>6))]:contrasts[label]=fidelity(student[mask],teacher_y[mask],g[mask],d['Q'])
    write_json(report/'controlled_fidelity.json',contrasts);np.savez_compressed(report/'controlled_predictions.npz',a=a,teacher=teacher_y,student=student)
    interactions(model,width,report,d,device)
    write_json(report/'complete.json',{'width':width,'lambda':coefficient,'status':'complete'})

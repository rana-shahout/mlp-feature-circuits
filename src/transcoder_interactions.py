"""Complete signed student/teacher interaction comparisons on identical probes."""

import json
from pathlib import Path

import numpy as np
import torch

from .coordinates import original_coordinates, transcoder_coordinates
from .data import RULE_IDS, targets
from .interactions import decompose_corners
from .runtime import write_json


def tensor_statistics(c):
    mass=c.abs().sum(-1)
    square=c.square().sum(-1)
    safe=torch.where(mass>0,mass,torch.ones_like(mass))
    effective=torch.where(square>0,mass.square()/torch.where(square>0,square,torch.ones_like(square)),0)
    top=torch.topk(c.abs(),min(10,c.shape[-1]),dim=-1).values
    return {"value":c.sum(-1),"positive_sum":c.clamp_min(0).sum(-1),"negative_sum":c.clamp_max(0).sum(-1),
            "absolute_mass":mass,"effective_units":effective,"effective_fraction":effective/c.shape[-1],
            "top1_mass_fraction":top[...,:1].sum(-1)/safe,"top5_mass_fraction":top[...,:5].sum(-1)/safe,
            "top10_mass_fraction":top.sum(-1)/safe,
            "cancellation_fraction":torch.where(mass>0,1-c.sum(-1).abs()/safe,0)}


def _numpy_stats(c):
    return {key:value.cpu().numpy() for key,value in tensor_statistics(c).items()}


@torch.no_grad()
def compare_interactions(config,root,device):
    from scripts.run_transcoder import load_teacher, student_from_checkpoint, verify_sources
    from scripts.diagnose_mlp import finite_corners,write_rows
    verify_sources(root,config)
    selection=json.loads((root/"selection.json").read_text())
    source=Path(config["teacher_diagnostics"])
    original=source/"seed109"
    metadata=json.loads((original/"checkpoint_metadata.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())
    if metadata["checkpoint_sha256"]!=protocol["teacher_sha256"]:
        raise RuntimeError("Original interaction archive belongs to a different teacher")
    contexts=np.load(source/"contexts.npy")
    names=json.loads((source/"context_names.json").read_text())
    np.save(root/"interaction_contexts.npy",contexts)
    write_json(root/"interaction_context_names.json",names)
    teacher,teacher_config,_,_=load_teacher(config,device)
    teacher.double()
    with np.load(Path(config["teacher_run"])/"dataset.npz") as data:
        p,q=[torch.tensor(data[key],dtype=torch.float64,device=device) for key in ("P","Q")]
    threshold=teacher_config["evaluation"]["classification_threshold"]
    teacher_coord=original_coordinates(teacher,p,q)
    reference=root/"teacher_interaction_reference";reference.mkdir(exist_ok=True)
    for order in (2,3):
        src=original/f"order{order}"
        terms=np.load(src/"terms.npy")
        expected=np.load(src/"interactions.npz")["value"]
        destination=reference/f"order{order}";destination.mkdir(exist_ok=True)
        if (destination/"complete.json").exists():continue
        teacher_raw=np.lib.format.open_memmap(destination/"corner_outputs.npy",mode="w+",dtype="float64",shape=(len(terms),len(contexts),2**order,12))
        valid=np.zeros((len(terms),len(contexts)),dtype=bool)
        statistics={}
        for start in range(0,len(terms),config["interaction_batch_size"]):
            stop=min(start+config["interaction_batch_size"],len(terms))
            a,signs=finite_corners(contexts,terms[start:stop].tolist())
            tensor=torch.tensor(a,dtype=torch.float64,device=device)
            decomposition=decompose_corners(tensor,torch.tensor(signs,device=device),teacher_coord)
            raw=teacher((tensor@p.T).reshape(-1,p.shape[0])).reshape(*a.shape[:-1],12)
            direct=(raw@q*torch.tensor(signs,device=device)[None,None,:,None]).sum(-2)
            torch.testing.assert_close(decomposition["value"],direct,atol=config["interaction_atol"],rtol=config["interaction_rtol"])
            torch.testing.assert_close(direct,torch.tensor(expected[start:stop],device=device),atol=config["interaction_atol"],rtol=config["interaction_rtol"])
            stats=_numpy_stats(decomposition["contributions"])
            if not statistics:statistics={key:np.zeros(expected.shape) for key in stats}
            for key in stats:statistics[key][start:stop]=stats[key]
            teacher_raw[start:stop]=raw.cpu().numpy()
            predicted=(raw@q).cpu().numpy()>=threshold
            valid[start:stop]=np.all(predicted==targets(a).astype(bool),axis=(-1,-2))
        teacher_raw.flush();del teacher_raw
        np.save(destination/"all_corners_teacher_correct.npy",valid)
        np.savez_compressed(destination/"statistics.npz",**statistics)
        write_json(destination/"complete.json",{"teacher_sha256":protocol["teacher_sha256"],"order":order})
    for row in selection["candidates"]:
        folder=root/row["name"]
        output=folder/"interaction_comparison";output.mkdir(exist_ok=True)
        if (output/"complete.json").exists():continue
        model,checkpoint=student_from_checkpoint(folder,device)
        model.double()
        coordinates=transcoder_coordinates(model,p,q)
        np.savez_compressed(output/"coordinates.npz",K=coordinates["C1"].cpu().numpy(),A=coordinates["D2_eff"].cpu().numpy(),
                            e=coordinates["b1"].cpu().numpy(),semantic_bias=coordinates["semantic_bias"].cpu().numpy())
        summary={};families=[]
        for order in (2,3):
            terms=np.load(original/f"order{order}"/"terms.npy")
            ref=reference/f"order{order}"
            with np.load(ref/"statistics.npz") as stats:teacher_stats={key:stats[key] for key in stats.files}
            teacher_raw=np.load(ref/"corner_outputs.npy",mmap_mode="r")
            teacher_correct=np.load(ref/"all_corners_teacher_correct.npy")
            target=np.load(original/f"order{order}"/"interactions.npz")["ground_truth"]
            destination=output/f"order{order}";destination.mkdir(exist_ok=True)
            np.save(destination/"terms.npy",terms)
            full=np.lib.format.open_memmap(destination/"contributions.npy",mode="w+",dtype="float64",shape=(len(terms),len(contexts),8,row["width"]))
            corner_residual=np.lib.format.open_memmap(destination/"full_output_corner_residual.npy",mode="w+",dtype="float64",shape=teacher_raw.shape)
            student_stats={key:np.zeros_like(value) for key,value in teacher_stats.items()}
            unit_sd=np.lib.format.open_memmap(destination/"unit_distractor_context_sd.npy",mode="w+",dtype="float64",shape=(len(terms),8,row["width"]))
            with open(destination/"signed_comparison.csv","w",newline="") as handle:
                import csv
                writer=None
                for start in range(0,len(terms),config["interaction_batch_size"]):
                    stop=min(start+config["interaction_batch_size"],len(terms))
                    a,signs=finite_corners(contexts,terms[start:stop].tolist())
                    tensor=torch.tensor(a,dtype=torch.float64,device=device)
                    result=decompose_corners(tensor,torch.tensor(signs,device=device),coordinates)
                    raw=model(tensor@p.T)
                    direct=(raw@q*torch.tensor(signs,device=device)[None,None,:,None]).sum(-2)
                    torch.testing.assert_close(result["contributions"].sum(-1),direct,atol=config["interaction_atol"],rtol=config["interaction_rtol"])
                    torch.testing.assert_close(result["positive_sum"]+result["negative_sum"],direct,atol=config["interaction_atol"],rtol=config["interaction_rtol"])
                    c=result["contributions"].cpu().numpy()
                    full[start:stop]=c
                    corner_residual[start:stop]=raw.cpu().numpy()-teacher_raw[start:stop]
                    unit_sd[start:stop]=c[:,:4].std(axis=1)
                    stats=_numpy_stats(result["contributions"])
                    for key in stats:student_stats[key][start:stop]=stats[key]
                    for t in range(stop-start):
                        for context in range(len(contexts)):
                            for j in range(8):
                                record={"term_index":start+t,"inputs":",".join(map(str,terms[start+t])),"context":names[context],"output":j,
                                        "ground_truth":float(target[start+t,context,j]),"all_corners_teacher_correct":bool(teacher_correct[start+t,context]),
                                        "interaction_residual":float(stats["value"][t,context,j]-teacher_stats["value"][start+t,context,j])}
                                for key in stats:
                                    record["teacher_"+key]=float(teacher_stats[key][start+t,context,j])
                                    record["student_"+key]=float(stats[key][t,context,j])
                                if writer is None:writer=csv.DictWriter(handle,fieldnames=list(record));writer.writeheader()
                                writer.writerow(record)
            full.flush();corner_residual.flush();unit_sd.flush()
            np.savez_compressed(destination/"statistics.npz",**student_stats)
            residual=student_stats["value"]-teacher_stats["value"]
            np.savez_compressed(destination/"residuals.npz",interaction=residual,
                                student_distractor_sd=student_stats["value"][:,:4].std(1),
                                teacher_distractor_sd=teacher_stats["value"][:,:4].std(1),residual_distractor_sd=residual[:,:4].std(1))
            groups={"all":np.ones(residual.shape,dtype=bool),"nonzero_ground_truth":target!=0,
                    "teacher_effect_over_1e_minus6":np.abs(teacher_stats["value"])>1e-6,
                    "teacher_success_corners":np.broadcast_to(teacher_correct[:,:,None],residual.shape),
                    "teacher_failure_corners":np.broadcast_to(~teacher_correct[:,:,None],residual.shape)}
            summary[str(order)]={}
            summary[str(order)]["distractor_context_stability"]={
                "teacher_mean_sd":float(teacher_stats["value"][:,:4].std(1).mean()),
                "student_mean_sd":float(student_stats["value"][:,:4].std(1).mean()),
                "mean_residual_sd":float(residual[:,:4].std(1).mean()),
                "contexts":names[:4],"note":"Same semantic background; nuisance identities/counts vary."}
            for label,mask in groups.items():
                summary[str(order)][label]={"n":int(mask.sum()),
                    "mean_absolute_residual":float(np.abs(residual[mask]).mean()) if mask.any() else None,
                    "interaction_mse":float((residual[mask]**2).mean()) if mask.any() else None,
                    "signed_value_agreement":float((np.sign(student_stats["value"][mask])==np.sign(teacher_stats["value"][mask])).mean()) if mask.any() else None,
                    "teacher_mean_effective_units":float(teacher_stats["effective_units"][mask].mean()) if mask.any() else None,
                    "student_mean_effective_units":float(student_stats["effective_units"][mask].mean()) if mask.any() else None,
                    "teacher_mean_effective_fraction":float(teacher_stats["effective_fraction"][mask].mean()) if mask.any() else None,
                    "student_mean_effective_fraction":float(student_stats["effective_fraction"][mask].mean()) if mask.any() else None}
            for ids,j,label in [((8,9),4,"and3_pair89"),((8,9,10),4,"and3_triple"),((6,7),3,"xor_pair"),((4,16,17),7,"mixed_xor_triple")]:
                if len(ids)!=order:continue
                index=np.flatnonzero(np.all(terms==np.asarray(ids),axis=1))[0]
                c=np.asarray(full[index,:,j,:])
                for k in range(c.shape[-1]):
                    families.append({"family":label,"output_rule":RULE_IDS[j],"unit":k,
                                     "mean_signed_contribution":float(c[:,k].mean()),"mean_absolute_contribution":float(np.abs(c[:,k]).mean()),
                                     "positive_mean":float(np.maximum(c[:,k],0).mean()),"negative_mean":float(np.minimum(c[:,k],0).mean()),
                                     "context_sd":float(c[:,k].std()),"distractor_context_sd":float(c[:4,k].std()),"input_indices":",".join(map(str,ids))})
            del full,corner_residual,unit_sd
            print(f"INTERACTIONS {row['name']} order={order} complete",flush=True)
        write_rows(output/"feature_interaction_families.csv",families)
        write_json(output/"metrics.json",summary)
        write_json(output/"complete.json",{"student_sha256":row["checkpoint_sha256"],"teacher_sha256":protocol["teacher_sha256"],
                                             "contexts":str(source/"contexts.npy"),"note":"Complete signed student contributions; exact original-neuron archive remains referenced and unchanged."})
    verify_sources(root,config)

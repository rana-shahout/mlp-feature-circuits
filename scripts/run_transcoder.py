#!/usr/bin/env python3
"""Frozen seed-109 transcoder workflow. Final population is sealed by default."""

import argparse
from copy import deepcopy
import fcntl
from itertools import combinations, product
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from scripts.diagnose_mlp import finite_corners, write_rows
from scripts.prepare_coverage import matched_rows
from src.data import sample_dataset, targets, vector_keys
from src.evaluation import predict
from src.models import SparseTranscoder
from src.runtime import load_checkpoint, load_config, seed_everything, select_device, write_json
from src.train_mlp import config_digest, environment_record, file_digest, model_from_config, verify_checkpoint
from src.train_transcoder import fit_transcoder, validate_transcoder_config
from src.transcoder_evaluation import activation_statistics, fidelity_metrics, student_predict


def configuration_name(width, coefficient):
    return f"width{width}-l1-{coefficient:.6g}"


def frozen_source_hashes(config):
    paths = []
    for run in config["frozen_models"]:
        paths += [Path(run)/name for name in ("best.pt", "last.pt", "config.yaml", "dataset.npz")]
    paths += [Path(config["previous_evaluation"]), Path(config["teacher_diagnostics"])/"contexts.npy"]
    return {str(path.resolve()): file_digest(path) for path in paths}


def verify_sources(root, config):
    expected = json.loads((root/"protocol.json").read_text())
    if expected["config"] != config or expected["sources"] != frozen_source_hashes(config):
        raise RuntimeError("Frozen protocol or source teacher artifacts changed")


def load_teacher(config, device):
    run = Path(config["teacher_run"])
    c = load_config(run/"config.yaml")
    cp = load_checkpoint(run/"best.pt")
    verify_checkpoint(cp,c,run/"dataset.npz")
    dims = {"d_in": cp["model"]["W1.weight"].shape[1], "d_hidden": cp["model"]["W1.weight"].shape[0], "d_out": cp["model"]["W2.weight"].shape[0]}
    if dims != c["model"] or cp["model"]["W2.weight"].shape[1] != dims["d_hidden"] or c["seeds"]["model"] != 109:
        raise ValueError("Teacher checkpoint/config dimensions or seed disagree")
    model = model_from_config(c).to(device)
    model.load_state_dict(cp["model"])
    model.eval().requires_grad_(False)
    print(f"Frozen teacher seed109: {dims}",flush=True)
    return model,c,cp,dims


def prepare(config, root):
    if root.exists():
        verify_sources(root,config)
        if not (root/"reserved_final"/"manifest.json").exists():
            raise RuntimeError("Incomplete preparation; use a new experiment directory to preserve partial artifacts")
        print(f"Using frozen experiment: {root}",flush=True)
        return
    root.mkdir(parents=True,exist_ok=False)
    (root/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    shutil.copyfile(ROOT/"SPEC.md",root/"SPEC.snapshot.md")
    teacher,teacher_config,checkpoint,dims=load_teacher(config,torch.device("cpu"))
    protocol={"config":config,"sources":frozen_source_hashes(config),"teacher_dimensions":dims,
              "teacher_epoch":checkpoint["epoch"],"teacher_sha256":file_digest(Path(config["teacher_run"])/"best.pt"),
              "scientific_decision":"Imitate actual teacher including errors; unit decoder columns fix ReLU/L1 scale degeneracy; no teacher updates."}
    write_json(root/"protocol.json",protocol)
    final=config["final_population"]
    diagnostic=Path(config["teacher_diagnostics"])
    if not (diagnostic/"complete.json").exists():
        raise RuntimeError("Original teacher diagnostics must be complete")
    contexts=np.load(diagnostic/"contexts.npy")
    exclusions=[]
    with np.load(Path(config["teacher_run"])/"dataset.npz",allow_pickle=False) as data, np.load(config["previous_evaluation"],allow_pickle=False) as prior:
        for key in ("a","contrast_a","exclusion_a"):
            if key in data:
                exclusions.append(vector_keys(data[key]))
        exclusions += [vector_keys(prior["sample_a"]),vector_keys(prior["contrast_a"])]
        # Reserve against every already inspected pair/triple corner.
        for order in (2,3):
            terms=list(combinations(range(32),order))
            for start in range(0,len(terms),64):
                corners,_=finite_corners(contexts,terms[start:start+64])
                exclusions.append(vector_keys(corners).reshape(-1))
        # Include full rule truth-table and single-bit nuisance probes too.
        from src.data import ANTECEDENTS
        for ids in list(ANTECEDENTS)+[(i,) for i in range(18,32)]:
            for bits in product((0,1),repeat=len(ids)):
                a=contexts.copy();a[:,list(ids)]=bits;exclusions.append(vector_keys(a))
        excluded=np.unique(np.concatenate(exclusions))
        d=deepcopy(teacher_config["data"])
        d["n_samples"]=final["proposal_count"]
        proposals,mode,requested,generation=sample_dataset(d,final["sample_seed"])
        keep=np.flatnonzero(~np.isin(vector_keys(proposals),excluded))[:final["sample_limit"]]
        a=proposals[keep]
        if len(a)<512:
            raise RuntimeError(f"Only {len(a)} unseen final examples; increase bounded proposal budget")
        contrast,groups,members=matched_rows(final["contexts_per_rule"],final["contrast_seed"],teacher_config["coverage"],
                                              np.concatenate([excluded,vector_keys(a)]))
        reserved=root/"reserved_final";reserved.mkdir()
        np.savez_compressed(reserved/"population.npz",sample_a=a,sample_g=targets(a),sample_mode=mode[keep],
                            contrast_a=contrast,contrast_g=targets(contrast),contexts=groups,memberships=members,P=data["P"],Q=data["Q"])
        write_json(reserved/"manifest.json",{"status":"reserved_unscored","population_sha256":file_digest(reserved/"population.npz"),
                  "sample_n":len(a),"contrast_n":len(contrast),"generation":generation,
                  "note":"Independent seeds, disjoint from all saved training/validation/test and previously inspected probes. Exclusion conditions the distribution and can remove local truth-table corners. No teacher or student predictions were computed on this population."})
    print("Prepared independent final population; it remains unscored.",flush=True)


@torch.no_grad()
def teacher_cache(teacher,data,device,batch):
    p=torch.as_tensor(data["P"],device=device)
    cache={}
    for name in ("train","validation"):
        a=data["a"][data[name]]
        xs,ys=[],[]
        for start in range(0,len(a),batch):
            x=torch.as_tensor(a[start:start+batch],dtype=torch.float32,device=device)@p.T
            xs.append(x);ys.append(teacher(x).detach())
        cache[name]=(torch.cat(xs),torch.cat(ys))
    if any(p.requires_grad for p in teacher.parameters()):
        raise RuntimeError("Teacher must be frozen")
    return cache


def train_width(config,root,width,device):
    if width not in config["widths"]:
        raise ValueError("Width absent from frozen plan")
    verify_sources(root,config)
    teacher,c,cp,dims=load_teacher(config,device)
    with np.load(Path(config["teacher_run"])/"dataset.npz",allow_pickle=False) as data:
        cache=teacher_cache(teacher,data,device,config["evaluation_batch_size"])
    for coefficient in config["lambdas"]:
        name=configuration_name(width,coefficient)
        folder=root/name;folder.mkdir(exist_ok=True)
        with open(folder/".lock","w") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (folder/"trained.json").exists():
                print(f"Already trained: {name}",flush=True);continue
            identity={"protocol_sha256":file_digest(root/"protocol.json"),"width":width,"lambda":coefficient,
                      "teacher_sha256":file_digest(Path(config["teacher_run"])/"best.pt")}
            model=fit_transcoder(config,width,coefficient,*cache["train"],*cache["validation"],folder,identity)
            if any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Teacher acquired gradients")
            write_json(folder/"trained.json",{"identity":identity,"checkpoint_sha256":file_digest(folder/"best.pt")})
    verify_sources(root,config)


def select(config,root):
    verify_sources(root,config)
    candidates=[]
    for width in config["widths"]:
        for coefficient in config["lambdas"]:
            folder=root/configuration_name(width,coefficient)
            if not (folder/"trained.json").exists():
                raise RuntimeError(f"Training incomplete: {folder}")
            row=json.loads((folder/"validation.json").read_text())
            row.update(name=folder.name,checkpoint_sha256=file_digest(folder/"best.pt"))
            candidates.append(row)
    frontier=[row for row in candidates if not any(other["full_output_mse"]<=row["full_output_mse"] and other["mean_l0"]<=row["mean_l0"]
               and (other["full_output_mse"]<row["full_output_mse"] or other["mean_l0"]<row["mean_l0"]) for other in candidates)]
    choice={"basis":"validation_full_output_mse_and_mean_l0_only","best_fidelity":min(candidates,key=lambda row:row["full_output_mse"])["name"],
            "frontier":[row["name"] for row in frontier],"candidates":candidates}
    path=root/"selection.json"
    if path.exists() and json.loads(path.read_text())!=choice:
        raise RuntimeError("Selection is frozen; candidate files changed")
    write_json(path,choice)
    write_rows(root/"validation_tradeoff.csv",candidates)
    print(json.dumps(choice,indent=2),flush=True)


def student_from_checkpoint(folder,device):
    cp=load_checkpoint(folder/"best.pt")
    state=cp["model"]
    model=SparseTranscoder(state["encoder.weight"].shape[1],state["encoder.weight"].shape[0],state["decoder.weight"].shape[0]).to(device)
    model.load_state_dict(state);model.eval().requires_grad_(False)
    return model,cp


def evaluate(config,root,device,scope):
    verify_sources(root,config)
    selection=json.loads((root/"selection.json").read_text())
    if any(file_digest(root/row["name"]/"best.pt")!=row["checkpoint_sha256"] for row in selection["candidates"]):
        raise RuntimeError("A selected checkpoint changed")
    if scope=="final" and (root/"final_scored.json").exists():
        raise RuntimeError("Final population already scored; do not repeatedly reuse it for tuning")
    teacher,c,_,_=load_teacher(config,device)
    threshold=c["evaluation"]["classification_threshold"]
    with np.load(Path(config["teacher_run"])/"dataset.npz",allow_pickle=False) as data:
        if scope=="final":
            source=root/"reserved_final"/"population.npz"
            if file_digest(source)!=json.loads((root/"reserved_final"/"manifest.json").read_text())["population_sha256"]:
                raise RuntimeError("Reserved population changed")
        else:
            source=Path(config["previous_evaluation"])
        with np.load(source,allow_pickle=False) as saved:
            if scope=="final":
                populations={"final_sample":(saved["sample_a"],saved["sample_g"]),"final_contrast":(saved["contrast_a"],saved["contrast_g"])}
            else:
                populations={name:(data["a"][data[name]],data["g"][data[name]]) for name in ("train","validation","test")}
                populations.update(previous_sample=(saved["sample_a"],saved["sample_g"]),previous_contrast=(saved["contrast_a"],saved["contrast_g"]))
            names=selection["frontier"] if scope=="final" else [row["name"] for row in selection["candidates"]]
            teacher_outputs={name:predict(teacher,a,data["P"],device,config["evaluation_batch_size"])["output"] for name,(a,g) in populations.items()}
            for name in names:
                folder=root/name;model,cp=student_from_checkpoint(folder,device)
                output=folder/("final_evaluation" if scope=="final" else "diagnostic_evaluation")
                output.mkdir(exist_ok=False)
                results={}
                for pop,(a,g) in populations.items():
                    y,z=student_predict(model,a,data["P"],device,config["evaluation_batch_size"])
                    t=teacher_outputs[pop]
                    results[pop]=fidelity_metrics(y,t,g,data["Q"],threshold)
                    results[pop]["activation"]=activation_statistics(z)
                    np.savez_compressed(output/f"{pop}.npz",teacher=t,student=y,residual=y-t,z=z)
                    if pop.endswith("contrast"):
                        # Association with TEACHER errors, not a claim these units cause errors.
                        ts=t.astype(np.float64)@data["Q"].astype(np.float64)
                        wrong=(ts>=threshold)!=g.astype(bool)
                        a_write=data["Q"].T@model.decoder.weight.detach().cpu().numpy()
                        rows=[]
                        for j in range(8):
                            for k in range(z.shape[1]):
                                rows.append({"output":j,"unit":k,"teacher_error_n":int(wrong[:,j].sum()),"teacher_success_n":int((~wrong[:,j]).sum()),
                                   "activation_prevalence_error":float((z[wrong[:,j],k]>0).mean()) if wrong[:,j].any() else None,
                                   "activation_prevalence_success":float((z[~wrong[:,j],k]>0).mean()) if (~wrong[:,j]).any() else None,
                                   "mean_signed_write_on_error":float((z[wrong[:,j],k]*a_write[j,k]).mean()) if wrong[:,j].any() else None,
                                   "mean_signed_write_on_success":float((z[~wrong[:,j],k]*a_write[j,k]).mean()) if (~wrong[:,j]).any() else None})
                        write_rows(output/f"feature_error_associations_{pop}.csv",rows)
                write_json(output/"metrics.json",results)
                print(f"EVALUATED {name} scope={scope}",flush=True)
    if scope=="final":
        write_json(root/"final_scored.json",{"selection_sha256":file_digest(root/"selection.json"),"models":names})
    verify_sources(root,config)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["prepare","smoke","train","select","evaluate","interactions","report"])
    p.add_argument("--config",default="configs/transcoder_v1.yaml")
    p.add_argument("--run-dir",required=True)
    p.add_argument("--width",type=int)
    p.add_argument("--scope",choices=["diagnostic","final"],default="diagnostic")
    args=p.parse_args();root=Path(args.run_dir)
    config=yaml.safe_load(Path(args.config).read_text())
    validate_transcoder_config(config)
    try:
        if args.command=="prepare":
            prepare(config,root);return
        verify_sources(root,config)
        if args.command=="select":
            select(config,root);return
        if args.command=="report":
            from src.transcoder_report import write_report
            write_report(config,root);return
        seed_everything(config["student_seed"],True)
        device=select_device(config.get("device","cuda"))
        write_json(root/f"environment_{args.command}_{args.width}.json",environment_record(device))
        if args.command=="smoke":
            teacher,c,_,_=load_teacher(config,device)
            with np.load(Path(config["teacher_run"])/"dataset.npz") as data:
                a=torch.as_tensor(data["a"][data["train"][:32]],dtype=torch.float32,device=device)
                x=a@torch.as_tensor(data["P"],device=device).T
                from src.train_transcoder import imitation_loss
                student=SparseTranscoder(x.shape[1],128,12).to(device)
                with torch.no_grad():y=teacher(x)
                z=student.latents(x);loss,_,_=imitation_loss(student.decoder(z),y,z,.001);loss.backward()
                assert all(p.grad is None for p in teacher.parameters())
                assert torch.isfinite(loss)
                print("CUDA smoke passed; teacher has no gradients; smoke student discarded",flush=True)
        elif args.command=="train":
            train_width(config,root,args.width,device)
        elif args.command=="evaluate":
            evaluate(config,root,device,args.scope)
        elif args.command=="interactions":
            from src.transcoder_interactions import compare_interactions
            compare_interactions(config,root,device)
        verify_sources(root,config)
    except BaseException:
        if root.exists():
            (root/f"failure_{args.command}_{args.width}.log").write_text(traceback.format_exc())
        raise


if __name__=="__main__":main()

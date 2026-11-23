"""Read saved results; never train or change selection based on diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np

from .runtime import write_json


def write_report(config,root):
    from scripts.run_transcoder import verify_sources
    verify_sources(root,config)
    selected=json.loads((root/"selection.json").read_text())
    protocol=json.loads((root/"protocol.json").read_text())
    lines=["# First sparse transcoder experiment", "", "## Scientific target and scope", "",
           "Teacher: frozen original MLP seed 109, dimensions 16 → 1024 → 12, selected previously using validation metrics. The target is its actual full 12-dimensional output, including errors, not Qg. All five original checkpoint pairs are unchanged.",
           "Students: ReLU 16 → 128/256 → 12. Decoder columns have fixed unit L2 norm to remove the ReLU/L1 rescaling loophole. L1 is mean over examples of the SUM over all latent activations. Training uniformly samples the saved teacher training partition; no Boolean-label reweighting or direct semantic supervision is used.",
           "This is a single student initialization per width/penalty. Widths are smaller than the original teacher, so reduced unit count alone cannot establish recovery of a simpler mechanism.","",
           "## Validation-only selection and sparsity–fidelity tradeoff", "",
           f"Best fidelity checkpoint/configuration: **{selected['best_fidelity']}**. Every checkpoint minimizes validation full-output reconstruction MSE within its run. The validation-only non-dominated MSE/mean-L0 frontier is: {', '.join(selected['frontier'])}.",
           "Selection was frozen before examining original-test, previous-contrast, or controlled-interaction diagnostics. The independently seeded final population is reserved and **unscored** in this report.","",
           "| Configuration | Validation teacher MSE | Validation L0 | Contrast teacher MSE | Contrast semantic MSE | Exact teacher agreement | Student task exact | Teacher-error-row agreement |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows=[]
    for row in selected["candidates"]:
        name=row["name"]
        m=json.loads((root/name/"diagnostic_evaluation"/"metrics.json").read_text())["previous_contrast"]
        interaction=json.loads((root/name/"interaction_comparison"/"metrics.json").read_text())
        f=m["fidelity"]
        failure=f["teacher_failure_rows"]["exact_vector_agreement"]
        lines.append(f"| {name} | {row['full_output_mse']:.6g} | {row['mean_l0']:.2f} | {f['full_output_mse']:.6g} | {f['semantic_mse']:.6g} | {f['exact_vector_agreement']:.4f} | {m['student_task']['exact_vector_accuracy']:.4f} | {failure} |")
        item={**row,"contrast_full_mse":f["full_output_mse"],"contrast_semantic_mse":f["semantic_mse"],
              "contrast_teacher_agreement":f["exact_vector_agreement"],"student_task_exact":m["student_task"]["exact_vector_accuracy"]}
        item["pair_mae"]=interaction["2"]["nonzero_ground_truth"]["mean_absolute_residual"]
        item["triple_mae"]=interaction["3"]["nonzero_ground_truth"]["mean_absolute_residual"]
        item["pair_all_mse"]=interaction["2"]["all"]["interaction_mse"]
        item["triple_all_mse"]=interaction["3"]["all"]["interaction_mse"]
        item["pair_teacher_failure_mae"]=interaction["2"]["teacher_failure_corners"]["mean_absolute_residual"]
        item["triple_teacher_failure_mae"]=interaction["3"]["teacher_failure_corners"]["mean_absolute_residual"]
        rows.append(item)
    lines += ["", "Exact teacher agreement requires all eight thresholded semantic outputs to match the teacher on a row; student task exact instead compares all eight with Boolean labels. These quantities are deliberately separate. Teacher-error-row agreement conditions on rows where any teacher output is wrong; correcting such errors can reduce teacher fidelity.","",
              "## Preservation of nonlinear computation and concentration", "",
              "| Configuration | Nonzero-target pair residual MAE vs teacher | Triple residual MAE vs teacher | Teacher effective neurons (triples) | Student effective units | Teacher effective fraction | Student effective fraction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m=json.loads((root/row["name"]/"interaction_comparison"/"metrics.json").read_text())
        s=m["3"]["nonzero_ground_truth"]
        lines.append(f"| {row['name']} | {row['pair_mae']:.5g} | {row['triple_mae']:.5g} | {s['teacher_mean_effective_units']:.2f} | {s['student_mean_effective_units']:.2f} | {s['teacher_mean_effective_fraction']:.4f} | {s['student_mean_effective_fraction']:.4f} |")
    best_interaction=min(rows,key=lambda r:r["pair_all_mse"]+r["triple_all_mse"])
    best_row=next(row for row in rows if row["name"]==selected["best_fidelity"])
    best_behavior=json.loads((root/best_row["name"]/"diagnostic_evaluation"/"metrics.json").read_text())["previous_contrast"]
    best_signed=json.loads((root/best_row["name"]/"interaction_comparison"/"metrics.json").read_text())["3"]["nonzero_ground_truth"]
    findings=["## Main findings", "",
        f"The validation-selected configuration is **{best_row['name']}**, with validation full-output teacher MSE {best_row['full_output_mse']:.6g} and mean L0 {best_row['mean_l0']:.2f}.",
        f"On the previously inspected contrasts, full-vector teacher agreement is **{100*best_behavior['fidelity']['exact_vector_agreement']:.2f}%**. Student Boolean-task exact accuracy is {100*best_behavior['student_task']['exact_vector_accuracy']:.2f}%, versus {100*best_behavior['teacher_task']['exact_vector_accuracy']:.2f}% for the frozen teacher. Task accuracy is not the training or selection objective.",
        f"On the nonzero-ground-truth probe subset, pair/triple interaction residual MAEs against the ACTUAL teacher are **{best_row['pair_mae']:.4f} / {best_row['triple_mae']:.4f}**. All probes, including spurious teacher interactions and teacher-failure corners, are assessed separately below. These remaining residuals preclude claiming exact nonlinear imitation.",
        f"On nonzero-target triples, effective units average {best_signed['student_mean_effective_units']:.2f} in the student versus {best_signed['teacher_mean_effective_units']:.2f} original neurons. However, those are {100*best_signed['student_mean_effective_fraction']:.2f}% versus {100*best_signed['teacher_mean_effective_fraction']:.2f}% of model width. Fewer absolute units therefore do not establish stronger width-normalized concentration or a recovered sparse original mechanism.",
        "The low-penalty student is a candidate hypothesis generator with quantified failures, not a faithful substitute for every teacher error. High-penalty runs trade substantial fidelity for sparsity. No teacher retraining, original-neuron intervention, or subsequent experiment is launched by this report.", ""]
    lines=lines[:2]+findings+lines[2:]
    lines += ["", f"Diagnostic interaction-fidelity leader among this fixed sweep: **{best_interaction['name']}**, using pair-MSE + triple-MSE over ALL measured teacher interactions, including non-Boolean/spurious effects. This is a descriptive post-selection comparison, not a change to validation selection or a claim on untouched data.",
              "Participation-ratio effective unit count is (sum |contribution|)^2 / sum contribution^2, with zero for an all-zero vector. It handles cancellation and is invariant to uniform rescaling. Reported fractions divide by each model's width; top-1/5/10 absolute-mass fractions, complete positive/negative sums, and cancellation fractions are saved for every case.",
              "Fewer effective units are meaningful only together with fidelity. Width reduction and poor reconstruction can mechanically reduce these counts. Compare each L1 run with its same-width zero-L1 control. No unit identity correspondence with original neurons is assumed.",
              "All 496 input pairs and 4,960 triples use the exact same 15 saved contexts for both models. Full signed student contributions and full 12-dimensional corner residuals are retained. Original-neuron contributions remain in the referenced original diagnostic archive. Every student signed sum was checked against direct student finite differences; teacher reference values were checked against direct frozen-teacher differences.",
              "Distractor-context stability is measured on the shared zero/2/4/6-distractor backgrounds, holding other semantic inputs fixed. Per-unit context standard deviations and teacher/student/residual interaction standard deviations are saved; variation across unrelated semantic gate conditions is not labeled nuisance stability.","",
              "| Configuration | All-pair MSE vs actual teacher | All-triple MSE vs actual teacher | Pair MAE on teacher-failure corners | Triple MAE on teacher-failure corners |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['pair_all_mse']:.6g} | {row['triple_all_mse']:.6g} | {row['pair_teacher_failure_mae']} | {row['triple_teacher_failure_mae']} |")
    lines += ["",
              "| Configuration | Pair residual SD across distractor contexts | Triple residual SD across distractor contexts |", "|---|---:|---:|"]
    for row in rows:
        measured=json.loads((root/row["name"]/"interaction_comparison"/"metrics.json").read_text())
        lines.append(f"| {row['name']} | {measured['2']['distractor_context_stability']['mean_residual_sd']:.5g} | {measured['3']['distractor_context_stability']['mean_residual_sd']:.5g} |")
    lines += ["",
              "## Three-way AND and XOR feature candidates", "",
              "The table displays features with the largest mean absolute realized interaction contribution, retaining signs. These are feature candidates, not isolated Boolean concepts. Every unit's contributions are saved in feature_interaction_families.csv. Features/units are local to each frozen student checkpoint and are not aligned across independently trained configurations.","",
              "| Configuration | Family | Unit | Mean signed contribution | Mean absolute contribution | Context SD |", "|---|---|---:|---:|---:|---:|"]
    candidates=[]
    for row in rows:
        with open(root/row["name"]/"interaction_comparison"/"feature_interaction_families.csv") as handle:
            features=list(csv.DictReader(handle))
        for family in ("and3_triple","xor_pair","mixed_xor_triple"):
            chosen=sorted((f for f in features if f["family"]==family and float(f["mean_absolute_contribution"])>0),key=lambda f:float(f["mean_absolute_contribution"]),reverse=True)[:3]
            for f in chosen:
                lines.append(f"| {row['name']} | {family} | {f['unit']} | {float(f['mean_signed_contribution']):+.5g} | {float(f['mean_absolute_contribution']):.5g} | {float(f['context_sd']):.5g} |")
                with np.load(root/row["name"]/"interaction_comparison"/"coordinates.npz") as coordinates:
                    unit=int(f["unit"]);j={"and3_triple":4,"xor_pair":3,"mixed_xor_triple":7}[family]
                    inputs=[int(i) for i in f["input_indices"].split(',')]
                    detector=[float(coordinates["K"][unit,i]) for i in inputs]
                    bias=float(coordinates["e"][unit]);write=float(coordinates["A"][j,unit])
                candidates.append({"configuration":row["name"],**f,"semantic_input_indices":inputs,"K_detector_weights":detector,
                                   "encoder_bias":bias,"semantic_write":write,
                                   "checkpoint_sha256":row["checkpoint_sha256"],
                                   "status":"candidate_for_original_weight_investigation_not_mechanistic_alignment"})
    lines += ["", "## Association with difficult teacher examples", "",
              "Activation prevalence and signed output writes are compared between teacher-success and teacher-error subsets separately for every semantic output. The following g4 features show the largest activation-prevalence differences; small supports and correlated activations prevent causal interpretation.","",
              "| Configuration | Unit | Teacher-error support | Activation prevalence on error | On success | Mean signed g4 write on error |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        with open(root/row["name"]/"diagnostic_evaluation"/"feature_error_associations_previous_contrast.csv") as handle:
            features=[f for f in csv.DictReader(handle) if f["output"]=="4" and f["activation_prevalence_error"]
                      and abs(float(f["activation_prevalence_error"])-float(f["activation_prevalence_success"]))>0]
        for f in sorted(features,key=lambda f:abs(float(f["activation_prevalence_error"])-float(f["activation_prevalence_success"])),reverse=True)[:3]:
            lines.append(f"| {row['name']} | {f['unit']} | {f['teacher_error_n']} | {float(f['activation_prevalence_error']):.4f} | {float(f['activation_prevalence_success']):.4f} | {float(f['mean_signed_write_on_error']):+.5g} |")
    lines += ["", "## Preservation of each teacher error type in the validation-selected model", "",
              "| Output | Teacher error count | Erroneous bit preserved fraction | Erroneous bit corrected fraction |", "|---|---:|---:|---:|"]
    for entry in best_behavior["fidelity"]["per_output"]:
        lines.append(f"| {entry['rule']} | {entry['teacher_error_count']} | {entry['teacher_errors_preserved_fraction']} | {entry['teacher_errors_corrected_fraction']} |")
    lines += ["", "These are per-output error bits, not whole-row agreement. Whole-row agreement on teacher-error rows also requires all other outputs to match. Error preservation is a fidelity measure; an accidental Boolean correction is not an optimization success for this study.", "",
              "## Interpretation and next investigation", "",
              "Use candidates from configurations that preserve the teacher with acceptable residuals; sparse but low-fidelity students can invent different rules. Follow candidate K=EP detectors, biases and A=Q.T B signed writes into the exact original C1/D2_eff neuron contributions on the same contexts. Do not equate a transcoder feature with a semantic output, or a student unit with one original neuron.",
              "Shared behavior and smaller contribution counts do not prove identical internal mechanisms. No original-neuron ablation, clamping, or weight editing was performed in this experiment. Such interventions and matched controls require a subsequent scientific decision.",
              "All teacher errors are retained as targets; student disagreement and any accidental corrections remain failure evidence for imitation. This report does not justify improving the original Boolean-task model.",
              "The new reserved population was generated with independent seeds after excluding training, validation, original test, earlier evaluation populations and all inspected diagnostic probes. Exclusion conditions the population; it is not an unqualified IID claim. It must only be scored after committing the validation-selected frontier; do not use it for another sweep.","",
              "![Validation sparsity fidelity frontier](sparsity_fidelity.png)"]
    write_json(root/"feature_candidates.json",candidates)
    write_json(root/"report_metrics.json",rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for width in config["widths"]:
        points=sorted([r for r in rows if r["width"]==width],key=lambda r:r["mean_l0"])
        ax.plot([r["mean_l0"] for r in points],[r["full_output_mse"] for r in points],"o-",label=f"width {width}")
        for r in points:ax.annotate(f"λ={r['lambda']:g}",(r["mean_l0"],r["full_output_mse"]),fontsize=8)
    ax.set_xlabel("Validation mean L0 (strict positive activations)");ax.set_ylabel("Validation full-output MSE vs teacher");ax.legend();fig.tight_layout()
    fig.savefig(root/"sparsity_fidelity.png",dpi=160);plt.close(fig)
    verify_sources(root,config)
    write_json(root/"complete.json",{"teacher_sources_unchanged":True,"final_population_scored":False,
                                    "selection_sha256":__import__('hashlib').sha256((root/'selection.json').read_bytes()).hexdigest()})
    (root/"report.md").write_text("\n".join(lines)+"\n")
    print(f"Research report: {root}/report.md",flush=True)

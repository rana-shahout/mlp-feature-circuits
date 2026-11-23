"""Only frozen-teacher full outputs enter the optimization target."""

import json
import time

import numpy as np
import torch

from .models import SparseTranscoder
from .runtime import load_checkpoint, save_checkpoint, seed_everything, write_json
from .train_mlp import epoch_learning_rate


def validate_transcoder_config(config):
    t=config["training"]
    if not config["widths"] or any(not isinstance(w,int) or w<=0 for w in config["widths"]) or len(set(config["widths"]))!=len(config["widths"]):
        raise ValueError("Widths must be distinct positive integers")
    penalties=np.asarray(config["lambdas"],dtype=float)
    if not len(penalties) or not np.isfinite(penalties).all() or np.any(penalties<0) or 0 not in penalties or len(set(penalties))!=len(penalties):
        raise ValueError("L1 sweep must be finite, nonnegative, distinct, and include zero")
    required={"checkpoint_selection":"validation_full_output_mse","decoder_constraint":"unit_l2_columns",
              "l1_reduction":"mean_over_examples_of_sum_over_units","sampling":"uniform_original_teacher_train_partition",
              "lr_schedule":"cosine"}
    if any(t.get(key)!=value for key,value in required.items()):
        raise ValueError("Unknown transcoder objective/selection/normalization protocol")
    if any(not isinstance(t[key],int) or t[key]<=0 for key in ("epochs","batch_size")):
        raise ValueError("Epochs and batch size must be positive integers")
    if not np.isfinite([t["learning_rate"],t["minimum_learning_rate"]]).all() or not 0<t["minimum_learning_rate"]<=t["learning_rate"]:
        raise ValueError("Learning-rate bounds must be finite and positive")
    if config["final_population"].get("score_in_initial_run") is not False:
        raise ValueError("Initial workflow must reserve the final population without scoring it")


def imitation_loss(prediction, teacher_output, z, coefficient):
    mse = torch.nn.functional.mse_loss(prediction, teacher_output, reduction="mean")
    l1 = z.sum(dim=1).mean()  # mean_B sum_K |z|, with nonnegative ReLU z.
    return mse + coefficient*l1, mse, l1


@torch.no_grad()
def validation(model, x, y, batch):
    mse_sum, l1_sum, l0_sum = 0., 0., 0.
    for start in range(0, len(x), batch):
        xb, yb = x[start:start+batch], y[start:start+batch]
        z = model.latents(xb)
        mse_sum += (model.decoder(z)-yb).square().sum().item()
        l1_sum += z.sum().item()
        l0_sum += (z>0).sum().item()
    return {"full_output_mse": mse_sum/y.numel(), "mean_l1_sum": l1_sum/len(x), "mean_l0": l0_sum/len(x)}


def fit_transcoder(config, width, coefficient, x_train, y_train, x_validation, y_validation, folder, identity):
    """No g/Q arguments: the loss cannot silently supervise Boolean labels."""
    t = config["training"]
    seed_everything(config["student_seed"], True)
    model = SparseTranscoder(x_train.shape[1], width, y_train.shape[1]).to(x_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    generator = torch.Generator(device="cpu").manual_seed(config["student_seed"])
    start_epoch, best, best_epoch, history = 1, float("inf"), None, []
    last = folder / "last.pt"
    if last.exists():
        state = load_checkpoint(last)
        if state["identity"] != identity:
            raise ValueError("Resume identity differs from frozen protocol/teacher")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["shuffle_rng"])
        start_epoch, best, best_epoch, history = state["epoch"]+1, state["best_mse"], state["best_epoch"], state["history"]
    for epoch in range(start_epoch, t["epochs"]+1):
        started = time.monotonic()
        lr = epoch_learning_rate(t, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(x_train.device)
        mse_total, l1_total = 0., 0.
        for start in range(0, len(order), t["batch_size"]):
            idx = order[start:start+t["batch_size"]]
            x, y = x_train[idx], y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            z = model.latents(x)
            loss, mse, l1 = imitation_loss(model.decoder(z), y, z, coefficient)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite transcoder objective")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Non-finite transcoder gradient")
            optimizer.step(); model.normalize_decoder()
            mse_total += mse.item()*len(x); l1_total += l1.item()*len(x)
        model.eval()
        val = validation(model, x_validation, y_validation, config["evaluation_batch_size"])
        if not np.isfinite(val["full_output_mse"]):
            raise RuntimeError("Non-finite validation fidelity")
        improved = val["full_output_mse"] < best
        if improved:
            best, best_epoch = val["full_output_mse"], epoch
        row = {"epoch": epoch, "learning_rate": lr, "train_mse": mse_total/len(x_train),
               "train_l1_sum": l1_total/len(x_train), "validation": val, "best_epoch": best_epoch,
               "seconds": time.monotonic()-started}
        history.append(row)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "width": width, "lambda": coefficient, "identity": identity, "best_mse": best, "best_epoch": best_epoch,
                 "history": history, "shuffle_rng": generator.get_state(), "config": config}
        if improved:
            save_checkpoint(folder / "best.pt", state)
        save_checkpoint(last, state)
        write_json(folder / "history.json", history)
        if epoch == 1 or epoch % 25 == 0 or epoch == t["epochs"]:
            print(json.dumps({"width": width, "lambda": coefficient, **row}), flush=True)
    model.load_state_dict(load_checkpoint(folder / "best.pt")["model"])
    model.eval()
    metrics = validation(model, x_validation, y_validation, config["evaluation_batch_size"])
    write_json(folder / "validation.json", {"width": width, "lambda": coefficient, "epoch": best_epoch, **metrics})
    return model

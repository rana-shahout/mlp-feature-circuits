"""Audit signed gate-prediction error covariance using frozen discovery summaries."""
import numpy as np
import torch
from src.runtime import select_device,write_json
from .run import OUT,RUN


def main():
    device=select_device('cuda')
    with np.load(OUT/'heldout_exact.npz') as f:a=torch.tensor(f['a'],device=device);h=torch.tensor(f['h'],device=device)
    with np.load(RUN/'fixed_gate_parameters.npz') as f:p={k:torch.tensor(f[k],device=device) for k in f.files}
    with np.load(OUT/'native_coordinates.npz') as f:d=torch.tensor(f['D2_eff'],device=device)
    units=[224,192,30];result={}
    for label,pred in [('invariant',torch.relu(a[...,[7,8]]@p['pair_C1'].T+p['invariant_offset'])),('context2',torch.relu(a@p['retained_C1'].T+p['offset']))]:
        error=((pred[:,:,units]-h[:,:,units])*d[4,units]).reshape(-1,3);center=error-error.mean(0);cov=center.T@center/len(center)
        result[label]={'neurons':units,'per_neuron_signed_g4_error_rmse':error.square().mean(0).sqrt().cpu().tolist(),'signed_error_covariance':cov.cpu().tolist(),'group_signed_g4_error_rmse':float(error.sum(1).square().mean().sqrt())}
    write_json(OUT/'gate_prediction_error_covariance.json',result);print(result)

if __name__=='__main__':main()

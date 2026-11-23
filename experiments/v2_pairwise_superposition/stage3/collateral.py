"""Saved signed g7 collateral cancellation, not a second case study."""
import numpy as np
from src.runtime import write_json
from .run import OUT


def main():
    with np.load(OUT/'heldout_exact.npz') as f:
        full=f['native_value'][:,7];group=f['native_contributions'][:,7,[224,192,30]].sum(-1);remaining=full-group
    result={'output':7,'full_teacher_mean_interaction':float(full.mean()),'full_teacher_mean_absolute_interaction':float(np.abs(full).mean()),'selected_trio_mean_interaction':float(group.mean()),'remaining_original_neurons_mean_interaction':float(remaining.mean()),'interpretation':'Measured compensation in the actual teacher, not an assumption of exact Boolean invariance.'}
    write_json(OUT/'collateral_cancellation.json',result);print(result)

if __name__=='__main__':main()

"""Required matrix-reload prediction check; no feature decisions or new populations."""
import sys,shutil
from pathlib import Path
import numpy as np
import torch
from .run import read,write,tensor,arr,bits,sha,OLD,LABELS
run=Path(sys.argv[1]);p=np.load(run/'audit_predictions.npz');names=['W1','b1','W2','b2','D2','d2','C2','b2_feature','R2'];A={k:tensor(np.load(run/'matrices'/f'{k}.npy')) for k in names};mx={k:0. for k in ['native','feature_only','residual']}
with torch.no_grad():
 for start in range(0,len(p['keys']),512):
  sl=slice(start,start+512);h=(tensor(bits(p['keys'][sl]))@A['W1'].T+A['b1']).relu();f=h@A['D2'].T+A['d2'];values=dict(native=h@A['W2'].T+A['b2'],feature_only=f@A['C2'].T+A['b2_feature'],residual=h@A['R2'].T)
  for k,v in values.items():
   err=float(np.max(abs(arr(v)-p[k][sl])));mx[k]=max(mx[k],err);np.testing.assert_allclose(arr(v),p[k][sl],atol=1e-10,rtol=1e-10)
lock=read(run/'feature_lock.json');assert all(sha(run/f)==h for f,h in lock.items())
# Preserve original source provenance, selection statistics, and input manifests alongside the run.
for name in ['protocol.json','context_manifest.json','manifest_lock.json','readout_lock.json','fit_statistics.json','selected_models.json','solver_diagnostics.json','alpha_grid.csv']:
 shutil.copy2(OLD/name,run/'source_snapshot'/('v05_'+name))
write(run/'reload_prediction_checks.json',dict(CUDA=True,rows=len(p['keys']),max_abs_by_prediction=mx,passed=True,amplified_readout_max_by_output={label:float(abs(p['amplified_readout'][:,j]).max()) for j,label in enumerate(LABELS)}))
write(run/'artifact_hashes.json',{str(x.relative_to(run)):sha(x) for x in run.rglob('*') if x.is_file() and x.name!='artifact_hashes.json'})
shutil.copy2(Path(__file__),run/'source_snapshot'/Path(__file__).name)
print('RELOADED PREDICTIONS PASS',mx)

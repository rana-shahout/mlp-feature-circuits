import inspect
import numpy as np
import pytest
import torch
from .run import calibration, factor, metrics, tensor, arr

def test_exact_calibration(tmp_path):
 r=calibration(tmp_path)
 assert len(r['A']['accepted_ids'])==3
 assert len(r['B']['accepted_ids'])==0
 np.testing.assert_allclose(r['B']['MSE'],.125)
 np.testing.assert_allclose(r['A_incomplete']['matrices']['R2'],[[0,0,-2]],atol=1e-12)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_status_priority_and_zero_change_edges():
 h=tensor([[0],[0],[1],[1]]);z=tensor([[0],[0],[1],[1]]);M=tensor([[1]]);d=tensor([0]);stats=(tensor([.25]),tensor([2**-.5]));edges=np.array([[0,2],[1,3],[0,1]])
 rows,_=metrics(h,z,M,d,stats,edges)
 assert rows[0]['status']=='UNASSESSED'
 assert rows[0]['NMSE']==0
 rows,_=metrics(h,z,M,tensor([10]),stats,edges)
 assert rows[0]['status']=='REJECT_THIS_READOUT' and 'NMSE' in rows[0]['failed']
 # All true-zero-change edges must still contribute error to numerator.
 h=tensor(np.tile([[0],[1],[1],[2]],(40,1)));z=tensor(np.tile([[0],[0],[1],[1]],(40,1)))
 edges=np.array([[4*k+i,4*k+j] for k in range(40) for i,j in [(0,2),(1,3),(0,1)]])
 rows,_=metrics(h,z,M,tensor([0]),stats,edges)
 assert rows[0]['finite_change_RRMSE']>0

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_factorization_decoder_independence_and_identities():
 assert list(inspect.signature(factor).parameters)==['D','d','W','b']
 for D in [tensor(np.eye(4)),tensor([[1,0,0,0],[2,0,0,0]]),tensor(np.empty((0,4)))]:
  W=tensor([[1,2,3,4]]);b=tensor([.5]);d=torch.zeros(len(D),device='cuda',dtype=torch.float64)
  decoder=tensor([[1,2,3]])
  result=factor(D,d,W,b);decoder*=1000;result2=factor(D,d,W,b)
  for x,y in zip(result[:4],result2[:4]):torch.testing.assert_close(x,y)
  J,C,R,off,diag=result;h=tensor([[.2,.3,.1,2.]])
  torch.testing.assert_close((h@D.T+d)@C.T+off+h@R.T,h@W.T+b)
  torch.testing.assert_close(D@J@D,D);torch.testing.assert_close(J@D@J,J)
  if len(D)==4:assert R.norm()<1e-12

"""Supplement existing AND reports with the first locked legacy nuisance edge."""
import sys,shutil
from pathlib import Path
import numpy as np
import torch
from .core import *
from .run import npz
from ..workflow_v04.report import table,vec

def main():
 run=Path(sys.argv[1]);verify(run,'manifest_lock.json');verify(run,'readout_lock.json');man=read(run/'context_manifest.json');bank=man['legacy_contexts']['7,8'];edges=bank['nuisance_edges']
 if not edges:print('No legacy AND nuisance edge');return
 b0,bit,b1=edges[0];pair=(7,8);cache=dict(np.load(run/'legacy_activations.npz'));pred=dict(np.load(run/'legacy_predictions.npz'));lookup={int(k):i for i,k in enumerate(cache['keys'])};kk=keys(corners([b0,b1],pair));ix=np.array([[lookup[int(k)] for k in row] for row in kk]);aa=tensor(corners([b0,b1],pair));cp=torch.load(run/'readout.pt',map_location='cuda',weights_only=False);panels=read(run/'display_panels.json')
 for m in [173,266]:
  old=dict(np.load(run/f'feature_{m}_worked.npz'));C=tensor(old['C1']);b=tensor(old['b1']);K=tensor(old['K']);pre=aa@C.T+b;tcpre=aa@K+float(old['encoder_bias']);h=tensor(cache['h'][ix]);v=h*cp['M'][m];zh=v.sum(-1)+cp['d'][m];true=tensor(cache['z'][ix,m]);torch.testing.assert_close(pre.relu(),h,atol=1e-9,rtol=1e-8);torch.testing.assert_close(tcpre.relu(),true,atol=1e-9,rtol=1e-8);np.testing.assert_allclose(numpy(zh),pred['native'][ix,m],atol=1e-9,rtol=1e-8);torch.testing.assert_close(pre[1,0]-pre[0,0],C[:,bit],atol=1e-9,rtol=1e-8)
  text=f'\n## Secondary legacy nuisance edge from the locked manifest\n\nThe primary AND bank has no nuisance edge. This supplemental worked edge is the **first already ordered legacy edge** in the pre-readout-lock manifest: background{b0}→{b1}, adding a{bit}. It reuses already scored legacy inputs; it is not new primary confirmation, and one edge cannot meet the32-change support requirement. No error ranking or new input selection was used. Readout membership of the two four-corner sets: '+str([[man['membership'][str(int(k))] for k in row] for row in kk])+'.\n\n'
  ids=panels[str(m)];other=[r for r in range(256) if r not in ids]
  for i,bg in enumerate([b0,b1]):
   rows=[]
   for r in ids[:6]:rows.append([r,f'{float(pre[i,0,r]):.7f}',f'{float(C[r,pair[0]]):.7f}, {float(C[r,pair[1]]):.7f}',vec(numpy(pre[i,:,r])),vec(numpy(v[i,:,r]))])
   rows += [['other displayed','—','—','—',vec(numpy(v[i,:,ids[6:]].sum(-1)))],['undisplayed remainder','—','—','—',vec(numpy(v[i,:,other].sum(-1)))],['true fitted d','—','—','—',vec(np.repeat(float(cp['d'][m]),4))]]
   text+=f'Background{bg}: TC q={float(tcpre[i,0]):.8f}, preactivation={vec(numpy(tcpre[i]))}, actual z={vec(numpy(true[i]))}, zhat={vec(numpy(zh[i]))}.\n\n'+table(['native row','q (includes true b1)','u,v','native pre corners','M-weighted latent corners'],rows)+'\n\n'
  text+=f'True nuisance latent change={vec(numpy(true[1]-true[0]))}; readout change={vec(numpy(zh[1]-zh[0]))}. Native q shift is C1[:,{bit}], TC q shift={float(K[bit]):+.8f}. Full signed contribution changes and unchanged true biases are retained in feature_{m}_legacy_matched.npz. This local accounting does not establish nuisance invariance or general response compatibility.\n'
  path=run/f'feature_{m}.md';shutil.copy2(path,run/f'feature_{m}.prematched.md');path.write_text(path.read_text()+text);npz(run/f'feature_{m}_legacy_matched.npz',keys=kk,background_keys=np.array([b0,b1]),nuisance=np.array(bit),native_pre=pre,native_h=h,latent_contributions=v,TC_pre=tcpre,z=true,zhat=zh,C1=C,b1=b,K=K,encoder_bias=old['encoder_bias'],M=cp['M'][m],d=cp['d'][m],panel=np.array(ids))
 shutil.copy2(__file__,run/'source_snapshot'/Path(__file__).name);c=read(run/'complete.json');c['feature_reports']={str(m):sha(run/f'feature_{m}.md') for m in FEATURES};c['matched_report_source_sha256']=sha(__file__);write(run/'complete.json',c);print('Legacy matched report supplemented; locked M/d and original manifests unchanged')
if __name__=='__main__':main()

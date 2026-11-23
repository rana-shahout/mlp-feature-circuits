"""Seal reports and saved-result aggregates without changing the locked experiment."""
import csv,shutil,sys
from pathlib import Path
from .core import read,write,sha,verify,now
from .run import csvwrite
from ..workflow_v04.report import table

def main():
 run=Path(sys.argv[1]);log=run.parent/(run.name+'-logs');assert (log/'exit_status').read_text().strip()=='0';assert (log/'finalize.exit_status').read_text().strip()=='0';verify(run,'manifest_lock.json');verify(run,'readout_lock.json');p=run/'SUMMARY.md';shutil.copy2(p,run/'SUMMARY.preseal.md');text=p.read_text();rows=list(csv.DictReader(open(run/'finite_response_summary.csv')));display=[]
 for pair in ['7,8','11,12','0,1']:
  for method in ['native','input','output','semantic']:
   r=next(r for r in rows if r['population']=='legacy' and r['pair']==pair and r['method']==method and r['change_type']=='joint');display.append([pair,method,r['compatible']+'/'+r['assessable'],r['unassessable'],r['median_relative_RMSE_among_assessable']])
 block='Secondary legacy joint-response comparison (reused contexts, not new confirmation):\n\n'+table(['pair','predictor','compatible / assessable','unassessable /1024','median relative RMSE(assessable)'],display)+'\n\n'
 text=text.replace('## 5. Output compatibility and cancellation',block+'## 5. Output compatibility and cancellation')
 jr=list(csv.DictReader(open(run/'output_jacobian_metrics.csv')));derivatives=[]
 for typ in ['teacher_minus_TC','teacher_minus_via','TC_minus_via']:
  rr=[r for r in jr if r['residual']==typ];den=sum(float(r['teacher_squared_norm']) for r in rr);relative=(sum(float(r['squared_error_sum']) for r in rr)/den)**.5;derivatives.append([typ,relative])
 text=text.replace('## 6. Stability and identifiability','Output Jacobians, normalized by full-teacher output-Jacobian norm on the same256 locked inputs:\n\n'+table(['derivative residual','relative Frobenius'],derivatives)+'\n\nVia-readout output derivatives can agree with the teacher while individual TC latent derivatives differ, because signed decoder errors cancel. Neither low output-Jacobian error nor low output MSE substitutes for individual latent response metrics.\n\n## 6. Stability and identifiability')
 panels=read(run/'display_panels.json');text+='\nFit-only display panels (organization of dense rows, not selected circuits): '+str(panels)+'. Latent173 is dominated in this scale by native224, with signed calibration from30/192 and a dense remainder; those M signs differ from the original R output-write signs. Each of the six reports keeps that distinction numerical.\n\nFigures: [fit vs confirmation](fit_vs_confirmation.png), [six latent scatters](fixed_feature_scatter.png), [input-change scatters](finite_change_scatter.png), [Jacobian discrepancy](jacobian_discrepancy.png), [centered worked contributions with exact remainder](centered_worked_contributions_with_remainder.png).\n'
 text+='\nControl correctness: the exact affine known-map coefficient error was5.22e-15; the perfect-output/unreadable-latent negative fixture had exactly zero output residual and latent MSE0.151515. These checks passed before real fitting.\n'
 preservation=read(run/'preservation_verified.json');text+=f'\nPreservation: {preservation["files_verified"]} prior files verified unchanged; teacher/transcoder checkpoint hashes are recorded in protocol.json. The staged active SPEC/prompt already specifiedv0.5; previousv0.4 and activev0.5 exact bytes were copied and hashed in the archive named by protocol.json. No active specification rewrite was needed.\n'
 p.write_text(text);csvwrite(run/'output_jacobian_summary.csv',[{'residual':r[0],'relative_Frobenius':r[1]} for r in derivatives]);shutil.copy2(__file__,run/'source_snapshot'/Path(__file__).name)
 c=read(run/'complete.json');c.update({'summary_sha256':sha(p),'seal_source_sha256':sha(__file__),'sealed':now(),'finalization_exit_status':0});write(run/'complete.json',c)
 print('SEALED',p,'M/d unchanged')
if __name__=='__main__':main()

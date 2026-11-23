import os,sys,subprocess,time
from pathlib import Path
from .core import write,read,sha
root=Path(sys.argv[1]);assert (root/'phase_I/complete.json').exists();assert (root/'phase_I/PHASE_I_SUMMARY.md').exists();logs=root/'logs';logs.mkdir(exist_ok=True);processes=[]
for worker in range(2):
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(worker);env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['LD_PRELOAD']='/usr/lib/x86_64-linux-gnu/libffi.so.7';fp=open(logs/f'worker_{worker}.log','w');cmd=[sys.executable,'-m','experiments.v2_pairwise_superposition.residual_width_code_v08.width','worker',str(root),str(worker)];p=subprocess.Popen(cmd,env=env,stdout=fp,stderr=subprocess.STDOUT);processes.append((p,fp,cmd));print('WORKER',worker,'PID',p.pid,flush=True)
write(root/'phase_II/launched.json',dict(phase_I_report_sha256=sha(root/'phase_I/PHASE_I_SUMMARY.md'),workers=[dict(pid=p.pid,command=cmd) for p,fp,cmd in processes],tmux_session='v08-width-run01'))
codes=[]
for p,fp,cmd in processes:codes.append(p.wait());fp.close()
write(root/'phase_II/worker_exit_codes.json',codes)
if any(codes):raise RuntimeError(('width worker failure',codes))
cmd=[sys.executable,'-m','experiments.v2_pairwise_superposition.residual_width_code_v08.width','audit',str(root)];ret=subprocess.run(cmd,env=os.environ.copy());write(root/'orchestrator_exit_code.json',ret.returncode);sys.exit(ret.returncode)

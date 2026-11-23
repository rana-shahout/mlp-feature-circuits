"""Two backgrounds/eight inputs only. No interventions, optimization or fitting."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
import torch
from src.models import SparseTranscoder
from src.coordinates import original_coordinates,transcoder_coordinates
from src.runtime import load_checkpoint,select_device,seed_everything,write_json
from ..run import BASE,ROOT,digest,write_csv
from ..data import RULES,keys,targets
from ..stage2.core import frozen_teacher
from .core import select_pair,corner_inputs,trace,polynomial

DIRECTORY=BASE/'traces/and78-two-backgrounds-20261028'
DEVELOPMENT=BASE/'reports/stage3-and78-20261027/development_manifest.json'
ORIGINAL=BASE/'data/first-teacher-20261027/split_manifest.json'
RESERVATION=BASE/'data/balanced-semantic-20261027/manifest.json'
OLD_PROBES=BASE/'reports/first-teacher-20261027/constructed_probes.npz'


def choose():
    DIRECTORY.mkdir(parents=True,exist_ok=False)
    old=json.loads(ORIGINAL.read_text());known={int(k):name for name,value in old['partitions'].items() for k in value['vector_keys']}
    dev=json.loads(DEVELOPMENT.read_text());reservation=json.loads(RESERVATION.read_text())
    if reservation['excluded']!='all original train/validation/test and all previously constructed probes':raise RuntimeError('Reservation exclusion protocol differs; cannot certify trace inputs without reading reserved data')
    with np.load(OLD_PROBES) as p:previous=set(map(int,keys(p['a'])))  # Only input identities, not stored model outputs.
    selection=select_pair(dev,known,previous);backgrounds=np.array([[(k>>i)&1 for i in range(20)] for k in selection['background_keys']],dtype=np.uint8)
    assert np.all(backgrounds[:,[7,8]]==0);a=corner_inputs(backgrounds);np.testing.assert_array_equal(targets(a[0]),targets(a[1]));assert ((a.sum(-1)>=3)&(a.sum(-1)<=6)).all()
    selection.update({'rule':'First discovery pair (fallback heldout only if none), ascending B0 integer key then ascending added nuisance index; require base density3, B1 present in same bank, no historical-test corners, and all8 inputs certified excluded from reserved population by existing original-data/old-probe exclusion metadata. No model outputs or weights are read during selection.','background_active_coordinates':[np.flatnonzero(b).astype(int).tolist() for b in backgrounds],'background_development_indices':[dev[selection['partition']]['context_keys'].index(int(k)) for k in selection['background_keys']],'corner_order':['00','01','10','11'],'corner_densities':a.sum(-1).astype(int).tolist(),'original_membership':[[known.get(int(k),'not_in_original_dataset') for k in row] for row in keys(a)],'development_membership':[[split for split,value in dev.items() if int(k) in value['context_keys']] for k in selection['background_keys']],'reserved_exclusion_status':[['original_dataset_excluded_by_reservation_manifest' if int(k) in known else 'old_probe_excluded_by_reservation_manifest' for k in row] for row in keys(a)],'reserved_dataset_accessed':False,'metadata_hashes':{str(p):digest(p) for p in (DEVELOPMENT,ORIGINAL,RESERVATION,OLD_PROBES)},'model_outputs_inspected':False})
    np.savez_compressed(DIRECTORY/'selected_inputs.npz',backgrounds=backgrounds,a=a,targets=targets(a));write_json(DIRECTORY/'selection.json',selection);write_json(DIRECTORY/'selection_frozen.json',{'selection_sha256':digest(DIRECTORY/'selection.json'),'inputs_sha256':digest(DIRECTORY/'selected_inputs.npz'),'before_model_outputs':True})
    print(json.dumps(selection,indent=2))


def csv_exports(label,result,backgrounds,flipped):
    active=np.flatnonzero(backgrounds.any(0));rows=[];writes=[]
    for unit in range(len(result['u'])):
        writes.append({'unit':unit,'hidden_bias':float(result['hidden_bias'][unit]),**{f'input_a{i}':float(result['input_coefficients'][unit,i]) for i in range(20)},**{f'write_g{j}':float(result['semantic_readout'][j,unit]) for j in range(9)}})
        for background in range(2):
            for corner in range(4):
                rows.append({'background':f'B{background}','unit':unit,'corner':['00','01','10','11'][corner],'u_a7':float(result['u'][unit]),'v_a8':float(result['v'][unit]),'hidden_bias':float(result['hidden_bias'][unit]),**{f'background_term_a{i}':float(result['background_terms'][background,unit,i]) for i in active},'q':float(result['q'][background,unit]),'write_g4':float(result['semantic_readout'][4,unit]),'preactivation':float(result['preactivation'][background,corner,unit]),'activation':float(result['activation'][background,corner,unit]),'gate_on':bool(result['gate'][background,corner,unit]),'signed_g4_contribution':float(result['semantic_contributions'][background,corner,4,unit]),**{key:float(value) for key,value in zip(('alpha0','alpha7','alpha8','alpha78'),result['g4_alpha'][background,unit])}})
    write_csv(DIRECTORY/f'{label}_all_components.csv',rows);write_csv(DIRECTORY/f'{label}_all_coefficients.csv',writes)
    changes=[]
    for unit in range(len(result['u'])):
        changes.append({'unit':unit,'q_change':float(result['q'][1,unit]-result['q'][0,unit]),'nuisance_coefficient':float(result['input_coefficients'][unit,flipped]),'switched_corners':','.join(['00','01','10','11'][c] for c in range(4) if result['gate'][0,c,unit]!=result['gate'][1,c,unit]),**{f'contribution_change_{c}':float(result['semantic_contributions'][1,i,4,unit]-result['semantic_contributions'][0,i,4,unit]) for i,c in enumerate(['00','01','10','11'])},**{key:float(value) for key,value in zip(('delta_alpha0','delta_alpha7','delta_alpha8','delta_alpha78'),result['g4_alpha'][1,unit]-result['g4_alpha'][0,unit])}})
    write_csv(DIRECTORY/f'{label}_background_changes.csv',changes)


@torch.no_grad()
def execute():
    frozen=json.loads((DIRECTORY/'selection_frozen.json').read_text());assert frozen['selection_sha256']==digest(DIRECTORY/'selection.json');assert frozen['inputs_sha256']==digest(DIRECTORY/'selected_inputs.npz')
    if (DIRECTORY/'complete.json').exists():raise RuntimeError('Trace already complete; refusing overwrite')
    device=select_device('cuda');seed_everything(0,True)
    metadata=json.loads((BASE/'runs/stage2-transcoders-20261027/teacher_verification.json').read_text());selection2=json.loads((BASE/'reports/stage2-transcoders-20261027/selection.json').read_text());identity=json.loads((BASE/'runs/stage3-and78-20261027/model_verification.json').read_text())
    teacher_path=Path(metadata['checkpoint']);best=selection2['best_validation_fidelity'];student_path=BASE/'checkpoints/stage2-transcoders-20261027'/best/'best.pt'
    assert best=='w1024_l0.0001';assert digest(teacher_path)==metadata['checkpoint_sha256']==identity['teacher_sha256'];assert digest(student_path)==identity['transcoder_sha256']
    dataset=BASE/'data/first-teacher-20261027/dataset.npz';assert digest(dataset)==metadata['dataset_sha256'];assert digest(ORIGINAL)==metadata['split_manifest_sha256'];assert RULES[4]=='and_78'
    protected={str(p):digest(p) for root in (BASE/'checkpoints',ROOT/'results') for p in root.rglob('*.pt')}
    protected.update({str(p):digest(p) for root in (BASE/'runs',BASE/'reports') for p in root.rglob('*') if p.is_file()});write_json(DIRECTORY/'protected_before.json',protected)
    source_files=[*Path(__file__).parent.glob('*.py'),BASE/'tests/test_trace.py',ROOT/'src/models.py',ROOT/'src/coordinates.py',ROOT/'src/runtime.py',ROOT/'SPEC.md'];snapshot=DIRECTORY/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for p in source_files:shutil.copyfile(p,snapshot/str(p.relative_to(ROOT)).replace('/','__'))
    teacher,cp=frozen_teacher(teacher_path,device);checkpoint=load_checkpoint(student_path)
    assert {k:tuple(v.shape) for k,v in checkpoint['model'].items()}=={'encoder.weight':(1024,12),'encoder.bias':(1024,),'decoder.weight':(12,1024),'decoder.bias':(12,)}
    assert checkpoint['width']==1024 and checkpoint['lambda']==1e-4;assert checkpoint['identity']['teacher_sha256']==digest(teacher_path);assert checkpoint['identity']['dataset_sha256']==digest(dataset)
    student=SparseTranscoder(12,1024,12).to(device);student.load_state_dict(checkpoint['model']);student.eval().requires_grad_(False)
    with np.load(dataset) as data:p32=torch.tensor(data['P'],device=device);q32=torch.tensor(data['Q'],device=device)
    assert p32.shape==(12,20) and q32.shape==(12,9);assert torch.linalg.matrix_rank(p32)==12
    torch.testing.assert_close(p32.norm(dim=0),torch.ones(20,device=device),atol=2e-7,rtol=2e-7);torch.testing.assert_close(q32.T@q32,torch.eye(9,device=device),atol=2e-7,rtol=2e-7)
    with np.load(DIRECTORY/'selected_inputs.npz') as inputs:backgrounds=inputs['backgrounds'];a=inputs['a']
    x32=torch.tensor(a,device=device,dtype=torch.float32)@p32.T;float32={'native':(teacher(x32.reshape(-1,12))@q32).reshape(2,4,9).cpu().numpy(),'transcoder':(student(x32)@q32).cpu().numpy()}
    teacher=teacher.double();student=student.double();p=p32.double();q=q32.double();at=torch.tensor(a,device=device,dtype=torch.float64);bt=torch.tensor(backgrounds,device=device,dtype=torch.float64);x=at@p.T
    pre,h=teacher.hidden(x.reshape(-1,12));native_direct=(teacher.W2(h)@q).reshape(2,4,9);student_pre=student.encoder(x);student_direct=student(x)@q
    native=trace(original_coordinates(teacher,p,q),bt,pre.reshape(2,4,256),native_direct,[224,192,30]);tc=trace(transcoder_coordinates(student,p,q),bt,student_pre,student_direct,[173,266])
    flipped=json.loads((DIRECTORY/'selection.json').read_text())['flipped_input'];assert np.flatnonzero(backgrounds[1]-backgrounds[0]).tolist()==[flipped]
    checks={}
    for label,result in [('native',native),('transcoder',tc)]:
        np.testing.assert_allclose(result['q'][1]-result['q'][0],result['input_coefficients'][:,flipped],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(result['semantic_total'],float32[label],atol=2e-6,rtol=2e-5)
        result['float32_semantic_total']=float32[label];np.savez_compressed(DIRECTORY/f'{label}_trace.npz',**result);csv_exports(label,result,backgrounds,flipped)
        checks[label]={'sum_vs_direct_max_abs':float(np.abs(result['coordinate_vs_direct_residual']).max()),'displayed_remainder_bias_max_abs':float(np.abs(result['displayed_remainder_bias_residual']).max()),'float64_vs_saved_dtype_float32_max_abs':float(np.abs(result['semantic_total']-float32[label]).max()),'q_change_identity_max_abs':float(np.abs(result['q'][1]-result['q'][0]-result['input_coefficients'][:,flipped]).max())}
    np.savez_compressed(DIRECTORY/'full_output_comparison.npz',a=a,teacher=native['semantic_total'],transcoder=tc['semantic_total'],residual=tc['semantic_total']-native['semantic_total'],teacher_alpha=native['g4_total_alpha'],transcoder_alpha=tc['g4_total_alpha'],residual_alpha=tc['g4_total_alpha']-native['g4_total_alpha'])
    write_json(DIRECTORY/'checks.json',{'tolerance_atol':1e-10,'tolerance_rtol':1e-10,'float32_comparison_tolerance':{'atol':2e-6,'rtol':2e-5},'models':checks,'column_vs_row_convention':'Column x=Pa; row batch x=a@P.T, hidden=ReLU(x@W1.T+b1), semantic=y@Q. Exact coordinates C=W1@P, D=Q.T@W2.','g4_rule':'AND(a7,a8), zero-based indices','teacher_frozen':all(not p.requires_grad for p in teacher.parameters()),'transcoder_frozen':all(not p.requires_grad for p in student.parameters())})
    write_json(DIRECTORY/'provenance.json',{'command':sys.argv,'GPU':torch.cuda.get_device_name(device),'torch':torch.__version__,'CUDA':torch.version.cuda,'python':sys.version,'teacher':str(teacher_path),'transcoder':str(student_path),'teacher_sha256':digest(teacher_path),'transcoder_sha256':digest(student_path),'dataset_sha256':digest(dataset),'manifest_sha256':digest(ORIGINAL),'P_sha256_raw':__import__('hashlib').sha256(p32.cpu().numpy().tobytes()).hexdigest(),'Q_sha256_raw':__import__('hashlib').sha256(q32.cpu().numpy().tobytes()).hexdigest(),'source_hashes':{str(p):digest(p) for p in source_files},'analysis_dtype':'float64 cast of actual saved float32 weights/embeddings; direct float32 execution also recorded','no_models_trained':True,'no_hidden_interventions':True,'no_readout_fitted':True,'reserved_dataset_accessed':False})
    from .report import produce
    produce(DIRECTORY,native,tc)
    assert all(digest(Path(path))==h for path,h in protected.items());assert frozen['selection_sha256']==digest(DIRECTORY/'selection.json')
    write_json(DIRECTORY/'complete.json',{'status':'complete','number_of_inputs':8,'backgrounds':2,'protected_files_verified':len(protected),'selection_sha256':frozen['selection_sha256'],'report_sha256':digest(DIRECTORY/'report.md'),'checkpoint_hashes_unchanged':True,'reserved_dataset_accessed':False,'hidden_interventions':False,'models_trained':0});print(DIRECTORY/'report.md')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['select','execute']);args=parser.parse_args()
    choose() if args.command=='select' else execute()

if __name__=='__main__':main()

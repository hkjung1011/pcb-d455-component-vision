"""Package reproducible local evidence and portable inference weights, not raw datasets."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT.parents[1] / 'outputs'
DEST = OUTPUT / 'PCB_D455_R03'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def cp(source, relative=None):
    source=Path(source)
    if not source.is_file():raise FileNotFoundError(source)
    target=DEST/(relative or source.relative_to(ROOT))
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def main():
    from make_release_summary import verified_aggregate, make_registry
    aggregate=verified_aggregate()
    if read(ROOT/'selected_models.json')!=make_registry(aggregate):
        raise SystemExit('Inference registry differs from frozen validation selection; rebuild the release summary')
    archive=OUTPUT/'PCB_D455_R03.zip'
    if archive.exists():raise SystemExit('Existing archive preserved')
    if DEST.exists() and any(DEST.iterdir()):
        raise SystemExit('Preserve existing release: choose a new revision path')
    DEST.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/'scripts').glob('*.py'):cp(source)
    for source in ROOT.glob('*.ps1'):cp(source)
    for source in (ROOT/'configs').glob('*.yaml'):cp(source)
    for source in (ROOT/'models').glob('*.pt'):cp(source)
    for source in (ROOT/'src').rglob('*.py'):cp(source)
    for name in ['README.md','INFERENCE.md','RELEASE_NOTES_DRAFT.md','claude_review_application.md','selected_models.json','model_registry.json','requirements-recorded.txt']:
        cp(ROOT/name)
    for source in (ROOT/'annotation').rglob('*'):
        if source.is_file():cp(source)
    for source in (ROOT/'reports').rglob('*'):
        if not source.is_file():continue
        relative=source.relative_to(ROOT/'reports')
        # Exclude obsolete candidates, bulky console progress and source-photo QA.
        if relative.parts[0] in ['contact_sheets','qualitative_commons','inference_pi_smoke'] or source.name.endswith('_console.log'):continue
        if relative.parts[0]=='evaluation_suite' and 'preview_' in source.name:continue
        if source.name in ['board_fingerprints.json','board_global_split_audit.json','board_split_conflict_STOP.json']:continue
        if source.suffix.lower() in ['.json','.md','.png','.jpg','.svg','.csv','.log','.txt']:cp(source)
    for mid in aggregate['models']:
        for name in ['architecture.json','environment.json','epoch_updates.json','run_summary.json']:
            cp(ROOT/'runs'/mid/name)
        for name in ['args.yaml','results.csv','results.png','weights/best.pt','weights/last.pt']:
            cp(ROOT/'runs'/mid/'fit'/name)
    # Local manifests intentionally retain original absolute cache paths.
    for folder in ['board_detect_v2','board_segment_v2']:
        for name in ['manifest.json','data.yaml']:
            cp(ROOT/'data'/folder/name)
    for name in ['native_manifest.json','training_manifest.json','component_records.json','component_tile_records.json','component_data_manifest.json','componentdetectdata.yaml','componentdetect_pi_test.yaml','preparation_verification.json','wacv_acquisition.json','README_RESEARCH.md']:
        cp(ROOT/'component_assets'/name)
    for name in ['ACQUISITION_HANDOFF.md','commons_records.json','commons_draft_annotations.json','acquire_commons_holdout.py','select_commons_holdout.py']:
        cp(ROOT/'assets'/name)
    for name in ['acquisition_summary.json','native_geometry_audit.json']:
        cp(ROOT/'assets/iotkits_v1'/name)
    # Local inference is self-contained; training/evaluation still require cached images.
    paths={'original_workspace':str(ROOT),'raw_source_photos_and_archives_included':False,
           'annotations_and_provenance_manifests_included':True,
           'inference_weights_and_code_included':True,
           'training_and_evaluation_require_original_local_cache':True,
           'source_datasets':{'micro':'C:/Users/hkjun/Documents/mcu-vision/data/raw/curated/micro_pcb_images',
             'iotkits':str(ROOT/'assets/iotkits_v1/native'),'wacv':str(ROOT/'component_assets/wacv_original'),
             'commons':str(ROOT/'assets/commons_holdout')},
           'note':'Manifests and historical evidence retain absolute paths and hashes. Moving this folder alone does not move the source datasets.'}
    (DEST/'local_data_paths.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2),encoding='utf-8')
    inventory=[{'path':str(p.relative_to(DEST)).replace('\\','/'),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(DEST.rglob('*')) if p.is_file()]
    (DEST/'SHA256SUMS.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.rglob('*')):
            if p.is_file():z.write(p,Path(DEST.name)/p.relative_to(DEST))
    with zipfile.ZipFile(archive) as z:
        broken=z.testzip()
        if broken:raise ValueError(broken)
    result={'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
            'files':len(inventory)+1,'zip_crc_verified':True,'native_evaluation_status':aggregate['status'],
            'not_d455_or_target4_mask_verified':True}
    (OUTPUT/'PCB_D455_R03_검증기록.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()

"""Sequential R04 validation -> frozen selection -> development-holdout evaluation.

Run only after both training arms finish. Identical completed evaluations are
reused after hash verification. Partial or failed jobs are retained and stop the
run; this program never erases outputs or silently reruns a holdout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ARMS = ['baseline', 'improved']
PIPELINES = ['native', 'dual']
POLICY = {
    'version': 'r04-matched-ab-native-evaluation-v1',
    'selection_rank': ['validation bbox AP50-95 max300 descending', 'validation bbox AP50-95 max100 descending',
                       'validation offline predict/postprocess p50 ascending', 'candidate id ascending'],
    'confidence': 'Single global val micro-F1 confidence on .05:.95 by .05; highest confidence breaks F1 ties; class PR is diagnostic only',
    'sequence': 'All epochs x both pipelines for BOTH arms on val, freeze all selections, then one general/Pi holdout run per selected arm',
    'global_recommendation': 'Compare the arm winners using the same VAL-only rank; never use development-holdout performance',
    'evaluation_scope': 'development_holdout_not_pristine_final',
    'bootstrap': 'General test:200 source-board-group recall resamples; Pi:0 and no one-group CI',
    'device': '0',
    'd455_verified': False,
}
REQUIRED_OUTPUTS = ['metrics.json', 'native_ground_truth.json', 'bbox_predictions.json', 'samples.json',
                    'native_input_inventory.json', 'inference_profiles.json']
VAL_OUTPUTS = ['validation_threshold_curve.json', 'validation_per_class_pr.json']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def log_stage(root, message):
    line = f'[{now()}] {message}'
    print(line, flush=True)
    progress = root.parent/'r04_review'/'progress.log'
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open('a', encoding='utf-8') as handle:
        handle.write(line+'\n')


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'Invalid {name}: {value!r}')
    return float(value)


def rank(candidate):
    metric = candidate['metrics']
    if metric.get('split') != 'val' or metric.get('evaluation_role') != 'validation_selection':
        raise ValueError('Only validation can select a checkpoint/pipeline/arm')
    ap300 = finite(metric['bbox']['ap50_95_max300'], 'validation AP300')
    ap100 = finite(metric['bbox']['ap50_95_max100'], 'validation AP100')
    latency = finite(metric['latency']['offline_predict_and_postprocess_ms_p50'], 'validation latency')
    if not (0 <= ap100 <= 1 and 0 <= ap300 <= 1 and latency >= 0):
        raise ValueError('Validation metric out of range')
    return (-ap300, -ap100, latency, candidate['candidate_id'])


def select_candidate(candidates):
    if not candidates:
        raise ValueError('No validation candidates')
    choice = min(candidates, key=rank)
    conf = finite(choice['metrics']['operating']['confidence'], 'confidence')
    if not 0 <= conf <= 1:
        raise ValueError('Invalid selected confidence')
    return choice


def validate_manifest(manifest):
    if manifest.get('names') != ['resistor', 'capacitor', 'ic', 'connector']:
        raise ValueError('R04 target4 ontology mismatch')
    rows = manifest['records']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate native image IDs')
    seen_groups, seen_hashes = {}, {}
    for row in rows:
        for value, seen in [(row['group_id'], seen_groups), (row['sha256'], seen_hashes)]:
            if value in seen and seen[value] != row['split']:
                raise ValueError('Native source group/hash crosses train, val or holdout')
            seen[value] = row['split']
        for obj in [row]+row['objects']:
            status = ' '.join(str(obj.get(k, '')) for k in ['annotation_status', 'annotation_type', 'label_status', 'mask_origin']).lower()
            if any(token in status for token in ['pseudo', 'unreviewed', 'draft', 'ai_generated', 'sam_generated']):
                raise ValueError('Unreviewed/inferred annotation cannot be evaluation ground truth')
        if row['split'] == 'test' and row.get('cohort') not in ['general_test', 'pi_test']:
            raise ValueError('Unexpected development-holdout cohort')
    for split, cohort in [('val', None), ('test', 'general_test'), ('test', 'pi_test')]:
        selected = [r for r in rows if r['split'] == split and (cohort is None or r.get('cohort') == cohort)]
        if not selected or not any(row['objects'] for row in selected):
            raise ValueError(f'Missing positive native partition {split}/{cohort}')


def load_candidates(root, arm, protocol):
    path = root/'runs'/arm/'candidates.json'
    records = read(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f'candidates.json must be a nonempty list: {path}')
    expected = protocol['save_epoch_candidates']
    epochs = [r['epoch'] for r in records]
    if sorted(epochs) != sorted(expected) or len(set(epochs)) != len(epochs):
        raise ValueError(f'Candidate epochs differ from predeclared set: {arm}: {epochs} vs {expected}')
    candidates = []
    previous_calls = 0
    cap = protocol['max_direct_optimizer_calls_per_arm']
    for row in sorted(records, key=lambda r: r['epoch']):
        epoch, calls = row['epoch'], row['optimizer_calls']
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= protocol['max_epochs_per_arm']:
            raise ValueError('Invalid candidate epoch')
        if isinstance(calls, bool) or not isinstance(calls, int) or not previous_calls < calls <= cap:
            raise ValueError('Candidate direct optimizer counts must strictly increase within budget')
        previous_calls = calls
        weights = Path(row['path'])
        if not weights.is_absolute():
            weights = root/weights
        weights = weights.resolve()
        if not weights.is_file() or sha(weights) != row['sha256']:
            raise ValueError(f'Candidate checkpoint missing or SHA mismatch: {weights}')
        for pipeline in PIPELINES:
            candidates.append({'arm': arm, 'epoch': epoch, 'optimizer_calls': calls, 'pipeline': pipeline,
                               'candidate_id': f'{arm}__epoch_{epoch:02d}__{pipeline}', 'weights': str(weights),
                               'checkpoint_sha256': row['sha256'],
                               'candidate_training_metadata': {k: v for k, v in row.items() if k not in ['path', 'sha256']}})
    return candidates, {'path': str(path), 'sha256': sha(path), 'final_optimizer_calls': previous_calls}


def preflight(root, python):
    training_protocol_path = root/'protocol.json'
    training_protocol = read(training_protocol_path)
    if training_protocol.get('arms') != ARMS or training_protocol.get('inference_candidates') != PIPELINES:
        raise ValueError('Training protocol arms/pipelines differ from this fixed evaluation suite')
    if training_protocol.get('max_epochs_per_arm', 21) > 20:
        raise ValueError('Protocol exceeds authorized per-arm epoch budget')
    manifest_path = root/'data/native_manifest.json'
    manifest = read(manifest_path)
    validate_manifest(manifest)
    manifest_hash = sha(manifest_path)
    if manifest_hash != training_protocol['native_manifest_sha256']:
        raise ValueError('Native manifest differs from frozen training protocol')
    evaluator = root/'scripts/evaluate_r04.py'
    checkpoint_sets, metadata = {}, {}
    for arm in ARMS:
        checkpoint_sets[arm], metadata[arm] = load_candidates(root, arm, training_protocol)
    if len({v['final_optimizer_calls'] for v in metadata.values()}) != 1:
        raise ValueError('Training arms do not have matched final direct optimizer-call budgets')
    return {'policy': POLICY, 'training_protocol_path': str(training_protocol_path),
            'training_protocol_sha256': sha(training_protocol_path), 'manifest_path': str(manifest_path),
            'manifest_sha256': manifest_hash, 'evaluator_path': str(evaluator), 'evaluator_sha256': sha(evaluator),
            'orchestrator_sha256': sha(__file__), 'python_executable': str(python),
            'python_executable_sha256': sha(python), 'candidates_files': metadata, 'candidates': checkpoint_sets,
            'native_class_names': manifest['names'], 'annotation_limitations': manifest.get('annotation_limitations', manifest.get('limitations'))}


def verify_inputs(evidence, candidate):
    expected_files = {evidence['training_protocol_path']: evidence['training_protocol_sha256'],
                      evidence['manifest_path']: evidence['manifest_sha256'],
                      evidence['evaluator_path']: evidence['evaluator_sha256'],
                      candidate['weights']: candidate['checkpoint_sha256']}
    for meta in evidence['candidates_files'].values():
        expected_files[meta['path']] = meta['sha256']
    for path, expected in expected_files.items():
        if sha(path) != expected:
            raise ValueError(f'Input changed since evaluation preflight: {path}')


def binding_for(evidence, protocol_hash, candidate, split, cohort, confidence, bootstrap, frozen_sha):
    return {'protocol_sha256': protocol_hash, 'candidate_id': candidate['candidate_id'], 'arm': candidate['arm'],
            'epoch': candidate['epoch'], 'optimizer_calls': candidate['optimizer_calls'], 'pipeline': candidate['pipeline'],
            'checkpoint_sha256': candidate['checkpoint_sha256'], 'manifest_sha256': evidence['manifest_sha256'],
            'evaluator_sha256': evidence['evaluator_sha256'], 'split': split, 'cohort': cohort,
            'operating_confidence': confidence, 'bootstrap': bootstrap, 'selection_sha256': frozen_sha, 'device': '0'}


def command(evidence, candidate, output, split, cohort, confidence, bootstrap):
    cmd = [evidence['python_executable'], '-B', evidence['evaluator_path'], '--manifest', evidence['manifest_path'],
           '--weights', candidate['weights'], '--output', str(output), '--split', split,
           '--pipeline', candidate['pipeline'], '--device', '0', '--bootstrap', str(bootstrap)]
    if cohort is not None:
        cmd.extend(['--cohort', cohort])
    if confidence is not None:
        cmd.extend(['--operating-conf', format(confidence, '.17g')])
    return cmd


def check_metric(metric, binding):
    for key in ['checkpoint_sha256', 'manifest_sha256', 'evaluator_sha256', 'pipeline', 'split', 'cohort']:
        if metric.get(key) != binding[key]:
            raise ValueError(f'Evaluator metric binding mismatch: {key}')
    expected_role = 'validation_selection' if binding['split'] == 'val' else 'development_holdout_not_pristine_final'
    if metric.get('evaluation_role') != expected_role or metric.get('d455_verified') is not False:
        raise ValueError('Evaluation scope mislabeled')
    if binding['split'] == 'test':
        if abs(finite(metric['operating']['confidence'], 'test confidence')-binding['operating_confidence']) > 1e-12:
            raise ValueError('Holdout threshold differs from frozen validation confidence')
        if metric.get('threshold_source') != 'supplied_frozen_confidence':
            raise ValueError('Holdout threshold must be supplied, never retuned')


def validate_completed_files(record, output):
    if not record.get('output_sha256'):
        raise ValueError('Completed job lacks output hashes')
    required = REQUIRED_OUTPUTS+(VAL_OUTPUTS if record['binding']['split'] == 'val' else [])
    if set(record['output_sha256']) != set(required):
        raise ValueError('Completed job output inventory differs from expected files')
    for name, expected in record['output_sha256'].items():
        if not (output/name).is_file() or sha(output/name) != expected:
            raise ValueError(f'Completed evaluation file changed: {output/name}')


def evaluate_job(root, suite, evidence, protocol_hash, candidate, split, *, cohort=None, confidence=None, bootstrap=0, frozen_sha=None):
    if split == 'test' and (confidence is None or frozen_sha is None):
        raise ValueError('Holdout execution requires already frozen confidence and selection file hash')
    if split == 'test' and sha(suite/'selections_frozen.json') != frozen_sha:
        raise ValueError('Frozen selections changed before holdout evaluation')
    name = f'{candidate["candidate_id"]}__val' if split == 'val' else f'{candidate["arm"]}__{cohort}'
    output, control, log = suite/'evaluations'/name, suite/'jobs'/f'{name}.json', suite/'logs'/f'{name}.log'
    binding = binding_for(evidence, protocol_hash, candidate, split, cohort, confidence, bootstrap, frozen_sha)
    binding_hash = fingerprint(binding)
    verify_inputs(evidence, candidate)
    if control.exists():
        record = read(control)
        if record.get('binding_sha256') != binding_hash:
            raise ValueError(f'Existing job uses different inputs: {control}')
        if record.get('status') != 'completed':
            raise ValueError(f'Incomplete/failed job retained without automatic rerun: {control}')
        validate_completed_files(record, output)
        metric = read(output/'metrics.json')
        check_metric(metric, binding)
        log_stage(root, f'REUSE {name}: identical inputs and output hashes verified')
        return {'candidate_id': candidate['candidate_id'], 'candidate': candidate, 'metrics': metric,
                'metrics_path': str(output/'metrics.json'), 'metrics_sha256': record['output_sha256']['metrics.json'],
                'job_path': str(control), 'job_sha256': sha(control)}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Nonempty untracked evaluation retained: {output}')
    cmd = command(evidence, candidate, output, split, cohort, confidence, bootstrap)
    record = {'status': 'running', 'started_utc': now(), 'binding': binding, 'binding_sha256': binding_hash, 'command': cmd}
    save(control, record)
    log.parent.mkdir(parents=True, exist_ok=True)
    log_stage(root, f'RUN {name} checkpoint={candidate["checkpoint_sha256"][:12]}')
    process = None
    try:
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        with log.open('w', encoding='utf-8') as handle:
            process = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1, env=environment)
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                if line.startswith('EVALUATED '):
                    log_stage(root, f'{name}: {line.strip()}')
            return_code = process.wait()
        if return_code:
            raise RuntimeError(f'Evaluator exited {return_code}; see {log}')
        verify_inputs(evidence, candidate)
        if frozen_sha is not None and sha(suite/'selections_frozen.json') != frozen_sha:
            raise ValueError('Frozen selections changed during holdout evaluation')
        metric = read(output/'metrics.json')
        check_metric(metric, binding)
        required = REQUIRED_OUTPUTS+(VAL_OUTPUTS if split == 'val' else [])
        output_hashes = {name: sha(output/name) for name in required}
        record.update(status='completed', finished_utc=now(), return_code=0, output_sha256=output_hashes, log_sha256=sha(log))
        save(control, record)
        log_stage(root, f'DONE {name}: AP300={metric["bbox"]["ap50_95_max300"]:.6f} AP100={metric["bbox"]["ap50_95_max100"]:.6f}')
        return {'candidate_id': candidate['candidate_id'], 'candidate': candidate, 'metrics': metric,
                'metrics_path': str(output/'metrics.json'), 'metrics_sha256': output_hashes['metrics.json'],
                'job_path': str(control), 'job_sha256': sha(control)}
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        record.update(status='failed', finished_utc=now(), error=f'{type(error).__name__}: {error}')
        save(control, record)
        log_stage(root, f'FAILED {name}: {error}; outputs retained, execution stopped')
        raise


def compact(result):
    metric = result['metrics']
    return {'candidate_id': result['candidate_id'], 'candidate': result['candidate'],
            'metrics_path': result['metrics_path'], 'metrics_sha256': result['metrics_sha256'],
            'job_path': result['job_path'], 'job_sha256': result['job_sha256'],
            **{k: metric.get(k) for k in ['split', 'cohort', 'evaluation_role', 'images', 'groups', 'native_gt_instances',
                'checkpoint_sha256', 'manifest_sha256', 'bbox', 'latency', 'group_bootstrap_recall95', 'group_bootstrap_exclusion_reason']},
            'operating': {k: v for k, v in metric['operating'].items() if k != 'per_image'}}


def frozen_selection_payload(protocol_hash, validation, winners):
    overall = select_candidate(list(winners.values()))
    return {'protocol_sha256': protocol_hash, 'test_results_used_for_selection': False,
            'evaluation_scope': 'development_holdout_not_pristine_final',
            'recommended_arm': overall['candidate']['arm'], 'recommended_candidate_id': overall['candidate_id'],
            'recommendation_basis': 'Validation only; does not establish real-world or statistically significant superiority',
            'rank': POLICY['selection_rank'],
            'all_validation_evidence': {arm: [{'candidate_id': r['candidate_id'], 'metrics_path': r['metrics_path'],
                                              'metrics_sha256': r['metrics_sha256'], 'job_sha256': r['job_sha256']}
                                             for r in rows] for arm, rows in validation.items()},
            'selections': {arm: {'candidate': winner['candidate'], 'checkpoint_sha256': winner['candidate']['checkpoint_sha256'],
                                'manifest_sha256': winner['metrics']['manifest_sha256'],
                                'operating_confidence': winner['metrics']['operating']['confidence'],
                                'validation_metrics_path': winner['metrics_path'], 'validation_metrics_sha256': winner['metrics_sha256'],
                                'validation_bbox': winner['metrics']['bbox'],
                                'validation_latency': winner['metrics']['latency']}
                           for arm, winner in winners.items()}}


def execute(root, python):
    root = root.resolve()
    suite = root/'reports/evaluation_suite'
    log_stage(root, 'R04 evaluation preflight: check both arm candidates, native GT and frozen training protocol')
    evidence = preflight(root, python)
    protocol_hash = fingerprint(evidence)
    protocol_file = suite/'protocol.json'
    if protocol_file.exists():
        stored = read(protocol_file)
        if stored.get('protocol_sha256') != protocol_hash or fingerprint(stored['evidence']) != protocol_hash:
            raise ValueError('Existing evaluation suite belongs to changed protocol/evidence; never mix results')
    else:
        if suite.exists() and any(suite.iterdir()):
            raise ValueError('Nonempty evaluation suite without protocol record')
        save(protocol_file, {'created_utc': now(), 'protocol_sha256': protocol_hash, 'evidence': evidence})
    validation, winners = {}, {}
    log_stage(root, 'PHASE 1: validate all predeclared epoch x pipeline candidates for both arms')
    for arm in ARMS:
        validation[arm] = []
        for candidate in evidence['candidates'][arm]:
            validation[arm].append(evaluate_job(root, suite, evidence, protocol_hash, candidate, 'val'))
        winners[arm] = select_candidate(validation[arm])
        log_stage(root, f'VAL ARM WINNER {arm}: {winners[arm]["candidate_id"]}')
    log_stage(root, 'PHASE 2: freeze all arm selections and VAL-only overall recommendation before any holdout')
    payload = frozen_selection_payload(protocol_hash, validation, winners)
    selection_file = suite/'selections_frozen.json'
    if selection_file.exists():
        locked = read(selection_file)
        if fingerprint({k: v for k, v in locked.items() if k != 'frozen_utc'}) != fingerprint(payload):
            raise ValueError('Frozen selection differs from completed validation; cannot rewrite it')
    else:
        save(selection_file, {'frozen_utc': now(), **payload})
    selection_sha = sha(selection_file)
    aggregate_path = suite/'aggregate.json'
    aggregate = {'status': 'SELECTIONS_FROZEN_DEVELOPMENT_HOLDOUT_PENDING', 'protocol_sha256': protocol_hash,
                 'selection_sha256': selection_sha, 'policy': POLICY, 'recommended_arm': payload['recommended_arm'],
                 'recommended_candidate_id': payload['recommended_candidate_id'],
                 'fresh_final_test_available': False, 'd455_verified': False,
                 'arms': {arm: {'selection': payload['selections'][arm],
                                'validation_candidates': [compact(r) for r in validation[arm]], 'tests': {}} for arm in ARMS}}
    save(aggregate_path, aggregate)
    log_stage(root, f'SELECTION FROZEN sha={selection_sha}; recommended_arm={payload["recommended_arm"]}')
    log_stage(root, 'PHASE 3: one general and one Pi development-holdout evaluation per selected arm; no retuning')
    for arm in ARMS:
        choice = payload['selections'][arm]
        for cohort, bootstrap in [('general_test', 200), ('pi_test', 0)]:
            result = evaluate_job(root, suite, evidence, protocol_hash, choice['candidate'], 'test', cohort=cohort,
                                  confidence=choice['operating_confidence'], bootstrap=bootstrap, frozen_sha=selection_sha)
            aggregate['arms'][arm]['tests'][cohort] = compact(result)
            save(aggregate_path, aggregate)
    aggregate.update(status='COMPLETED_DEVELOPMENT_HOLDOUT_NOT_D455_VALIDATED', completed_utc=now())
    save(aggregate_path, aggregate)
    log_stage(root, f'R04 EVALUATION COMPLETE: {aggregate_path}; recommended by VAL={payload["recommended_arm"]}')
    return aggregate


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def candidate(self, name, ap300=.3, ap100=.2, latency=10, split='val'):
            return {'candidate_id': name, 'candidate': {'arm': name.split('__')[0]},
                    'metrics': {'split': split, 'evaluation_role': 'validation_selection' if split == 'val' else 'development_holdout_not_pristine_final',
                                'bbox': {'ap50_95_max300': ap300, 'ap50_95_max100': ap100},
                                'latency': {'offline_predict_and_postprocess_ms_p50': latency},
                                'operating': {'confidence': .35}}}
        def test_ap300_primary(self):
            self.assertEqual(select_candidate([self.candidate('a', .4, .2), self.candidate('b', .3, .9)])['candidate_id'], 'a')
        def test_ap100_secondary(self):
            self.assertEqual(select_candidate([self.candidate('a', .4, .2), self.candidate('b', .4, .3)])['candidate_id'], 'b')
        def test_latency_then_id(self):
            self.assertEqual(select_candidate([self.candidate('a', latency=12), self.candidate('b', latency=10)])['candidate_id'], 'b')
            self.assertEqual(select_candidate([self.candidate('b'), self.candidate('a')])['candidate_id'], 'a')
        def test_holdout_forbidden(self):
            with self.assertRaises(ValueError):
                select_candidate([self.candidate('a', split='test')])
        def test_nonfinite_refused(self):
            with self.assertRaises(ValueError):
                select_candidate([self.candidate('a', ap300=float('nan'))])
        def test_command_frozen_threshold(self):
            ev={'python_executable':'python','evaluator_path':'eval.py','manifest_path':'m.json'}
            cmd=command(ev, {'weights':'w.pt','pipeline':'dual'}, Path('out'),'test','pi_test',.35,0)
            self.assertIn('--operating-conf',cmd)
            self.assertEqual(cmd[cmd.index('--operating-conf')+1], format(.35,'.17g'))
        def test_hash_binding_changes(self):
            a={'split':'test','confidence':.35,'checkpoint':'abc'}
            self.assertNotEqual(fingerprint(a),fingerprint(dict(a,confidence=.4)))
        def test_test_metric_threshold_tamper(self):
            binding={'checkpoint_sha256':'w','manifest_sha256':'m','evaluator_sha256':'e','pipeline':'dual',
                     'split':'test','cohort':'pi_test','operating_confidence':.35}
            metric={**{k:v for k,v in binding.items() if k!='operating_confidence'},
                    'evaluation_role':'development_holdout_not_pristine_final','d455_verified':False,
                    'operating':{'confidence':.40},'threshold_source':'supplied_frozen_confidence'}
            with self.assertRaises(ValueError):
                check_metric(metric,binding)
        def test_incomplete_output_inventory_refused(self):
            record={'binding':{'split':'val'},'output_sha256':{'metrics.json':'fake'}}
            with self.assertRaises(ValueError):
                validate_completed_files(record,Path('no-output-needed'))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--self-test', action='store_true', help='CPU-only orchestration policy tests; no evaluator subprocess')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    try:
        execute(args.root.resolve(), args.python.resolve())
    except BaseException as error:
        log_stage(args.root.resolve(), f'R04 EVALUATION STOPPED: {type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()

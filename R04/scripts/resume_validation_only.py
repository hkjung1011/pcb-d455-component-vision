"""Resume frozen R04 validation and freeze selections, then STOP before holdout.

Delegates evaluation and selection to the existing immutable orchestrator.
This execution boundary adds no changed metric, data, or selection rule.
"""
from pathlib import Path
import sys
import run_r04_evaluation as runner


def main():
    root = Path(__file__).resolve().parents[1]
    suite = root / 'reports/evaluation_suite'
    evidence = runner.preflight(root, Path(sys.executable).resolve())
    protocol_hash = runner.fingerprint(evidence)
    stored = runner.read(suite / 'protocol.json')
    if stored.get('protocol_sha256') != protocol_hash or runner.fingerprint(stored['evidence']) != protocol_hash:
        raise ValueError('Frozen protocol changed; cannot resume')
    if any(runner.read(p).get('binding', {}).get('split') == 'test' for p in (suite / 'jobs').glob('*.json')):
        raise ValueError('Validation-only wrapper is intended before the first holdout')
    validation, winners = {}, {}
    for arm in runner.ARMS:
        validation[arm] = [runner.evaluate_job(root, suite, evidence, protocol_hash, c, 'val')
                           for c in evidence['candidates'][arm]]
        winners[arm] = runner.select_candidate(validation[arm])
    payload = runner.frozen_selection_payload(protocol_hash, validation, winners)
    selection_path = suite / 'selections_frozen.json'
    if selection_path.exists():
        locked = runner.read(selection_path)
        if runner.fingerprint({k: v for k, v in locked.items() if k != 'frozen_utc'}) != runner.fingerprint(payload):
            raise ValueError('Existing selection differs; cannot rewrite')
    else:
        runner.save(selection_path, {'frozen_utc': runner.now(), **payload})
    selection_sha = runner.sha(selection_path)
    aggregate = {
        'status': 'SELECTIONS_FROZEN_DEVELOPMENT_HOLDOUT_PENDING',
        'protocol_sha256': protocol_hash, 'selection_sha256': selection_sha,
        'policy': runner.POLICY, 'recommended_arm': payload['recommended_arm'],
        'recommended_candidate_id': payload['recommended_candidate_id'],
        'fresh_final_test_available': False, 'd455_verified': False,
        'arms': {arm: {'selection': payload['selections'][arm],
                       'validation_candidates': [runner.compact(r) for r in validation[arm]],
                       'tests': {}} for arm in runner.ARMS},
    }
    runner.save(suite / 'aggregate.json', aggregate)
    runner.save(suite / 'validation_only_boundary.json', {
        'status': 'VALIDATION_COMPLETE_HOLDOUT_NOT_STARTED', 'completed_utc': runner.now(),
        'wrapper_sha256': runner.sha(__file__), 'protocol_sha256': protocol_hash,
        'selection_sha256': selection_sha, 'validation_jobs': sum(map(len, validation.values())),
        'recommended_arm': payload['recommended_arm'],
        'note': 'Frozen original functions reused. Await separate read-only pre-holdout audit.'})
    runner.log_stage(root, 'VALIDATION-ONLY STOP: 40 jobs completed; selections frozen; no holdout executed')


if __name__ == '__main__':
    main()

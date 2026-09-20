"""Run one complete ProteinGym assay locally; inputs/weights remain private.

Install the v0.4.0 release wheel with its esm extra and pandas, then use:
python reproduce.py --data /data/assays --checkpoint /weights/esm2_t6_8M_UR50D.pt --output /new/private/run
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import socket
from pathlib import Path

import numpy as np
import torch
from rewirebench import evaluate, export, prepare, run
from rewirebench.adapters.esm import ESM2Adapter
from rewirebench.protocols import proteingym as pg

ASSAY = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
ASSAY_SHA = "dd911d925eca79329a496eb6cb7b181ab448e15db244daf1c1f9c01e8a704c2a"

def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    # Block socket construction after resources are prepared. The entire inference,
    # evaluation and export below therefore execute without network access.
    def offline(*args, **kwargs):
        raise RuntimeError('Network disabled during local inference and evaluation')
    socket.socket = offline
    model = {'name': 'ESM-2 esm2_t6_8M_UR50D masked marginals',
             'training_overlap': 'unreported; UniRef50 pretraining may overlap benchmark proteins',
             'input_information': 'Wild-type protein sequence and amino-acid substitutions; no MSA, structure or assay labels',
             'configuration': {'checkpoint': 'esm2_t6_8M_UR50D', 'strategy': 'masked_marginals', 'seed': 0, 'device': 'cpu', 'threads': 1}}
    opts = {'assay_ids': [ASSAY], 'expected_hashes': {ASSAY: ASSAY_SHA}}
    smoke = prepare(pg.PROTOCOL_ID, source=a.data, output=a.output/'smoke-prepared', limit=10, **opts)
    smoke_report = run(smoke, ESM2Adapter(a.checkpoint), output=a.output/'smoke-result', model=model)
    assert smoke_report['scope'] == 'smoke' and not smoke_report['metrics']
    data = prepare(pg.PROTOCOL_ID, source=a.data, output=a.output/'prepared', **opts)
    adapter = ESM2Adapter(a.checkpoint)
    report = run(data, adapter, output=a.output/'evaluation', model=model)
    assert report['coverage'] == {'denominator': 2972, 'scored': 2972, 'unscored': 0}
    assert report['scope'] == 'subset' and not report['metrics']
    assert report['protocol_results']['per_assay'][ASSAY]['status'] == 'complete'
    predictions = a.output/'evaluation'/'predictions.json'
    recomputed = evaluate(data, predictions, output=a.output/'rescored', model=model)
    assert recomputed['protocol_results'] == report['protocol_results']
    # Execute the pinned official NDCG/top-recall functions and use its exact
    # Spearman/AUC/MCC formulas (upstream_performance.py lines 212-223).
    spec = importlib.util.spec_from_file_location('upstream', pg.resource_path('upstream_performance.py'))
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    values = json.loads(predictions.read_text())
    truth = np.array([r['target'] for r in data['rows']])
    binary = np.array([r['target_binary'] for r in data['rows']])
    scores = np.array([values[r['id']] for r in data['rows']])
    raw = {'Spearman': upstream.spearmanr(truth, scores)[0],
           'AUC': upstream.roc_auc_score(binary, scores),
           'MCC': upstream.matthews_corrcoef(binary, scores >= np.median(scores)),
           'NDCG': upstream.calc_ndcg(truth, scores),
           'Top_recall': upstream.calc_toprecall(truth, scores)}
    expected = {k: float(np.round(v, 3)) for k, v in raw.items()}
    actual = report['protocol_results']['per_assay'][ASSAY]['metrics']
    assert actual == expected, (actual, expected)
    audit = {'review_method': 'automated_execution_and_source_formula_comparison',
             'upstream_revision': pg.UPSTREAM_REVISION,
             'upstream_evaluator_sha256': hashlib.sha256(pg.resource_path('upstream_performance.py').read_bytes()).hexdigest(),
             'reference_sha256': pg.REFERENCE_SHA256,
             'checks': {'all_expected_variants_scored': True, 'unique_ids_validated_by_sdk': True,
                        'no_unscored_variants': True, 'smoke_separate_from_complete_assay': True,
                        'no_full_suite_metric': True, 'held_out_labels_excluded_from_adapter': True,
                        'network_blocked_during_execution': True, 'saved_predictions_recompute_exactly': True,
                        'rounded_metrics_match_pinned_upstream_formulas': True},
             'raw_reference_metrics': {k: float(v) for k,v in raw.items()},
             'rounded_metrics': expected, 'absolute_tolerance': 1e-12,
             'maximum_rounded_absolute_difference': max(abs(actual[k]-expected[k]) for k in expected),
             'scope': 'one_complete_assay_of_217; not_a_whole_suite_score',
             'scientific_reproduction': 'new_local_evaluation_not_reproduction_of_a_published_model_score',
             'versions': {k: importlib.metadata.version(k) for k in ['rewirebench','numpy','scipy','scikit-learn','fair-esm','torch','pandas']},
             'hardware': {'architecture': platform.machine(), 'processor': platform.processor(), 'threads': 1, 'accelerator': 'none'},
             'timing_scope': 'SDK inference_and_fit_seconds excludes model loading and data preparation; includes all selected predictions, using cached masked-position logits'}
    write(a.output/'audit.json', audit)
    export(report, output=a.output/'submission.json')
    print(json.dumps({'metrics': actual, 'coverage': report['coverage'], 'seconds': report['execution']['inference_and_fit_seconds']}, indent=2))

if __name__ == '__main__':
    main()

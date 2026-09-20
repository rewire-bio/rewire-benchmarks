"""Independently replay the pinned upstream masked-marginal loop on saved inputs."""
import argparse
import ast
import csv
import hashlib
import json
import socket
import time
from pathlib import Path

import torch
from rewirebench.adapters.esm import ESM2Adapter
from rewirebench.protocols import proteingym as pg

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--checkpoint', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
def offline(*args, **kwargs):
    raise RuntimeError('Network disabled')
socket.socket = offline
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
data = json.loads((a.run/'prepared/prepared.json').read_text())
predictions = json.loads((a.run/'evaluation/predictions.json').read_text())
source = pg.resource_path('upstream_esm_compute_fitness.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest() == '35186603513cae44fb6a2a44c72cb085db469115e145a267bd7b78350b9bb1ba'
# Execute the actual pinned upstream label_row definition without importing its
# unrelated CLI/MSA dependencies. It sums mutant-minus-WT, not the reverse.
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'label_row')
namespace = {}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned upstream label_row>', 'exec'), namespace)  # noqa: S102 - SHA256-verified upstream function only
load_start = time.perf_counter()
adapter = ESM2Adapter(a.checkpoint)
load_seconds = time.perf_counter()-load_start
sequence = data['rows'][0]['inputs']['wild_type_sequence']
assert all(r['inputs']['wild_type_sequence'] == sequence for r in data['rows'])
_, _, batch_tokens = adapter.converter([('target', sequence)])
original_tokens = batch_tokens.clone()
all_token_probs = []
start = time.perf_counter()
# Upstream lines486-504, with CPU replacing .cuda(); includes BOS/EOS masks.
for i in range(batch_tokens.size(1)):
    masked = batch_tokens.clone()
    masked[0, i] = adapter.alphabet.mask_idx
    with torch.no_grad():
        token_probs = torch.log_softmax(adapter.model(masked)['logits'], dim=-1)
    all_token_probs.append(token_probs[:, i])
token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
assert torch.equal(original_tokens, batch_tokens)
reference_scores = {r['id']: namespace['label_row'](r['inputs']['mutant'], sequence, token_probs, adapter.alphabet, 1) for r in data['rows']}
differences = [abs(reference_scores[k]-predictions[k]) for k in reference_scores]
assert max(differences) <= 1e-9
# Check repeated cache requests and unchanged caller input independently of labels.
inputs = [dict(data['rows'][i]['inputs'], id=data['rows'][i]['id']) for i in [0, 819, 820, 2971]]
before = json.dumps(inputs, sort_keys=True)
first = adapter.predict(inputs)
second = adapter.predict(inputs)
assert first == second and json.dumps(inputs, sort_keys=True) == before
assert all(abs(first[k]-predictions[k]) <= 1e-9 for k in first)
ref = next(r for r in csv.DictReader(pg.resource_path('DMS_substitutions.csv').open()) if r['DMS_id'] == 'AMFR_HUMAN_Tsuboyama_2023_4G3O')
assert ref['raw_DMS_directionality'] == '1' and ref['raw_DMS_phenotype_name'] == 'ddG_ML_float'
result = {'review_method': 'independent_pinned_source_execution', 'upstream_revision': pg.UPSTREAM_REVISION,
          'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
          'source_location': 'compute_fitness.py label_row lines240-250 and masked-marginals lines486-512; CPU device only',
          'compared_variants': len(differences), 'maximum_absolute_prediction_difference': max(differences),
          'absolute_tolerance': 1e-9, 'result': 'passed', 'cache_repeat_and_input_immutability': 'passed',
          'direction': 'log P(mutant)-log P(wild_type), summed for multi-mutants; DMS_score higher is better',
          'source_raw_phenotype': ref['raw_DMS_phenotype_name'], 'source_raw_directionality': ref['raw_DMS_directionality'],
          'validation_model_loading_seconds': load_seconds,
          'validation_masked_loop_plus_comparison_seconds': time.perf_counter()-start,
          'timing_note': 'Separate verification process, not original inference timing or an end-to-end speed benchmark'}
a.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
print(json.dumps(result, indent=2))

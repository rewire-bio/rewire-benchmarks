---
name: find-model-for-use-case
description: Find which biological models (DNA, RNA, protein, single-cell) have published evidence for a stated use case, using the Rewire benchmark knowledge graph. Use when the user asks which model to use for a task such as prioritising splicing variants, ranking rare-disease variants or predicting missense effects, or asks what evidence exists for a model on a use case.
---

# Find the right model for a use case

All tools here are on the MCP server `rewire-benchmark-graph`. It is read-only and holds
reviewed published results. It runs nothing.

The graph does not name a best model. It holds use cases, the assessments that bear on each
one, and the results reported on those assessments. Your answer is that evidence, set out so
the user can choose, with its limits stated.

## Steps

1. `dataset_info`: note the release `version`. Every value you quote comes from that release.
2. `search` with `{"text": "<the user's words>", "types": ["rb:UseCase"]}`, or `find_entity`
   with `"type": "rb:UseCase"`. Release `2026-11-20-72316db0764b` holds 31 use cases. If none
   fits, say so and stop at step 7; do not stretch a use case to fit.
3. `describe_resource` on the use case. Read `rb:question`, `rb:decision`, `rb:clinicalScope`,
   `rb:exclusions`, `rb:evidenceGaps` and `rb:depthTier`, and the `rb:assessedBy` links. Pass
   `"limit": 100`; a use case can carry more than 50 statements.
4. `sparql_query` for the assessments and how much evidence each holds:

   ```sparql
   SELECT ?assessment ?label (COUNT(DISTINCT ?result) AS ?results) (COUNT(DISTINCT ?s) AS ?systems) WHERE {
     id:use-case-splicing-follow-up rb:assessedBy ?assessment . ?assessment rdfs:label ?label .
     OPTIONAL { ?result rb:resultOn ?assessment ; rb:resultFor ?s }
   } GROUP BY ?assessment ?label ORDER BY DESC(?results)
   ```

5. `sparql_query` for the results on one assessment and one metric, with coverage and status:

   ```sparql
   SELECT ?system ?value ?scored ?eligible ?status WHERE {
     ?result rb:resultOn id:rewire-mfass-v2 ; rb:resultFor ?s ; rb:metric metric:average-precision ;
             mls:hasValue ?value ; rb:status ?status .
     ?s rdfs:label ?system .
     OPTIONAL { ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }
   } ORDER BY DESC(?value)
   ```

   On that release it returns five systems, from Pangolin (0.3888, 8301 of 8324 scored) to the
   training prior (0.0378). Values come back as typed literals such as `"0.61"^^xsd:decimal`.
6. `run_query` with `{"name": "contested_results_for_use_case", "params": {"use_case": "id:..."}}`
   for results on these assessments that another study disagrees with, and
   `{"name": "discrepancies_for_system", "params": {"system": "id:..."}}` for one model.
   `list_queries` shows every named query. A discrepancy is a lead, not a finding of error.
7. If the graph has no use case or no results, say so. Offer `search` without a type filter
   for related benchmarks, and do not fill the gap from memory.

More queries, and the vocabulary, are in the `bio-model-benchmarks` skill at
`reference/kg-queries.md`.

## Rules for the answer

- Rank only inside one assessment and one metric. A number from one paper's protocol is not
  comparable with a number from another's.
- Give coverage beside every value (`scored` of `eligible`). A model scored on fewer inputs
  was not tested on the same set.
- Give the status of each result. `status:reproduced` means a Rewire run produced it;
  `status:source_checked` means it was checked against its source and not rerun.
- Show the trivial baseline on the same assessment beside the models, where one exists.
- Say when a ranking rests on one study, and report every contested result from step 6.
- Repeat the use case's own evidence gaps and exclusions. They are part of the answer.
- For clinical use cases, the graph reports published evaluations. It does not validate a
  model for patient care, and you must not say it does.

## Report

Give: the release ID; the use case and its question; one table per assessment (system, value,
coverage, status, source); the baseline; contested results; evidence gaps; and what could be
rerun today. Only a result on a `rewirebench` protocol with a recipe can be rerun here: see the
`run-benchmark` and `verify-published-result` skills.

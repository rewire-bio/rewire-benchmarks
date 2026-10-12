# Knowledge-graph queries

Every query here was run against KG release `2026-10-12-cba623554ce3` with
`rdf-kg-mcp` v0.1.0. Record IDs and values quoted below are from that release; a
later release can add records or supersede them, so always read the current values
rather than copying these.

The server predeclares the bundle prefixes (`rb:`, `id:`, `mls:`, `skos:`, `dcterms:`,
`schema:`, `metric:`, `status:` and others), so queries need no `PREFIX` lines. The
default graph is the union of asserted, inferred and ontology graphs. Pass
`graphs: ["asserted"]` to see only stated facts.

## Vocabulary you need

| Term | Meaning |
| --- | --- |
| `rb:Benchmark`, `rb:Protocol`, `rb:Task` | What was evaluated. A protocol fixes data, split and scoring. |
| `rb:System` | Anything tested: `rb:Model`, `rb:Configuration`, `rb:Pipeline`, `rb:Baseline`, `rb:Method` |
| `rb:Evaluation` | One system on one assessment and dataset. `rb:testedSystem`, `rb:testedOn`, `rb:dataset` |
| `rb:Result` | One metric value. `rb:evaluation`, `rb:metric`, `mls:hasValue`, `rb:printedValue`, `rb:status` |
| `rb:resultFor`, `rb:resultOn`, `rb:resultOnDataset` | Inferred shortcuts from a result to its system, assessment and dataset |
| `rb:resultWithin`, `rb:testedWithin`, `rb:within` | Inferred: result, evaluation or protocol inside a benchmark, transitively |
| `rb:coverage` | Node with `rb:scoredCount` and `rb:eligibleCount` (the original denominator) |
| `rb:uncertainty` | Node with `rb:lowerBound`, `rb:upperBound`, `rb:halfWidth`, `rb:confidenceLevel`, `rb:uncertaintyType` |
| `rb:missingValue` | Node with `rb:missingField` and `rb:missingReason` when a field, such as an interval, was not reported |
| `rb:comparison` (on evaluations) | JSON: `protocol_id`, `split`, `dataset_version`, `population` |
| `rb:reproduction` (on evaluations) | JSON: `recipe_id`, `applicability` (`rescore_predictions` or `generate_and_evaluate`), explanation |
| `rb:runRecipes` (on protocols and benchmarks) | JSON array of run recipes, each with `protocol_id`, `purpose`, `requirements`, `limitations` and per-runtime `instructions` carrying a `status` |
| `rb:status` values | `status:reproduced` (reproduced by a Rewire run), `status:source_checked` (checked against the source, not reproduced), plus `discovered`, `needs_review`, `disputed`, `superseded`, `excluded` |

Metric concepts carry `skos:prefLabel`; the other records carry `rdfs:label`.
Metric keys differ from report keys: `metric:precision-at-100` is
`precision_at_capacity` with `capacity` 100 in an MFASS report, and
`metric:average-precision` is `average_precision_sklearn`.

## Find things

`dataset_info` with no arguments returns the release ID (`version`), graph sizes
and the manifest checksum. Record the release ID in your report.

`find_entity` ranks label matches, and `type` accepts inferred types:

```json
{"text": "MFASS v2", "type": "rb:Protocol"}
```

returns `id:rewire-mfass-v2`. Other tested calls:
`{"text": "ProteinGym", "type": "rb:Benchmark"}` returns `id:discovery-benchmark-proteingym`;
`{"text": "ESM-2", "type": "rb:System"}` returns 35 systems (catalogue models and
paper-specific configurations); `{"text": "Pangolin", "type": "rb:System"}` returns
`id:rewire-model-pangolin-maskfalse` among others. Several records can share a
label, so check `types` and then `describe_resource`.

`describe_resource` with `{"iri": "id:rewire-result-baseline-kmer-position-v2-precision-at-100"}`
lists every statement about the result, each marked asserted or inferred, with labels.

## Published results on a protocol, with coverage

```sparql
SELECT ?system ?metric ?value ?scored ?eligible ?status WHERE {
  ?result rb:resultOn id:rewire-mfass-v2 ; rb:resultFor ?s ; rb:metric ?m ;
          mls:hasValue ?value ; rb:status ?status .
  ?s rdfs:label ?system . ?m skos:prefLabel ?metric .
  OPTIONAL { ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }
} ORDER BY ?system ?metric
```

Returned 15 rows: the corrected k-mer baseline (P@100 0.61, AP 0.2864167459589237,
AUROC 0.7779498064677238, 8324/8324), DNABERT-2 frozen pair, Pangolin (8301/8324),
SpliceAI (8194/8324) and the training prior.

## Results for a model across assessments

```sparql
SELECT ?system ?assessment ?metric ?value WHERE {
  ?result rb:resultFor ?s ; rb:resultOn ?a ; rb:metric ?m ; mls:hasValue ?value .
  ?s rdfs:label ?system FILTER(CONTAINS(?system, "DNABERT-2"))
  ?a rdfs:label ?assessment . ?m skos:prefLabel ?metric .
} ORDER BY ?assessment ?system LIMIT 40
```

Returns published DNABERT-2 rows from BEND, DART-Eval and other papers. These are
the papers' own protocols; a result on another protocol is not comparable with a
`rewirebench` run.

## Results inside a benchmark

```sparql
SELECT ?system ?task ?metric ?value ?scored ?eligible WHERE {
  ?result rb:resultWithin id:discovery-benchmark-mrnabench ; rb:resultFor ?s ; rb:resultOn ?a ;
          rb:metric ?m ; mls:hasValue ?value .
  ?s rdfs:label ?system FILTER(CONTAINS(LCASE(?system), "composition"))
  ?a rdfs:label ?task . ?m skos:prefLabel ?metric .
  OPTIONAL { ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }
} LIMIT 10
```

Protocols inside a benchmark:

```sparql
SELECT ?protocol ?label WHERE {
  ?protocol a rb:Protocol ; rb:within ?b ; rdfs:label ?label .
  ?b rdfs:label "ProteinGym" .
} ORDER BY ?label
```

Benchmarks by number of results:

```sparql
SELECT ?benchmark ?name (COUNT(?result) AS ?results) WHERE {
  ?result rb:resultWithin ?benchmark . ?benchmark a rb:Benchmark ; rdfs:label ?name .
} GROUP BY ?benchmark ?name ORDER BY DESC(?results) LIMIT 15
```

## Everything needed to compare a run with one published result

```sparql
SELECT ?result ?metric ?value ?lower ?upper ?level ?noInterval ?scored ?eligible ?status ?split WHERE {
  ?result rb:resultFor id:rewire-model-baseline-kmer-position-v2 ; rb:resultOn id:rewire-mfass-v2 ;
          rb:metric ?m ; mls:hasValue ?value ; rb:status ?status ; rb:evaluation ?e .
  ?m skos:prefLabel ?metric .
  OPTIONAL { ?result rb:coverage ?c . ?c rb:scoredCount ?scored ; rb:eligibleCount ?eligible }
  OPTIONAL { ?result rb:uncertainty ?u .
             OPTIONAL { ?u rb:lowerBound ?lower } OPTIONAL { ?u rb:upperBound ?upper }
             OPTIONAL { ?u rb:confidenceLevel ?level } }
  OPTIONAL { ?result rb:missingValue ?mv . ?mv rb:missingField "uncertainty" ; rb:missingReason ?noInterval }
  OPTIONAL { ?e rb:comparison ?split }
}
```

For the k-mer baseline this returns no interval (`?noInterval` is
`missingness:unreported`), coverage 8324/8324, status `reproduced`, and the
comparison JSON `{"protocol_id":"mfass-v2","split":"split-v2.tsv","dataset_version":"bee9133b...","population":"8324/8324"}`.
Results that do carry an interval return `?lower`, `?upper` and `?level`, for example
`id:lit-structure-20261011-lipp-result-rmsd-posebusters-v2-308-alphafold-3`
(80.5, 95% interval 75.6 to 84.8).

## Run recipes

Recipe owner for a `rewirebench` protocol ID:

```sparql
SELECT ?owner ?label WHERE {
  ?owner rb:runRecipes ?recipes ; rdfs:label ?label .
  FILTER(CONTAINS(STR(?recipes), "\"protocol_id\":\"mfass-v2\""))
}
```

Returns `id:rewire-mfass-v2`. Read the JSON with
`SELECT ?recipes WHERE { id:rewire-mfass-v2 rb:runRecipes ?recipes }`. It holds three
recipes (`mfass-v2-rescore`, `mfass-v2-baseline`, `mfass-v2-dnabert2`), each with
`requirements` (data, licence, hardware, software, weights), `limitations` and
instructions for `python`, `command_line`, `podman`, `apptainer` and `slurm`.
Every instruction in this release has `status: source_reviewed_not_executed`.

Which recipe a published evaluation maps to, and its split:

```sparql
SELECT ?evaluation ?reproduction ?comparison WHERE {
  id:rewire-result-baseline-kmer-position-v2-auroc rb:evaluation ?evaluation .
  OPTIONAL { ?evaluation rb:reproduction ?reproduction }
  OPTIONAL { ?evaluation rb:comparison ?comparison }
}
```

Returns recipe `mfass-v2-rescore` with applicability `rescore_predictions`: it
rescores the archived predictions and does not by itself regenerate them.

## Sources behind a result

```sparql
SELECT ?source ?title ?url ?locator WHERE {
  id:rewire-result-baseline-kmer-position-v2-auroc dcterms:source ?source ; rb:sourceLocator ?locator .
  OPTIONAL { ?source rdfs:label ?title } OPTIONAL { ?source schema:url ?url }
}
```

Returns the pinned artifact tree at `bee9133b83f3aedaf2bbb9013f1875515845607e` and the
locator `benchmarks/mfass/results/baseline-kmer-position-v2.json :: auroc`. Older
source URLs use the previous repository owner (`timini/rewire-benchmarks`); the same
commit exists in `rewire-bio/rewire-benchmarks`.

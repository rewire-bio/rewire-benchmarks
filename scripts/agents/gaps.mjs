// Gap scanner (#60). Once a week, read the newest published release of the benchmark data and
// find, for tier 1 (DNA and RNA genomics) use cases:
//   - use cases with no direct evidence;
//   - systems ranked on evidence from a single study;
//   - systems with no result from an independent paper or a Rewire run;
//   - open discrepancies between studies that a rerun on a shared protocol would settle;
//   - run recipes nobody has executed (no receipt).
// Each gap gets a value (clinical relevance of its use cases and how many rankings it could
// change) and, where the data gives a basis, a GPU-hour and dollar estimate. A gap is filed only
// when the release shows released code, weights, a container or a recipe for every system, shows
// where the benchmark data can be obtained, and records no successor to its systems; otherwise it
// is blocked and listed in the summary for a person to look at (runnability, dataAccess and
// successorOf below). No model call decides what is filed. The best few are filed here as "Proposed verification run" or "Proposed benchmark" issues, labelled proposal
// and tier-1.
//
// Every number, name, command and link in an issue comes from the release data or this
// repository. Sonnet 5.5 writes only the "Why it matters" paragraph, from computed facts, and a
// paragraph that states a number or a link is dropped. Nothing is ever run: the scanner only
// proposes (rewire-bio/rdf-kg-mcp#6: nothing scheduled starts a GPU run).
//
//   node scripts/agents/gaps.mjs --dry-run [--out <dir>]                       fixtures, no network
//   node scripts/agents/gaps.mjs --report-only --release-dir <dir> [--out <dir>]
//   node scripts/agents/gaps.mjs --release-dir <dir> [--run-url <url>] [--out <dir>]   files issues
//   node scripts/agents/gaps.mjs --download <dir>                              fetch the newest release
//
// GITHUB_REPOSITORY is where proposals are filed (a rewire-bio repository); GH_TOKEN or
// GITHUB_TOKEN authenticates; ANTHROPIC_API_KEY is optional (without it every proposal is filed
// without the paragraph). GAPS_DATA_REPO overrides DATA_REPO. The Anthropic SDK is imported only
// when a model call is made, so tests and dry runs need no install.

import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { createGunzip } from 'node:zlib';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

export const ORG = 'rewire-bio';
// The evidence store whose releases are read. The only place its name is written; the workflow
// passes GAPS_DATA_REPO when the store moves.
export const DATA_REPO = 'rewire-bio/rewire-benchmark-data';
export const SITE = 'https://benchmarks.rewirebio.io';
export const PROMPT_PATH = 'agents/gaps.md';
export const FIXTURES = 'tests/agents-gaps-fixtures';
export const SMOKE_DIR = 'scripts/recipe_smoke';
export const MODELS = { why: 'claude-sonnet-5-5' };
export const LIMITS = {
  maxIssues: 5, // proposals filed per run
  maxRunUsd: 1, // no model call once the run's estimated cost passes this
  whyMaxTokens: 2000,
  commandLines: 60, // lines of recipe command shown before the rest is left to the recipe
  listed: 25, // gaps listed in the job summary
  factsPerList: 10,
};
// USD per million tokens, from the Claude pricing page (17 November 2026). A refusal fallback
// model not listed is priced as Opus 5.5.
export const PRICES = {
  'claude-sonnet-5-5': { input: 2, output: 10, cacheWrite: 2.5, cacheRead: 0.1 },
  'claude-sonnet-5': { input: 2, output: 10, cacheWrite: 2.5, cacheRead: 0.2 },
  'claude-opus-5-5': { input: 4, output: 20, cacheWrite: 5, cacheRead: 0.2 },
};
// Dated, sourced hourly GPU prices: { match: RegExp on the hardware text, usdPerGpuHour,
// source, date }. Empty: no price has been sourced yet, so no dollar figure is given for a GPU
// run. Add an entry only with its source and date; the issue quotes both.
export const GPU_PRICES = [];
export const LABELS = {
  proposal: { color: '5319e7', description: 'A run or benchmark an agent proposes; a person decides and runs it' },
  'tier-1': { color: '0e8a16', description: 'Depth tier 1: DNA and RNA genomics' },
};
export const MARKER = 'gap-proposal';
const MARKER_RE = /<!-- gap-proposal: ([A-Za-z0-9:._-]+) -->/g;
export const INDEPENDENT_ORIGINS = ['independent_paper', 'rewire_run'];
const CLINICAL_CONTEXTS = ['clinical', 'clinical_research'];
const GENOMIC_AREAS = ['dna-genomes', 'rna-transcriptomes'];
const PROTEIN_AREAS = ['proteins-complexes', 'protein-function', 'protein-structure'];
const NOT_EXECUTED = 'source_reviewed_not_executed';
const RERUNNABLE_SCOPES = ['same-protocol', 'same-task'];
const SYSTEM_KINDS = ['model', 'method', 'configuration', 'pipeline', 'baseline', 'service'];

export const TYPES = {
  'no-direct-evidence': 'Use case with no direct evidence',
  'single-study': 'Ranked on a single study',
  'no-independent': 'No independent evaluation',
  discrepancy: 'Open discrepancy a rerun would settle',
  'unverified-recipe': 'Run recipe never executed',
};

export const RANKING_RULE = 'Proposals are ordered by value per GPU-hour, at most one per benchmark a week: first runs whose recipe states they need no GPU (0 GPU-hours), by value; then runs with a GPU-hour estimate, by value per GPU-hour; then runs with no estimate, by value. Value is the clinical weight (2 when a use case it serves has a clinical context, else 1) times one plus the number of rankings it could change. A ranking is one tier 1 use case\'s active direct protocol on which at least two systems have results.';

export const NOT_RUN = 'Nothing has been run. The scanner only proposes: a person decides whether to run this and starts it (rewire-bio/rdf-kg-mcp#6). Recording any result goes through a pull request to the data store.';

/** Depth tier from areas: any genomic area is 1, else any protein area is 2, else 3 (docs/tiers.md in the data store). */
export function tierOf(areas = []) {
  if (areas.some((a) => GENOMIC_AREAS.includes(a))) return 1;
  if (areas.some((a) => PROTEIN_AREAS.includes(a))) return 2;
  return 3;
}

/** Lower case, every run of non-alphanumerics one space. */
export function normalise(text) {
  return String(text ?? '').toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

/**
 * Model text goes into public issues. Remove @-mentions, HTML comments (they could fake a
 * marker) and em and en dashes (house style).
 */
export function clean(text) {
  return String(text ?? '')
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/(^|[^\w`])@(?=[A-Za-z0-9])/g, '$1')
    .replace(/\s*\u2014\s*/g, ', ')
    .replace(/\u2013/g, '-')
    .trim();
}

// ---------- Reading the release ----------

const RESULT_LINE = /^\{"id":\s*"(?:[^"\\]|\\.)*",\s*"kind":\s*"result"/;

const recipesOf = (a) => [...(Array.isArray(a.run_recipes) ? a.run_recipes : []), ...(a.run_guide && typeof a.run_guide === 'object' ? [a.run_guide] : [])];

function slim(r) {
  const a = r.attributes ?? {};
  return {
    id: r.id,
    kind: r.kind,
    name: r.name ?? r.id,
    description: r.description ?? '',
    links: r.links ?? [],
    source_ids: r.source_ids ?? [],
    areas: r.facets?.areas ?? [],
    contexts: r.facets?.contexts ?? [],
    tier: a.depth_tier ?? null,
    origin: a.origin ?? null,
    url: a.url ?? null,
    doi: a.doi ?? null,
    run_recipes: Array.isArray(a.run_recipes) ? a.run_recipes : null,
    run_guide: a.run_guide && typeof a.run_guide === 'object' ? a.run_guide : null,
    combination: Boolean(a.combination_rule),
    metric: a.metric ?? null,
    unit: a.unit ?? null,
    printed: a.printed_value ?? null,
    value: a.numeric_value ?? null,
    approximate: Boolean(a.approximate ?? r.approximate),
    hardware: a.inference_hardware ?? a.hardware ?? null,
    // An evaluation that records only training hardware measures training, which is never a basis.
    trainingOnly: Boolean(a.training_hardware) && !a.inference_hardware,
    // What says whether a system can be run by software (runnability below).
    access: [a.access, a.access_reuse].filter((x) => typeof x === 'string'),
    baselineType: a.baseline_type ?? null,
    artifacts: ARTIFACT_KEYS.filter((k) => a[k] !== undefined && a[k] !== null && !/^(?:unreported|not stated|unknown|none)\b/i.test(String(a[k]))),
    adaptation: typeof (a.comparison?.adaptation ?? a.adaptation) === 'string' ? (a.comparison?.adaptation ?? a.adaptation) : null,
    // What says whether a dataset, protocol or benchmark's data can be obtained (dataAccess below).
    dataAccessNotes: [a.access, a.reuse_restrictions].filter((x) => typeof x === 'string'),
    notes: [a.scope_note, ...(Array.isArray(a.limitations) ? a.limitations : []), ...recipesOf(a).flatMap((r) => (Array.isArray(r?.limitations) ? r.limitations : []))].filter((x) => typeof x === 'string'),
    // What the run recipes and run guide say the data needs.
    recipeDataNotes: recipesOf(a).map((r) => r?.requirements?.data).filter((x) => typeof x === 'string'),
    dataRefs: DATA_REF_KEYS.filter((k) => typeof a[k] === 'string' && a[k].trim() && !/^(?:unreported|not stated|unknown|none)\b/i.test(a[k])).map((k) => `${k} ${a[k]}`),
    privateData: a.private_data_not_included === true,
    supersededBy: typeof a.superseded_by === 'string' ? a.superseded_by : null,
  };
}

/**
 * Records from the release's records.jsonl(.gz), slimmed. Results are skipped before parsing,
 * except runtime results, which can give a GPU-hour basis.
 */
export async function readRecords(path) {
  const recs = new Map();
  const runtimes = [];
  const stream = path.endsWith('.gz') ? createReadStream(path).pipe(createGunzip()) : createReadStream(path);
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    if (RESULT_LINE.test(line)) {
      if (!line.includes('"runtime"')) continue;
      const r = slim(JSON.parse(line));
      if (r.metric === 'runtime') runtimes.push(r);
      continue;
    }
    const r = slim(JSON.parse(line));
    recs.set(r.id, r);
  }
  return { recs, runtimes };
}

function readMaybeGz(path) {
  const buf = readFileSync(path);
  if (path.endsWith('.gz')) return JSON.parse(execFileSync('gzip', ['-dc'], { input: buf, maxBuffer: 1 << 30 }).toString('utf8'));
  return JSON.parse(buf.toString('utf8'));
}

/** The three files of one release in a folder: use-cases, records and discrepancies. */
export function releaseFiles(dir) {
  const names = readdirSync(dir);
  const find = (re) => {
    const f = names.find((n) => re.test(n));
    if (!f) throw new Error(`no file matching ${re} in ${dir}`);
    return join(dir, f);
  };
  return {
    useCases: find(/use-cases\.json(\.gz)?$/),
    records: find(/records\.jsonl(\.gz)?$/),
    discrepancies: find(/discrepancies\.json(\.gz)?$/),
  };
}

/**
 * Download the newest release into `dir`: the newest serving/<id> (in the order releases were
 * cut) that also has a kg/<id> bundle, as the disagreement watcher in the data store picks it.
 * Uses the gh CLI. Returns the release ID.
 */
export function downloadRelease(dir, repo, run = (args) => execFileSync('gh', args, { encoding: 'utf8', maxBuffer: 1 << 26 })) {
  const list = JSON.parse(run(['release', 'list', '--repo', repo, '--limit', '1000', '--json', 'tagName,createdAt,isDraft']));
  const id = latestRelease(list);
  if (!id) throw new Error(`no serving/<id> release with a kg/<id> bundle in ${repo}`);
  mkdirSync(dir, { recursive: true });
  run(['release', 'download', `serving/${id}`, '--repo', repo, '--dir', dir, '--clobber', '--pattern', `public.omics.releases.${id}.use-cases.json.gz`, '--pattern', `public.omics.releases.${id}.records.jsonl.gz`]);
  run(['release', 'download', `kg/${id}`, '--repo', repo, '--dir', dir, '--clobber', '--pattern', 'discrepancies.json']);
  return id;
}

/** Newest release ID with both a serving/ and a kg/ release, ordered by when serving/ was cut. */
export function latestRelease(list) {
  const live = list.filter((r) => !r.isDraft);
  const kg = new Set(live.filter((r) => r.tagName.startsWith('kg/')).map((r) => r.tagName.slice(3)));
  const serving = live.filter((r) => r.tagName.startsWith('serving/') && kg.has(r.tagName.slice(8)))
    .sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
  return serving.length ? serving[serving.length - 1].tagName.slice(8) : null;
}

// ---------- Index ----------

const githubRepo = (url) => {
  const m = /github\.com\/([^/\s]+)\/([^/\s#?]+)/i.exec(String(url ?? ''));
  return m ? `${m[1]}/${m[2].replace(/\.git$/, '')}`.toLowerCase() : null;
};

/**
 * Lookups over the release: links, system identity, studies, benchmarks and recipes.
 * Identity follows the discrepancy rules: a configuration is its model through configuration_of,
 * then family, then alias_of; a configuration with a combination rule is itself.
 * A study is a set of sources that share a DOI or a GitHub repository, or are joined by a
 * companion_of link (docs/discrepancies.md in the data store).
 */
export function buildIndex({ recs, runtimes = [] }) {
  const linkOf = (id, rel) => (recs.get(id)?.links ?? []).filter((l) => l.relation === rel).map((l) => l.target_id);
  const name = (id) => recs.get(id)?.name ?? id;

  function resolveSystem(id) {
    let cur = id;
    const seen = new Set();
    while (recs.has(cur) && !seen.has(cur)) {
      seen.add(cur);
      const r = recs.get(cur);
      if (r.combination) return cur;
      const next = r.kind === 'configuration'
        ? (linkOf(cur, 'configuration_of')[0] ?? linkOf(cur, 'family')[0] ?? linkOf(cur, 'alias_of')[0])
        : linkOf(cur, 'alias_of')[0];
      if (!next || !recs.has(next)) return cur;
      cur = next;
    }
    return cur;
  }

  // Studies by union-find over sources.
  const parent = new Map();
  const find = (x) => {
    if (!parent.has(x)) parent.set(x, x);
    let p = x;
    while (parent.get(p) !== p) p = parent.get(p);
    let c = x;
    while (parent.get(c) !== p) {
      const n = parent.get(c);
      parent.set(c, p);
      c = n;
    }
    return p;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(rb < ra ? ra : rb, rb < ra ? rb : ra);
  };
  for (const r of recs.values()) {
    if (r.kind !== 'source') continue;
    find(r.id);
    if (r.doi) union(r.id, `doi:${String(r.doi).toLowerCase()}`);
    const repo = githubRepo(r.url);
    if (repo) union(r.id, `repository:${repo}`);
    for (const t of linkOf(r.id, 'companion_of')) union(r.id, t);
  }
  const studyOf = (sourceId) => find(sourceId);

  // part_of upwards to the first benchmark.
  function benchmarkOf(id) {
    const queue = [id];
    const seen = new Set();
    while (queue.length) {
      const cur = queue.shift();
      if (seen.has(cur)) continue;
      seen.add(cur);
      if (cur !== id && recs.get(cur)?.kind === 'benchmark') return cur;
      if (seen.size > 20) break;
      queue.push(...linkOf(cur, 'part_of'));
    }
    return recs.get(id)?.kind === 'benchmark' ? id : null;
  }

  /** Every record a record is part_of, transitively (its tasks, suites and benchmarks). */
  function ancestors(id) {
    const out = new Set();
    const queue = [...linkOf(id, 'part_of')];
    while (queue.length && out.size < 50) {
      const cur = queue.shift();
      if (out.has(cur)) continue;
      out.add(cur);
      queue.push(...linkOf(cur, 'part_of'));
    }
    return out;
  }

  // Evaluations by the protocol, task or benchmark they assess.
  const evalsByAssessment = new Map();
  for (const r of recs.values()) {
    if (r.kind !== 'evaluation') continue;
    for (const a of linkOf(r.id, 'assessment')) evalsByAssessment.set(a, [...(evalsByAssessment.get(a) ?? []), r.id]);
  }
  // Records that are part_of each record, one level down.
  const children = new Map();
  for (const r of recs.values()) for (const t of linkOf(r.id, 'part_of')) children.set(t, [...(children.get(t) ?? []), r.id]);
  function descendants(id) {
    const out = new Set([id]);
    const queue = [id];
    while (queue.length && out.size < 5000) for (const c of children.get(queue.shift()) ?? []) if (!out.has(c)) { out.add(c); queue.push(c); }
    return out;
  }

  /** The record that carries the recipe for a protocol: the protocol itself, else its benchmark. */
  function recipeOwner(id) {
    const own = recs.get(id);
    if (own && (own.run_recipes?.length || own.run_guide)) return id;
    const b = benchmarkOf(id);
    const br = b ? recs.get(b) : null;
    return br && (br.run_recipes?.length || br.run_guide) ? b : null;
  }

  /** The system record and every configuration that resolves to it. */
  let members = null;
  function membersOf(systemId) {
    if (!members) {
      members = new Map();
      for (const r of recs.values()) {
        if (!SYSTEM_KINDS.includes(r.kind)) continue;
        const s = resolveSystem(r.id);
        members.set(s, [...(members.get(s) ?? []), r.id]);
      }
    }
    return members.get(systemId) ?? [systemId];
  }

  return { recs, runtimes, linkOf, name, resolveSystem, studyOf, benchmarkOf, ancestors, evalsByAssessment, descendants, recipeOwner, membersOf };
}

function systemOfEvaluation(ix, evalId) {
  const sys = ix.linkOf(evalId, 'system')[0];
  return sys ? ix.resolveSystem(sys) : null;
}

function recordTier(ix, id) {
  const r = ix.recs.get(id);
  if (!r) return null;
  return r.tier ?? tierOf(r.areas);
}

// ---------- Recipes and estimates ----------

/** Every status a recipe owner records: each instruction of each recipe, and the run guide. */
export function recipeStatuses(rec) {
  const out = [];
  for (const recipe of rec?.run_recipes ?? []) for (const i of recipe.instructions ?? []) if (i.status) out.push(i.status);
  if (rec?.run_guide?.status) out.push(rec.run_guide.status);
  return out;
}

/** The recipe that best fits a generate-and-evaluate run: purpose generate_and_evaluate first. */
function mainRecipe(rec) {
  const list = rec?.run_recipes ?? [];
  return list.find((r) => r.purpose === 'generate_and_evaluate') ?? list[0] ?? null;
}

/**
 * The command a person would run, verbatim from the store: the run guide's steps, else the main
 * recipe's instructions. Null when the store has neither.
 */
export function commandFor(rec) {
  if (!rec) return null;
  if (rec.run_guide?.steps?.length) {
    return {
      from: `run guide of \`${rec.id}\` (status \`${rec.run_guide.status ?? 'unknown'}\`)`,
      blocks: rec.run_guide.steps.map((s) => ({ title: s.title ?? '', lang: 'sh', code: s.shell ?? '' })).filter((b) => b.code),
    };
  }
  const r = mainRecipe(rec);
  if (!r?.instructions?.length) return null;
  const lang = (rt) => (rt === 'python' ? 'python' : 'sh');
  return {
    from: `recipe \`${r.id}\` of \`${rec.id}\`${r.version ? ` at \`${r.version}\`` : ''}`,
    blocks: r.instructions.filter((i) => ['command_line', 'python'].includes(i.runtime) || !i.runtime).map((i) => ({ title: i.title ?? '', lang: lang(i.runtime), code: i.code ?? '' })).filter((b) => b.code),
  };
}

const CPU_ONLY = /\bno (?:accelerator|gpu)s? (?:is |are )?(?:required|needed)\b|\bcpu[- ]only\b|\bdoes not need a gpu\b/i;

/** The hardware text of a recipe owner's main recipe, if any. */
function hardwareText(rec) {
  const r = mainRecipe(rec);
  return r?.requirements?.hardware ? clean(r.requirements.hardware) : null;
}

function hoursOf(value, unit) {
  const v = Number(value);
  if (!Number.isFinite(v)) return null;
  if (unit === 'hour') return v;
  if (unit === 'minute') return v / 60;
  if (unit === 'second') return v / 3600;
  if (unit === 'day') return v * 24;
  return null;
}

/** GPU count and model named in a hardware string: "2 × Quadro RTX 8000", "8 NVIDIA A100 GPUs". Null without both. */
export function gpuCount(text) {
  const t = String(text ?? '');
  if (!/gpu|nvidia|a100|h100|v100|l4\b|l40|a10\b|t4\b|rtx|tesla/i.test(t)) return null;
  const m = /(\d+)\s*(?:[×x*]\s*)?(?:nvidia|tesla|quadro|geforce|a100|h100|v100|l40|l4|a10|t4|rtx|gpus?)/i.exec(t);
  return m ? Number(m[1]) : null;
}

/**
 * GPU-hour and dollar estimate for running `systems` on `protocols`, with its basis.
 * 1. Runtime results recorded for the systems on the protocols, whose hardware names a GPU and
 *    its count: hours times GPU count, one result per system and protocol, summed. When results
 *    cover only some of the system and protocol pairs, the figure is a lower bound and says so.
 * 2. The recipe's hardware field says no GPU is needed: 0 GPU-hours and no GPU cost.
 * 3. Otherwise no estimate, with the hardware text if there is one.
 * Dollars come only from GPU_PRICES (dated and sourced), and only when every counted runtime
 * has a price; otherwise no dollar estimate.
 */
export function estimate(ix, { systems = [], protocols = [], owner = null }) {
  const sys = new Set(systems);
  const prot = new Set(protocols);
  const byPair = new Map(); // "system|protocol" -> first usable runtime
  for (const r of ix.runtimes) {
    // Runtime results are kept apart from the other records, so read the link off the result.
    const ev = (r.links ?? []).find((l) => l.relation === 'evaluation')?.target_id;
    if (!ev) continue;
    const s = systemOfEvaluation(ix, ev);
    const a = ix.linkOf(ev, 'assessment')[0];
    if (!sys.has(s) || !prot.has(a)) continue;
    if (ix.recs.get(ev)?.trainingOnly) continue;
    const pair = `${s}|${a}`;
    if (byPair.has(pair)) continue;
    const hw = r.hardware ?? ix.recs.get(ev)?.hardware;
    const n = gpuCount(hw);
    const h = hoursOf(r.value, r.unit);
    if (!n || h === null) continue;
    byPair.set(pair, { r, s, a, hw, gpuHours: h * n, price: GPU_PRICES.find((p) => p.match.test(String(hw))) });
  }
  if (byPair.size) {
    const parts = [...byPair.values()];
    const gpuHours = Number(parts.reduce((t, p) => t + p.gpuHours, 0).toPrecision(3));
    const pairs = sys.size * prot.size;
    const partial = byPair.size < pairs;
    const approx = parts.some((p) => p.r.approximate);
    const priced = parts.every((p) => p.price);
    const usd = priced ? Number(parts.reduce((t, p) => t + p.gpuHours * p.price.usdPerGpuHour, 0).toFixed(2)) : null;
    const each = parts.map((p) => `result \`${p.r.id}\`: runtime ${p.r.printed ?? p.r.value} ${p.r.unit}${p.r.approximate ? ' (approximate)' : ''} on ${clean(p.hw)}, for ${ix.name(p.s)} on ${ix.name(p.a)}`).join('; ');
    const caveats = [
      partial ? `a lower bound: runtimes are recorded for ${byPair.size} of the ${pairs} system and protocol pairs this run covers` : null,
      approx ? 'the paper gives an approximate runtime' : null,
    ].filter(Boolean);
    return {
      kind: 'gpu',
      gpuHours,
      lowerBound: partial,
      approximate: approx,
      usd,
      basis: `${each}.${caveats.length ? ` This is ${caveats.join(', and ')}.` : ''}`,
      usdBasis: priced ? [...new Set(parts.map((p) => `${p.price.usdPerGpuHour} USD per GPU-hour (${p.price.source}, ${p.price.date})`))].join('; ') : 'no dated, sourced hourly price is recorded for every GPU used (GPU_PRICES in scripts/agents/gaps.mjs).',
    };
  }
  const rec = owner ? ix.recs.get(owner) : null;
  const hw = hardwareText(rec);
  if (hw && CPU_ONLY.test(hw)) {
    return { kind: 'cpu', gpuHours: 0, usd: 0, basis: `the hardware field of the ${commandFor(rec)?.from ?? `recipe of \`${owner}\``} says "${hw}"`, usdBasis: 'no GPU is used, so there is no GPU cost. Runner time is not estimated.' };
  }
  return {
    kind: 'none',
    gpuHours: null,
    usd: null,
    basis: hw ? `the recipe's hardware field gives no figure ("${hw}"), and no runtime result is recorded for this system on this protocol.` : rec ? 'the recipe states no hardware, and no runtime result is recorded for this system on this protocol.' : 'there is no recipe and no runtime result for this run.',
    usdBasis: 'there is no GPU-hour estimate to price.',
  };
}

// ---------- Gaps ----------

function useCaseInfo(ix, uc) {
  const rec = ix.recs.get(uc.id);
  const contexts = rec?.contexts?.length ? rec.contexts : (uc.contexts ?? []);
  return {
    id: uc.id,
    title: uc.title ?? rec?.name ?? uc.id,
    slug: uc.slug ?? uc.id.replace(/^use-case-/, ''),
    question: uc.question ?? rec?.description ?? '',
    contexts,
    clinical: contexts.some((c) => CLINICAL_CONTEXTS.includes(c)),
    tier: rec?.tier ?? tierOf(rec?.areas?.length ? rec.areas : uc.area ? [uc.area] : []),
  };
}

const clinicalWeight = (ucs) => (ucs.some((u) => u.clinical) ? 2 : 1);

/**
 * Rankings: for each tier 1 use case, each active direct mapping whose eligible evaluations
 * cover at least two systems.
 */
export function rankingsOf(ix, useCasesFile) {
  const tier1 = new Map(useCasesFile.use_cases.map((u) => useCaseInfo(ix, u)).filter((u) => u.tier === 1).map((u) => [u.id, u]));
  const rankings = [];
  for (const m of useCasesFile.mappings ?? []) {
    if (!tier1.has(m.use_case_id) || m.lifecycle !== 'active' || m.relevance !== 'direct') continue;
    const systems = new Set((m.evaluation_ids ?? []).map((e) => systemOfEvaluation(ix, e)).filter(Boolean));
    if (systems.size >= 2) rankings.push({ useCase: m.use_case_id, protocol: m.protocol_id, systems });
  }
  return { tier1, rankings };
}

const gapKey = (...parts) => parts.join(':').replace(/[^A-Za-z0-9:._-]/g, '-');

/** Every gap in the release, unranked. */
export function findGaps(ix, useCasesFile, discrepanciesFile, { smokeChecks = [] } = {}) {
  const { tier1, rankings } = rankingsOf(ix, useCasesFile);
  const gaps = [];
  const mappingsByUc = new Map();
  for (const m of useCasesFile.mappings ?? []) mappingsByUc.set(m.use_case_id, [...(mappingsByUc.get(m.use_case_id) ?? []), m]);
  const benchKey = (protocol) => ix.benchmarkOf(protocol) ?? protocol;

  // 1. Tier 1 use cases with no active direct mapping.
  const ucDetails = new Map(useCasesFile.use_cases.map((u) => [u.id, u]));
  for (const uc of tier1.values()) {
    const maps = mappingsByUc.get(uc.id) ?? [];
    if (maps.some((m) => m.lifecycle === 'active' && m.relevance === 'direct')) continue;
    const proxy = maps.filter((m) => m.lifecycle === 'active' && m.relevance === 'proxy');
    const studiesBySystem = new Map();
    for (const m of proxy) for (const e of m.evaluation_ids ?? []) {
      const s = systemOfEvaluation(ix, e);
      if (!s) continue;
      const set = studiesBySystem.get(s) ?? new Set();
      for (const src of ix.recs.get(e)?.source_ids ?? []) set.add(ix.studyOf(src));
      studiesBySystem.set(s, set);
    }
    const topProxy = [...studiesBySystem.entries()].sort((a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0])).slice(0, 3).map(([s, set]) => ({ id: s, studies: set.size }));
    const proxyBenchmarks = [...new Set(proxy.map((m) => benchKey(m.protocol_id)))];
    const d = ucDetails.get(uc.id) ?? {};
    gaps.push({
      key: gapKey('use-case', uc.id),
      type: 'no-direct-evidence',
      kind: 'benchmark',
      title: `Proposed benchmark: ${uc.title}`,
      useCases: [uc],
      systems: [],
      baseline: d.baseline ? { name: d.baseline.name, ids: d.baseline.method_ids ?? [] } : null,
      proxySystems: topProxy,
      proxyMappings: proxy.length,
      proxyBenchmarks,
      evidenceGaps: d.evidence_gaps ?? [],
      plan: d.collection_plan ? { status: d.collection_plan.status, question: d.collection_plan.comparison_question, next: d.collection_plan.next_step } : null,
      benchmark: null,
      protocols: [],
      rankings: 1,
      rankingNote: 'it would give the use case its first direct ranking',
      owner: null,
    });
  }

  // 2 and 3. Systems ranked for a tier 1 use case on a single study, or with no independent result.
  const bySystem = new Map(); // key system|bench
  for (const uc of tier1.values()) {
    const direct = (mappingsByUc.get(uc.id) ?? []).filter((m) => m.lifecycle === 'active' && m.relevance === 'direct');
    const stats = new Map();
    for (const m of direct) for (const e of m.evaluation_ids ?? []) {
      const s = systemOfEvaluation(ix, e);
      if (!s) continue;
      const st = stats.get(s) ?? { studies: new Set(), origins: new Set(), protocols: new Set() };
      for (const src of ix.recs.get(e)?.source_ids ?? []) st.studies.add(ix.studyOf(src));
      st.origins.add(ix.recs.get(e)?.origin ?? 'unreported');
      st.protocols.add(m.protocol_id);
      stats.set(s, st);
    }
    for (const [s, st] of stats) {
      const ranked = rankings.filter((r) => r.useCase === uc.id && r.systems.has(s));
      if (!ranked.length) continue; // not ranked: no other system on its protocols
      const flags = [];
      // A second Rewire run is not a second study: a system whose only results are Rewire's own
      // runs is not flagged as single-study.
      if (st.studies.size === 1 && ![...st.origins].every((o) => o === 'rewire_run')) flags.push('single-study');
      if (![...st.origins].some((o) => INDEPENDENT_ORIGINS.includes(o))) flags.push('no-independent');
      if (!flags.length) continue;
      // The benchmark (or bare protocol) holding most of this system's rankings in this use case.
      const counts = new Map();
      for (const r of ranked) counts.set(benchKey(r.protocol), (counts.get(benchKey(r.protocol)) ?? 0) + 1);
      const bench = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0][0];
      const key = `${s}|${bench}`;
      const g = bySystem.get(key) ?? { system: s, bench, useCases: new Map(), flags: new Set(), studies: new Set(), origins: new Set(), protocols: new Set() };
      g.useCases.set(uc.id, uc);
      flags.forEach((f) => g.flags.add(f));
      st.studies.forEach((x) => g.studies.add(x));
      st.origins.forEach((x) => g.origins.add(x));
      for (const r of ranked) if (benchKey(r.protocol) === bench) g.protocols.add(r.protocol);
      bySystem.set(key, g);
    }
  }
  for (const g of bySystem.values()) {
    const protocols = [...g.protocols].sort();
    const rk = rankings.filter((r) => r.systems.has(g.system) && protocols.includes(r.protocol)).length;
    const flags = ['single-study', 'no-independent'].filter((f) => g.flags.has(f));
    const owner = protocols.map((p) => ix.recipeOwner(p)).find(Boolean) ?? null;
    gaps.push({
      key: gapKey('verify', g.system, g.bench),
      type: flags[0],
      flags,
      kind: 'verification',
      title: `Proposed verification run: ${ix.name(g.system)} on ${ix.name(g.bench)}`,
      useCases: [...g.useCases.values()],
      systems: [g.system],
      studies: g.studies.size,
      origins: [...g.origins].sort(),
      benchmark: g.bench,
      protocols,
      rankings: rk,
      rankingNote: `${ix.name(g.system)} is ranked on ${rk} tier 1 ranking${rk === 1 ? '' : 's'} on this benchmark`,
      owner,
    });
  }

  // 4. Open discrepancies a rerun on a shared protocol would settle.
  for (const f of discrepanciesFile?.discrepancies ?? []) {
    if (f.resolved || f.settled) continue;
    if (f.review_state && f.review_state !== 'open') continue;
    if (!RERUNNABLE_SCOPES.includes(f.scope)) continue;
    const tiers = (f.systems ?? []).map((s) => recordTier(ix, s)).filter((t) => t !== null);
    if (!tiers.length || Math.min(...tiers) !== 1) continue;
    const perStudy = (f.studies ?? []).map((s) => new Set((s.comparisons ?? []).map((c) => c.protocol).filter(Boolean)));
    const count = (sets) => {
      const m = new Map();
      for (const set of sets) for (const x of set) m.set(x, (m.get(x) ?? 0) + 1);
      return m;
    };
    const shared = [...count(perStudy).entries()].filter(([, n]) => n >= 2).map(([p]) => p).sort();
    // The anchor to rerun on: a protocol two studies share, else the most specific task, suite
    // or benchmark that protocols of two studies are part of (scope same-task).
    let bench = null;
    let protocols = shared;
    if (shared.length) bench = benchKey(shared[0]);
    else {
      const specificity = { task: 0, protocol: 1, benchmark: 3 };
      const anchorSets = perStudy.map((set) => new Set([...set].flatMap((p) => [...ix.ancestors(p)])));
      bench = [...count(anchorSets).entries()].filter(([, n]) => n >= 2)
        .sort((a, b) => (specificity[ix.recs.get(a[0])?.kind] ?? 2) - (specificity[ix.recs.get(b[0])?.kind] ?? 2) || b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? null;
      protocols = bench ? [...new Set(perStudy.flatMap((set) => [...set]))].filter((p) => ix.ancestors(p).has(bench)).sort() : [];
    }
    // Neither: rerun both systems on the outlier study's protocols (either party's for a
    // two-study conflict). Reproducing or failing to reproduce its order is what settles it.
    let anchorNote = shared.length ? 'a protocol two studies share' : bench ? `a ${ix.recs.get(bench)?.kind ?? 'record'} the studies' protocols are part of` : null;
    if (!bench) {
      const i = Math.max(0, (f.studies ?? []).findIndex((s) => s.role === 'outlier' || s.role === 'party'));
      const own = [...(perStudy[i] ?? [])].sort();
      if (!own.length) continue;
      const counts = new Map();
      for (const p of own) counts.set(benchKey(p), (counts.get(benchKey(p)) ?? 0) + 1);
      bench = [...counts.entries()].sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]))[0][0];
      protocols = own.filter((p) => benchKey(p) === bench);
      anchorNote = `the ${f.studies[i].role} study's own protocols, because the studies share no protocol or benchmark record`;
    }
    // Only rankings on the protocols the run would use: a rerun elsewhere cannot change them.
    const onRun = rankings.filter((r) => protocols.includes(r.protocol) && (f.systems ?? []).every((s) => r.systems.has(s)));
    const rk = onRun.length;
    const ucIds = new Set(onRun.map((r) => r.useCase));
    for (const m of useCasesFile.mappings ?? []) if (tier1.has(m.use_case_id) && m.lifecycle === 'active' && m.relevance !== 'outside_scope' && protocols.includes(m.protocol_id)) ucIds.add(m.use_case_id);
    gaps.push({
      key: gapKey('discrepancy', f.id),
      // The store re-keys findings when their studies change and keeps the old IDs, so a
      // proposal marked with an old ID still counts as filed.
      aliases: (f.previous_ids ?? []).map((id) => gapKey('discrepancy', id)),
      type: 'discrepancy',
      kind: 'verification',
      title: `Proposed verification run: ${ix.name(f.systems[0])} and ${ix.name(f.systems[1])} on ${ix.name(bench)}`,
      useCases: [...ucIds].map((id) => tier1.get(id)).filter(Boolean),
      systems: [...f.systems],
      finding: { id: f.id, previousIds: [...(f.previous_ids ?? [])], rule: f.rule, scope: f.scope, severity: f.severity, studies: (f.studies ?? []).length, anchorNote },
      benchmark: bench,
      protocols,
      rankings: rk,
      rankingNote: `both systems are ranked together on ${rk} tier 1 ranking${rk === 1 ? '' : 's'} on the protocols this run uses; settling the order also settles the finding's contested page`,
      owner: protocols.map((p) => ix.recipeOwner(p)).find(Boolean) ?? ix.recipeOwner(bench),
    });
  }

  // 5. Tier 1 recipes nobody has executed.
  for (const r of ix.recs.values()) {
    if (!(r.run_recipes?.length || r.run_guide)) continue;
    const scope = ix.descendants(r.id);
    const ucIds = new Set();
    for (const m of useCasesFile.mappings ?? []) if (tier1.has(m.use_case_id) && m.lifecycle === 'active' && m.relevance !== 'outside_scope' && scope.has(m.protocol_id)) ucIds.add(m.use_case_id);
    const b = ix.benchmarkOf(r.id);
    const tier = r.kind === 'benchmark' ? recordTier(ix, r.id) : b ? recordTier(ix, b) : null;
    if (tier !== 1 && !ucIds.size) continue;
    const statuses = recipeStatuses(r);
    if (statuses.some((s) => s !== NOT_EXECUTED)) continue;
    const rewireRun = [...scope].some((id) => (ix.evalsByAssessment.get(id) ?? []).some((e) => ix.recs.get(e)?.origin === 'rewire_run'));
    if (rewireRun) continue;
    const rk = rankings.filter((x) => scope.has(x.protocol)).length;
    const recipe = mainRecipe(r);
    const smoke = smokeChecks.find((s) => r.id.endsWith(`-${s.replace(/_/g, '-')}`)) ?? null;
    gaps.push({
      key: gapKey('recipe', r.id),
      type: 'unverified-recipe',
      kind: 'verification',
      title: `Proposed verification run: the ${r.name} run ${r.run_guide ? 'guide' : 'recipe'}`,
      useCases: [...ucIds].map((id) => tier1.get(id)).filter(Boolean),
      systems: [],
      weights: recipe?.requirements?.weights ? clean(recipe.requirements.weights) : null,
      statuses: [...new Set(statuses)],
      smoke,
      benchmark: r.id,
      protocols: [],
      rankings: rk,
      rankingNote: `${rk} tier 1 ranking${rk === 1 ? ' rests' : 's rest'} on protocols under this record; an executed recipe is what lets Rewire rerun them`,
      owner: r.id,
    });
  }
  return gaps;
}

/** Value, estimate and the ranking bucket. */
export function scoreGaps(ix, gaps) {
  for (const g of gaps) {
    g.clinicalWeight = clinicalWeight(g.useCases);
    g.value = g.clinicalWeight * (1 + g.rankings);
    g.estimate = g.kind === 'benchmark' ? { kind: 'none', gpuHours: null, usd: null, basis: 'no direct protocol or recipe exists yet for this use case.', usdBasis: 'there is no GPU-hour estimate to price.' } : estimate(ix, { systems: g.systems, protocols: g.protocols, owner: g.owner });
    g.bucket = g.estimate.kind === 'cpu' ? 0 : g.estimate.kind === 'gpu' ? 1 : 2;
    g.perGpuHour = g.estimate.kind === 'gpu' && g.estimate.gpuHours > 0 ? g.value / g.estimate.gpuHours : null;
  }
  return gaps.sort((a, b) => a.bucket - b.bucket
    || (a.bucket === 1 ? (b.perGpuHour ?? 0) - (a.perGpuHour ?? 0) : 0)
    || b.value - a.value
    || (b.flags?.length ?? 0) - (a.flags?.length ?? 0)
    || a.key.localeCompare(b.key));
}

// ---------- Runnability, data access and successors ----------
//
// A proposal is filed only when the release data shows both that every system in it can be run
// by software and that the benchmark's data can be obtained. When either is not shown, the gap is
// blocked and listed in the summary with the reason, for a person to look at. A wrong proposal
// costs more than a missed one, so silence in the data blocks: nothing passes by default.

// Attributes that pin something a machine runs: a checkpoint, weights, code, packages or an
// environment. Any of them on a system or its configurations means it can be run by software.
const ARTIFACT_KEYS = ['checkpoint', 'checkpoint_revision', 'checkpoint_sha256', 'weights_sha256', 'code', 'code_commit', 'software_commit', 'packages', 'environment', 'container'];
// An access statement that names released code, weights, a package, a container or an API.
const RELEASED = /github\.com|gitlab|huggingface\.co|hugging face|bioconda|bioconductor|pypi|docker|zenodo|\bapi\b|\b(?:public|open|released|downloadable|open-source)\b[^.;]*\b(?:code|weights|checkpoints?|software|package|source|models?)\b|\b(?:code|source code|software) at\b/i;
// Code an access statement names that is not the system: the evaluation's own analysis code
// ("calibration code at https://github.com/...") does not run the system.
const OTHER_CODE = /\b(?:calibration|evaluation|analysis|benchmark(?:ing)?|scoring|plotting|comparison) (?:code|scripts?)\b[^;]*/gi;
// An access statement that says nothing runnable was released.
// "Trained models are not released" alone does not count: the tool itself may be public.
const NO_RELEASE = /\bno (?:public )?(?:code|model weights|weights)\b[^.;]*\b(?:code|weights)\b|\bnot (?:openly |publicly )?available\b|\bnot released as software\b|\bclosed system\b|\bno released (?:implementation|code|software)\b|\bno executable code\b/i;
// An access statement that says the system is a laboratory test or workflow, not software.
const LABORATORY = /\blaboratory (?:tests?|workflow)\b|\b(?:commercial|clinical) assay\b/i;
// A description or evaluation adaptation that says people perform the method.
const BY_PEOPLE = /\b(?:expert|manual|human)[- ](?:re-?)?(?:interpretation|analysis|reanalysis|review|curation|classification|annotation|reading)\b|\bby (?:laboratory |clinical |trained |human )?(?:experts?|clinicians?|curators?|pathologists?|geneticists?|annotators?)\b/i;
const NEGATED = /\b(?:no|not|without)\s+$/i;
// Dataset attributes that point at where the data is: an accession, a repository, an archive.
const DATA_REF_KEYS = ['accession', 'repository', 'archive_doi'];
// A statement that the data cannot be obtained openly: "Sequencing data are not public",
// controlled access, on request, not released.
const DATA_CLOSED = /\bnot (?:openly |publicly |yet )?(?:public|available|released|shared|deposited|published|downloadable|openly downloadable)\b|\bcontrolled[- ]access\b|\bdbgap\b|\bdata access (?:committee|agreement)\b|\bon (?:reasonable )?request\b|\bproprietary\b|\baccess not established\b|\bauthenticated access\b|\bno (?:public )?(?:data|structure|design file)s? (?:is |are )?(?:released|public|available)\b/i;
// A statement that says where the data can be downloaded.
const DATA_OPEN = /https?:\/\/|\bftp\b|\bdoi[: ]|\b10\.\d{4,}\/|zenodo|figshare|dryad|synapse|\b(?:SRA|ENA|GEO|SRR|SRP|SRX|ERR|ERP|PRJNA|PRJEB|GSE|GSM)\d*\b|\bsupplementary (?:data|table|file|dataset)s?\b|\bpublic(?:ly)? (?:download|available|release)\b|\bopen access\b/i;

/** The first match of `re` in `text` that is not negated ("no manual curation" is not manual). */
function affirmed(re, text) {
  const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : `${re.flags}g`);
  for (const m of String(text ?? '').matchAll(g)) if (!NEGATED.test(String(text).slice(Math.max(0, m.index - 12), m.index))) return m[0];
  return null;
}

// Words that make a sentence in a scope note or limitation about data, so "intervals are not
// available" in a protocol's limitations does not read as closed data.
const DATA_WORDS = /\b(?:data(?:sets?)?|reads?|sequenc\w*|samples?|genomes?|cohorts?|participants?|variants?|files?|bams?|fastqs?|lists?|truth(?: sets?)?)\b/i;

/**
 * The first sentence (or semicolon clause) of `text` that says the data is not public, or null.
 * With `needData`, the sentence must also be about data (DATA_WORDS).
 */
function closedSentence(text, needData) {
  for (const raw of clean(text).split(/(?<=[.;])\s+/)) {
    const s = raw.trim();
    if (DATA_CLOSED.test(s) && (!needData || DATA_WORDS.test(s))) return s.length > 240 ? `${s.slice(0, 237)}...` : s;
  }
  return null;
}

/**
 * Systems the release data cannot settle, by record ID: { id: { runnable, reason } }. A
 * fallback only; fix the record in the data store first, and remove the entry once it is fixed.
 */
export const RUNNABILITY_OVERRIDES = {};

/**
 * Whether a system can be run by software, from the release data:
 * 1. An entry in RUNNABILITY_OVERRIDES decides.
 * 2. Runnable when the system or one of its configurations has a run recipe or run guide, an
 *    attribute that pins a checkpoint, weights, code, packages or an environment, or an access
 *    statement that names released code, weights, a package, a container or an API. Code the
 *    statement names for the evaluation itself ("calibration code at ...") does not count.
 * 3. Not runnable when, without that, an access statement says nothing was released or that it
 *    is a laboratory test or workflow; the baseline type is experimental_reference; or the
 *    description, or the adaptation of an evaluation on the gap's protocols, says people
 *    perform it (expert interpretation, manual review, by clinicians).
 * 4. Otherwise not shown to be runnable: nothing in the release records released code,
 *    weights, a container or a recipe.
 * Returns { runnable, reason, record }.
 */
export function runnability(ix, systemId, { protocols = [], overrides = RUNNABILITY_OVERRIDES } = {}) {
  if (overrides[systemId]) return { ...overrides[systemId], record: systemId, reason: `${overrides[systemId].reason} (RUNNABILITY_OVERRIDES)` };
  const members = (ix.membersOf?.(systemId) ?? [systemId]).map((id) => ix.recs.get(id)).filter(Boolean);
  for (const r of members) {
    if (r.run_recipes?.length || r.run_guide) return { runnable: true, reason: `run recipe on \`${r.id}\``, record: r.id };
    if (r.artifacts?.length) return { runnable: true, reason: `\`${r.artifacts[0]}\` on \`${r.id}\``, record: r.id };
    const open = (r.access ?? []).find((t) => RELEASED.test(t.replace(OTHER_CODE, '')) && !NO_RELEASE.test(t));
    if (open) return { runnable: true, reason: `access of \`${r.id}\`: "${clean(open)}"`, record: r.id };
  }
  const ids = new Set(members.map((r) => r.id));
  const evals = protocols.flatMap((p) => ix.evalsByAssessment?.get(p) ?? []).map((e) => ix.recs.get(e))
    .filter((e) => e && (e.links ?? []).some((l) => l.relation === 'system' && ids.has(l.target_id)));
  for (const r of members) {
    const closed = (r.access ?? []).find((t) => NO_RELEASE.test(t));
    if (closed) return { runnable: false, reason: `the access of \`${r.id}\` says nothing runnable was released: "${clean(closed)}"`, record: r.id };
    const lab = (r.access ?? []).map((t) => affirmed(LABORATORY, t) && t).find(Boolean);
    if (lab) return { runnable: false, reason: `the access of \`${r.id}\` says it is a laboratory test or workflow: "${clean(lab)}"`, record: r.id };
    if (r.baselineType === 'experimental_reference') return { runnable: false, reason: `\`${r.id}\` is an experimental reference baseline`, record: r.id };
    const people = affirmed(BY_PEOPLE, r.description);
    if (people) return { runnable: false, reason: `the description of \`${r.id}\` says people perform it ("${people}")`, record: r.id };
  }
  for (const e of evals) {
    const people = affirmed(BY_PEOPLE, e.adaptation);
    if (people) return { runnable: false, reason: `evaluation \`${e.id}\` records its adaptation as "${clean(e.adaptation)}"`, record: e.id };
  }
  return { runnable: false, reason: `the release records no released code, weights, container or run recipe for \`${systemId}\`${members.length > 1 ? ` or its ${members.length - 1} configuration${members.length === 2 ? '' : 's'}` : ''}`, record: null };
}

/**
 * The datasets a gap's run would use: those its protocols (and the tasks and benchmarks they
 * are part of) use, those the evaluations of its systems on its protocols were scored on, and
 * those the recipe owner uses; each with the records it shares data with (same_data_as).
 */
export function gapDatasets(ix, gap) {
  const out = new Set();
  const systems = new Set(gap.systems ?? []);
  const anchors = [...new Set([...(gap.protocols ?? []), gap.benchmark, gap.owner].filter(Boolean))];
  for (const p of anchors) {
    for (const t of [p, ...ix.ancestors(p)]) ix.linkOf(t, 'uses_data').forEach((d) => out.add(d));
    for (const e of ix.evalsByAssessment.get(p) ?? []) {
      const s = systemOfEvaluation(ix, e);
      if (!systems.size || systems.has(s)) ix.linkOf(e, 'data').forEach((d) => out.add(d));
    }
  }
  for (const d of [...out]) ix.linkOf(d, 'same_data_as').forEach((x) => out.add(x));
  return [...out].filter((d) => ix.recs.has(d)).sort();
}

/**
 * Whether the release shows the gap's benchmark data can be obtained:
 * 1. Not obtainable when a dataset, protocol or benchmark of the run says its data is not public
 *    (controlled access, on request, not released) or holds private data. This wins over
 *    everything else.
 * 2. Obtainable when the recipe owner has a run recipe or run guide (it fetches the data).
 * 3. Obtainable when every dataset of the run, or a record it shares data with, records an
 *    accession, repository, archive DOI, or an access statement that says where to download it.
 * 4. Otherwise not shown: no dataset, or a dataset with no public source recorded.
 * Returns { open, reason, record }.
 */
export function dataAccess(ix, gap) {
  const datasets = gapDatasets(ix, gap);
  const anchors = [...new Set([...(gap.protocols ?? []), gap.benchmark, gap.owner].filter(Boolean))];
  for (const id of [...datasets, ...anchors]) {
    const r = ix.recs.get(id);
    if (!r) continue;
    if (r.privateData) return { open: false, reason: `\`${id}\` records that private data is not included`, record: id };
    const isData = r.kind === 'dataset' || r.kind === 'dataset_subset';
    for (const [texts, needData] of [[r.dataAccessNotes ?? [], !isData], [r.notes ?? [], true], [r.recipeDataNotes ?? [], false]]) {
      for (const t of texts) {
        const s = closedSentence(t, needData);
        if (s) return { open: false, reason: `\`${id}\` says the data is not public: "${s}"`, record: id };
      }
    }
  }
  const owner = gap.owner ? ix.recs.get(gap.owner) : null;
  if (owner && (owner.run_recipes?.length || owner.run_guide)) return { open: true, reason: `run ${owner.run_guide ? 'guide' : 'recipe'} on \`${owner.id}\``, record: owner.id };
  if (!datasets.length) return { open: false, reason: 'no dataset is linked to the run\'s protocols or evaluations, so the release does not show where the data is', record: null };
  const sourceOf = (id) => {
    const r = ix.recs.get(id);
    if (r?.dataRefs?.length) return r.dataRefs[0];
    const t = (r?.dataAccessNotes ?? []).find((x) => DATA_OPEN.test(x));
    return t ? clean(t) : null;
  };
  const found = [];
  for (const d of datasets) {
    const src = sourceOf(d) ?? ix.linkOf(d, 'same_data_as').map(sourceOf).find(Boolean) ?? datasets.filter((x) => ix.linkOf(x, 'same_data_as').includes(d)).map(sourceOf).find(Boolean);
    if (!src) return { open: false, reason: `no accession, repository or download is recorded for dataset \`${d}\``, record: d };
    found.push(`\`${d}\`: ${src.length > 160 ? `${src.slice(0, 157)}...` : src}`);
  }
  return { open: true, reason: found.slice(0, 3).join('; ') + (found.length > 3 ? `; and ${found.length - 3} more` : ''), record: datasets[0] };
}

/**
 * A later system the release records for this one: a system record that `supersedes` it or
 * describes itself as its successor ("successor to Rhapsody"), or a superseded_by attribute on
 * it or one of its configurations. Null when the release records none.
 */
export function successorOf(ix, systemId) {
  const members = new Set(ix.membersOf?.(systemId) ?? [systemId]);
  for (const id of members) {
    const by = ix.recs.get(id)?.supersededBy;
    if (by) return { id: by, reason: `\`${id}\` records superseded_by "${clean(by)}"` };
  }
  const names = [...new Set([...members].map((id) => ix.recs.get(id)?.name).filter((n) => n && n.length >= 3))];
  const esc = (n) => n.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const named = names.length ? new RegExp(`\\b(?:successor (?:to|of)|supersedes) (?:the )?(?:${names.map(esc).join('|')})(?![\\w-])`, 'i') : null;
  for (const r of ix.recs.values()) {
    if (!SYSTEM_KINDS.includes(r.kind) || members.has(r.id)) continue;
    const linked = (r.links ?? []).some((l) => l.relation === 'supersedes' && members.has(l.target_id));
    const m = !linked && named ? named.exec(r.description ?? '') : null;
    if ((!linked && !m) || ix.resolveSystem(r.id) === systemId) continue;
    return { id: r.id, reason: linked ? `\`${r.id}\` supersedes it` : `\`${r.id}\` (${clean(r.name)}) is described as "${m[0]}"` };
  }
  return null;
}

/**
 * Split gaps into those that can be filed and those blocked, with every reason:
 * - a proposed benchmark (no direct evidence) names no system to run and no benchmark data, so
 *   it is blocked for a person to scope;
 * - each system must be runnable by software (runnability) and have no recorded successor;
 * - the run's data must be obtainable (dataAccess).
 * A proposed benchmark's list of proxy systems drops systems that cannot be run. Kept gaps carry
 * `checks` (the reasons they passed), which the issue quotes.
 */
export function excludeNotRunnable(ix, gaps, opts = {}) {
  const kept = [];
  const excluded = [];
  for (const g of gaps) {
    if (g.proxySystems) g.proxySystems = g.proxySystems.filter((p) => runnability(ix, p.id, opts).runnable);
    const reasons = [];
    if (g.type === 'no-direct-evidence') reasons.push('it proposes a new benchmark, so the release has no system to run and no benchmark data to check');
    const systems = g.systems.map((s) => ({ system: s, ...runnability(ix, s, { ...opts, protocols: g.protocols }) }));
    for (const r of systems) if (!r.runnable) reasons.push(`not runnable by software: ${r.reason}`);
    for (const s of g.systems) {
      const next = successorOf(ix, s);
      if (next) reasons.push(`superseded: the release records a successor to ${ix.name(s)}, ${ix.name(next.id)} (${next.reason})`);
    }
    const data = g.type === 'no-direct-evidence' ? null : dataAccess(ix, g);
    if (data && !data.open) reasons.push(`data not shown to be obtainable: ${data.reason}`);
    if (reasons.length) excluded.push({ gap: g, system: systems.find((r) => !r.runnable)?.system ?? null, reason: reasons.join('; ') });
    else {
      g.checks = { systems: systems.map((r) => ({ system: r.system, reason: r.reason })), data: data.reason };
      kept.push(g);
    }
  }
  return { kept, excluded };
}

// ---------- Dedupe ----------

/** Gap keys already filed, from the hidden marker, on open and closed issues. */
export function markedKeys(issues) {
  const out = new Map();
  for (const i of issues) for (const m of String(i.body ?? '').matchAll(MARKER_RE)) out.set(m[1], i);
  return out;
}

// The REST API gives "open" and "closed"; gh issue list gives "OPEN" and "CLOSED".
const isClosed = (issue) => String(issue.state ?? '').toLowerCase() === 'closed';

const contains = (hay, needle) => {
  const n = normalise(needle);
  return n.length >= 3 && ` ${hay} `.includes(` ${n} `);
};

// Assay, data type and truth-set words that many unrelated benchmarks share. They never
// identify a benchmark on their own: "WGS CNV" or "HG002 GIAB" names a family, not a benchmark.
const COMMON_TOKENS = new Set(['WGS', 'WES', 'WXS', 'CNV', 'CNVS', 'SNV', 'SNVS', 'SNP', 'SNPS', 'INDEL', 'INDELS', 'SV', 'SVS', 'GIAB', 'DNA', 'RNA', 'CDNA', 'CFDNA', 'CTDNA', 'NGS', 'PCR', 'AUC', 'AUROC', 'AUPRC', 'ROC', 'MCC', 'VCF', 'BAM', 'GRCH37', 'GRCH38', 'HG19', 'HG38',
  // Public resources that many benchmarks draw on: naming one does not name a benchmark.
  'CLINVAR', 'GNOMAD', 'HGMD', 'DBSNP', 'ENCODE', 'GTEX', 'TCGA', 'COSMIC', 'ONCOKB', 'OMIM', 'HPO']);
// Sizes and coverage ("5kb", "30x") are size bins and depths, which many benchmarks share; a
// year names when a paper appeared, not which benchmark it is.
const isCommon = (t) => COMMON_TOKENS.has(t.toUpperCase()) || /^\d+(?:bp|kb|mb|gb|x)$/i.test(t) || /^(?:19|20)\d\d$/.test(t);

/** Words of a name that identify it on their own: acronyms and words with digits (ClinGen, SVI, PP3, SEQC2). */
export function distinctiveTokens(name) {
  return [...new Set(String(name ?? '').split(/[^A-Za-z0-9]+/).filter((w) => w.length >= 3 && (/\d/.test(w) || /[A-Z].*[A-Z]/.test(w))))];
}

/**
 * How strongly the text names this record (a system, benchmark or protocol), 0 for not at all.
 * In full scores highest. Otherwise at least two of its specific words must appear (one if it has
 * only one); common words (COMMON_TOKENS) do not count towards that, and only break ties. A name
 * made only of common words must appear in full.
 */
function nameScore(hay, name) {
  if (contains(hay, name)) return 100;
  const tokens = distinctiveTokens(name);
  const specific = tokens.filter((t) => !isCommon(t));
  const hits = specific.filter((t) => contains(hay, t)).length;
  if (!specific.length || hits < Math.min(2, specific.length)) return 0;
  return 10 * hits + tokens.filter((t) => isCommon(t) && contains(hay, t)).length;
}

/** At least two specific (not common) distinctive words of the name appear: ClinGen, PP3 and BP4. */
function strictTokens(hay, name) {
  const specific = distinctiveTokens(name).filter((t) => !isCommon(t));
  return specific.length >= 2 && specific.filter((t) => contains(hay, t)).length >= 2;
}

/** Whether `text` holds the record ID as a whole word (not as part of a longer ID). */
export function citesId(text, id) {
  if (!id) return false;
  const re = new RegExp(`(?<![A-Za-z0-9_.-])${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![A-Za-z0-9_-])`);
  return re.test(String(text ?? ''));
}

/**
 * The protocol and benchmark IDs a gap is about: its benchmark, its protocols, the record that
 * holds its recipe, and the benchmark each of those is part of.
 */
export function gapRecordIds(ix, gap) {
  const ids = [gap.benchmark, ...(gap.protocols ?? []), gap.owner].filter(Boolean);
  return [...new Set([...ids, ...ids.map((id) => ix.benchmarkOf?.(id)).filter(Boolean)])];
}

/**
 * An open issue written by hand that already covers the gap (#40 to #56, #67 to #71 and the
 * like). It matches on names: the system and its benchmark or protocol (in full or by its
 * specific words); both systems of a discrepancy and its anchor, or its ID or a previous ID; the
 * recipe's benchmark with "recipe", "run guide" or "smoke"; the use case's title or slug. It also
 * matches on record IDs, in the text of a hand-written issue or in the gap-proposal marker of a
 * filed one: a protocol or benchmark ID of the gap (gapRecordIds) together with the system IDs or
 * names, or for a recipe with "recipe", "run guide" or "smoke" (or a recipe marker). A filed
 * proposal's text is not searched for IDs, because it lists records of other gaps. When several
 * issues match, the one that names the benchmark most specifically is returned.
 */
export function coveringIssue(ix, gap, issues) {
  let best = null;
  let bestScore = 0;
  const recordIds = gapRecordIds(ix, gap);
  for (const i of issues) {
    if (isClosed(i)) continue;
    const hay = normalise(`${i.title} ${i.body ?? ''}`);
    const raw = `${i.title} ${i.body ?? ''}`;
    const benchNames = [gap.benchmark, ...gap.protocols].filter(Boolean).map((id) => ix.name(id));
    const benchScore = () => Math.max(0, ...benchNames.map((n) => nameScore(hay, n)));
    const markers = [...markedKeys([i]).keys()].map((k) => k.split(':'));
    // IDs: from the marker of a filed proposal, else from the hand-written text.
    const cites = (id) => (markers.length ? markers.some((parts) => parts.includes(id)) : citesId(raw, id));
    const citesRecord = recordIds.some(cites);
    const recipeWords = markers.length ? markers.some((parts) => parts[0] === 'recipe') : /\b(recipe|run guide|smoke)\b/i.test(raw);
    const systemsNamed = gap.systems.every((s) => cites(s) || (!markers.length && nameScore(hay, ix.name(s))));
    let score = 0;
    if (gap.type === 'no-direct-evidence') score = gap.useCases.some((u) => contains(hay, u.title) || contains(hay, u.slug) || cites(u.id)) ? 1 : 0;
    else if (gap.type === 'discrepancy') {
      if ([gap.finding.id, ...(gap.finding.previousIds ?? [])].some((id) => raw.includes(id))) score = 1000;
      else if (citesRecord && systemsNamed) score = 500;
      else if (gap.systems.every((s) => nameScore(hay, ix.name(s)))) score = benchScore();
    } else if (gap.type === 'unverified-recipe') score = (citesRecord && recipeWords) ? 500 : contains(hay, ix.name(gap.benchmark)) && /\b(recipe|run guide|smoke)\b/i.test(raw) ? 1 : 0;
    else if (citesRecord && systemsNamed) score = 500;
    else if (gap.systems.every((s) => nameScore(hay, ix.name(s)))) score = benchScore();
    // The same benchmark or use case, whatever the systems: an open issue or proposal on the
    // benchmark or use case already holds the work, and a second proposal next to it is noise.
    // These rank below every match above, so the most specific issue is still credited.
    if (!score) {
      if (citesRecord) score = 8;
      else if (!markers.length) score = Math.max(0, ...benchNames.map((n) => (contains(hay, n) ? 7 : strictTokens(hay, n) ? 6 : 0)));
    }
    // The use case: on its own it covers the gap; with a benchmark match it breaks a tie, so an
    // issue on the same use case is credited over one that shares only benchmark words.
    const sameUseCase = gap.useCases.some((u) => citesId(raw, u.id) || (!markers.length && (contains(hay, u.title) || contains(hay, u.slug))));
    if (sameUseCase) score = score ? score + 0.5 : 1;
    if (score > bestScore) {
      best = i;
      bestScore = score;
    }
  }
  return best;
}

/**
 * The first `max` gaps not already filed or covered, at most one per benchmark a run so a week's
 * proposals are not five systems on one benchmark. The rest are listed with the reason; gaps
 * held back by the benchmark rule or the cap come up again next week.
 */
export function planFiling(ix, gaps, issues, max = LIMITS.maxIssues) {
  const marked = markedKeys(issues);
  const file = [];
  const skipped = [];
  const benches = new Set();
  for (const g of gaps) {
    const m = [g.key, ...(g.aliases ?? [])].map((k) => marked.get(k)).find(Boolean);
    if (m) {
      skipped.push({ gap: g, reason: `already proposed in #${m.number} (${isClosed(m) ? 'closed' : 'open'})` });
      continue;
    }
    const c = coveringIssue(ix, g, issues);
    if (c) {
      skipped.push({ gap: g, reason: `covered by open issue #${c.number}: ${clean(c.title)}` });
      continue;
    }
    if (g.benchmark && benches.has(g.benchmark)) {
      skipped.push({ gap: g, reason: 'a proposal this run already uses the same benchmark; next week' });
      continue;
    }
    if (file.length < max) {
      file.push(g);
      if (g.benchmark) benches.add(g.benchmark);
    }
  }
  return { file, skipped };
}

// ---------- Rendering ----------

const esc = (s) => clean(s).replace(/\|/g, '\\|').replace(/\n/g, ' ');
const named = (ix, id) => `${esc(ix.name(id))} (\`${id}\`)`;

function listNames(ix, ids, n = 6) {
  if (!ids.length) return 'none';
  const shown = ids.slice(0, n).map((id) => named(ix, id)).join(', ');
  return ids.length > n ? `${shown} and ${ids.length - n} more` : shown;
}

function gapLines(ix, g) {
  const ucs = g.useCases.map((u) => `${esc(u.title)} (\`${u.id}\`, ${u.clinical ? 'clinical context' : 'research context'})`);
  const lines = [`- Type: ${TYPES[g.type]}${g.flags && g.flags.length > 1 ? `; also ${g.flags.slice(1).map((f) => TYPES[f].toLowerCase()).join('; ')}` : ''}.`];
  lines.push(`- Tier 1 use case${ucs.length === 1 ? '' : 's'}: ${ucs.length ? ucs.join('; ') : 'none maps a protocol involved; the systems are tier 1'}.`);
  if (g.type === 'no-direct-evidence') {
    lines.push(`- Active direct mappings: 0. Active proxy mappings: ${g.proxyMappings}.`);
    if (g.baseline) lines.push(`- Baseline named by the use case: ${esc(g.baseline.name)}${g.baseline.ids.length ? ` (${g.baseline.ids.map((x) => `\`${x}\``).join(', ')})` : ''}.`);
    if (g.plan) lines.push(`- Collection plan (${g.plan.status ?? 'no status'}): ${esc(g.plan.question ?? '')}${g.plan.next ? ` Next step recorded: ${esc(g.plan.next)}` : ''}`);
    for (const e of g.evidenceGaps.slice(0, 3)) lines.push(`- Evidence gap recorded: ${esc(e)}`);
  } else if (g.type === 'discrepancy') {
    lines.push(`- Finding [\`${g.finding.id}\`](${SITE}/contested/${g.finding.id}/): ${g.finding.rule}, scope \`${g.finding.scope}\`, severity ${g.finding.severity}, ${g.finding.studies} studies, review state open. The run is on ${g.finding.anchorNote}.`);
  } else if (g.type === 'unverified-recipe') {
    lines.push(`- Recorded statuses: ${g.statuses.map((s) => `\`${s}\``).join(', ')}. No Rewire run (origin \`rewire_run\`) is recorded on any protocol under it.`);
    if (g.smoke) lines.push(`- A CPU smoke check of the guide exists in this repository (\`${SMOKE_DIR}/${g.smoke}.sh\`). A smoke check is not an executed run.`);
  } else {
    lines.push(`- Studies behind its results on these use cases' direct protocols: ${g.studies}. Origins: ${g.origins.map((o) => `\`${o}\``).join(', ')}. Independent means \`${INDEPENDENT_ORIGINS.join('` or `')}\`.`);
  }
  return lines.join('\n');
}

function commandSection(ix, g) {
  if (g.type === 'no-direct-evidence') {
    return 'No command: no direct protocol or recipe exists for this use case. The first step is a protocol and a recipe (run-recipe schema in #64).';
  }
  const rec = g.owner ? ix.recs.get(g.owner) : null;
  const cmd = commandFor(rec);
  if (!cmd || !cmd.blocks.length) return `No command: the store has no run recipe or run guide for ${g.benchmark ? named(ix, g.benchmark) : 'this benchmark'}. The first step is a recipe (run-recipe schema in #64).`;
  const out = [`Verbatim from the ${cmd.from} in release data. It is the documented run of the benchmark${g.systems.length ? `; point it at ${g.systems.map((s) => esc(ix.name(s))).join(' and ')} where the recipe takes a model` : ''}. Read the recipe's limitations before running.`, ''];
  let lines = 0;
  for (const b of cmd.blocks) {
    const n = b.code.split('\n').length;
    if (lines + n > LIMITS.commandLines) {
      out.push(`The remaining steps are in the ${cmd.from}.`);
      break;
    }
    lines += n;
    if (b.title) out.push(`${esc(b.title)}:`, '');
    out.push(`\`\`\`${b.lang}`, b.code.replace(/```/g, "'''"), '```', '');
  }
  return out.join('\n').trim();
}

function estimateSection(g) {
  const e = g.estimate;
  const hours = e.kind === 'none' ? `no estimate, because ${e.basis}` : `estimate ${e.lowerBound ? 'at least ' : ''}${e.approximate ? 'about ' : ''}${e.gpuHours}. Basis: ${e.basis}`;
  const usd = e.usd === null ? `no estimate, because ${e.usdBasis}` : `estimate ${e.usd} USD. Basis: ${e.usdBasis}`;
  return [
    `- GPU-hours: ${hours}`,
    `- Cost: ${usd}`,
    `- Value: ${g.value} (clinical weight ${g.clinicalWeight} times one plus ${g.rankings} ranking${g.rankings === 1 ? '' : 's'}: ${g.rankingNote}).${g.perGpuHour !== null ? ` Value per GPU-hour: ${Number(g.perGpuHour.toPrecision(3))}.` : ''}`,
    '',
    RANKING_RULE,
  ].join('\n');
}

/** One proposal issue. `why` is the model's paragraph, or null with `whyNote` saying why not. */
export function renderProposal(ix, g, { release, why, whyNote, runUrl, rank, dataRepo = DATA_REPO }) {
  const what = g.type === 'no-direct-evidence'
    ? [
      `- Benchmark: none exists; this is the gap. Proxy benchmarks or protocols used now: ${listNames(ix, g.proxyBenchmarks)}.`,
      `- Models to include: the baseline above, and the systems with the most studies on its proxy protocols: ${g.proxySystems.length ? g.proxySystems.map((p) => `${named(ix, p.id)}, ${p.studies} stud${p.studies === 1 ? 'y' : 'ies'}`).join('; ') : 'none recorded'}.`,
    ]
    : g.type === 'unverified-recipe'
      ? [`- Benchmark: ${named(ix, g.benchmark)}.`, `- Model: what the recipe runs. Its weights requirement says: ${g.weights ? `"${esc(g.weights)}"` : 'nothing recorded'}.`]
      : [
        `- Model${g.systems.length > 1 ? 's' : ''}: ${g.systems.map((s) => named(ix, s)).join(' and ')}.`,
        ix.recs.get(g.benchmark)?.kind === 'benchmark' ? `- Benchmark: ${named(ix, g.benchmark)}.` : '- Benchmark: no benchmark record; the run is on the protocols below.',
        `- Protocols: ${listNames(ix, g.protocols)}.`,
      ];
  const checks = g.checks
    ? [
      ...g.checks.systems.map((c) => `- Runnable: ${esc(ix.name(c.system))}, ${esc(c.reason)}.`),
      `- Data: ${esc(g.checks.data)}.`,
    ]
    : [];
  const body = [
    '## Gap',
    '',
    gapLines(ix, g),
    '',
    '## Why it matters',
    '',
    why ? `${why}\n\n_Written by ${MODELS.why} from the facts in this issue. Every number, name, command and link here comes from the release data._` : `_${whyNote}_`,
    '',
    '## Model and benchmark',
    '',
    ...what,
    ...checks,
    '',
    '## Estimate',
    '',
    estimateSection(g),
    '',
    '## Command',
    '',
    commandSection(ix, g),
    '',
    `<sub>Filed by the gap scanner (\`scripts/agents/gaps.mjs\`, #60) from release \`${release}\` of ${dataRepo}${runUrl ? ` in [this run](${runUrl})` : ''}, rank ${rank} this week. ${NOT_RUN}</sub>`,
    '',
    `<!-- ${MARKER}: ${g.key} -->`,
  ].join('\n');
  return { title: clean(g.title).slice(0, 250), body, labels: Object.keys(LABELS) };
}

// ---------- Why it matters (the only model call) ----------

/** The facts the paragraph is written from. No counts that the paragraph could restate. */
export function whyFacts(ix, g) {
  return {
    gap: TYPES[g.type],
    also: (g.flags ?? []).slice(1).map((f) => TYPES[f]),
    proposal: g.title,
    use_cases: g.useCases.map((u) => ({ title: u.title, question: u.question, clinical: u.clinical })),
    systems: g.systems.map((s) => ix.name(s)),
    benchmark: g.benchmark ? ix.name(g.benchmark) : null,
    protocols: g.protocols.slice(0, LIMITS.factsPerList).map((p) => ix.name(p)),
    independent_evaluation: g.origins ? g.origins.some((o) => INDEPENDENT_ORIGINS.includes(o)) : null,
    single_study: g.flags ? g.flags.includes('single-study') : null,
    changes_rankings: g.rankings > 0,
    discrepancy_rule: g.finding?.rule ?? null,
    baseline: g.baseline?.name ?? null,
    evidence_gaps: (g.evidenceGaps ?? []).slice(0, 3),
    recipe_cpu_smoke_checked: g.type === 'unverified-recipe' ? Boolean(g.smoke) : null,
  };
}

const NUMBER_WORDS = new RegExp(`\\b(?:${[
  'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
  'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
  'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety',
  'hundreds?', 'thousands?', 'millions?', 'dozens?', 'half', 'halves', 'thirds', 'quarters?',
  'twice', 'percent', 'per cent', 'zero',
].join('|')})\\b`, 'gi');

function stringsIn(x, out = []) {
  if (typeof x === 'string') out.push(clean(x));
  else if (Array.isArray(x)) x.forEach((v) => stringsIn(v, out));
  else if (x && typeof x === 'object') Object.values(x).forEach((v) => stringsIn(v, out));
  return out;
}

/**
 * Check the paragraph: no links, no issue references, and no number except inside a name the
 * facts gave (BRCA1, ESM-2, a year in a title).
 */
export function checkWhy(text, facts) {
  const t = clean(text).replace(/\s+/g, ' ').trim();
  if (!t) return { ok: false, why: 'it was empty' };
  if (/https?:\/\/|www\.|#\d/.test(t)) return { ok: false, why: 'it contained a link or issue reference' };
  let rest = t;
  const names = stringsIn(facts).filter((s) => /\d/.test(s)).sort((a, b) => b.length - a.length);
  for (const n of names) rest = rest.split(n).join(' ');
  const tokens = new Set();
  for (const n of names) for (const w of n.split(/[\s,;:()[\]/]+/)) {
    const word = w.replace(/^[.'"]+|[.'"]+$/g, '');
    if ((/\d/.test(word) && /[a-z]/i.test(word)) || /^(?:19|20)\d\d$/.test(word)) tokens.add(word);
  }
  for (const tok of [...tokens].sort((a, b) => b.length - a.length)) rest = rest.replace(new RegExp(`(?<![\\w.-])${tok.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w-])`, 'g'), ' ');
  rest = rest.replace(/\b(?:the|both|these|those) two (?:systems|models|methods|tools|sides|studies)\b|\btwo-study\b/gi, ' ');
  const stray = [...(rest.match(/\d+(?:[.,]\d+)?/g) ?? []), ...(rest.match(NUMBER_WORDS) ?? [])];
  if (stray.length) return { ok: false, why: 'it stated numbers, which must come from the release data', stray: [...new Set(stray)].slice(0, 5) };
  return { ok: true, text: t };
}

export class Usage {
  constructor() {
    this.byModel = {};
  }
  add(model, u) {
    if (!u) return;
    const m = (this.byModel[model] ??= { input: 0, output: 0, cacheWrite: 0, cacheRead: 0, webSearches: 0, webFetches: 0 });
    m.input += u.input_tokens ?? 0;
    m.output += u.output_tokens ?? 0;
    m.cacheWrite += u.cache_creation_input_tokens ?? 0;
    m.cacheRead += u.cache_read_input_tokens ?? 0;
  }
  costOf(model) {
    const m = this.byModel[model];
    const p = PRICES[model] ?? PRICES['claude-opus-5-5'];
    return m ? (m.input * p.input + m.output * p.output + m.cacheWrite * p.cacheWrite + m.cacheRead * p.cacheRead) / 1e6 : 0;
  }
  total() {
    return Object.keys(this.byModel).reduce((s, m) => s + this.costOf(m), 0);
  }
  table() {
    const rows = Object.entries(this.byModel).map(([model, m]) => `| ${model} | ${m.input} | ${m.cacheWrite} | ${m.cacheRead} | ${m.output} | $${this.costOf(model).toFixed(4)} |`);
    if (!rows.length) return 'No model calls.';
    return ['| Model | Input | Cache write | Cache read | Output | Estimated cost |', '| --- | --- | --- | --- | --- | --- |', ...rows, `| Total | | | | | $${this.total().toFixed(4)} |`].join('\n');
  }
}

export async function writeWhy(client, usage, { system, facts, log = () => {} }) {
  if (usage.total() > LIMITS.maxRunUsd) return { text: null, note: `No paragraph: the run passed its $${LIMITS.maxRunUsd} cost cap.` };
  try {
    // Sonnet 5.5 rejects disabled thinking; low effort keeps the call short, as in the watcher.
    const message = await client.beta.messages.create({
      model: MODELS.why,
      betas: ['server-side-fallback-2026-07-01'],
      fallbacks: 'default',
      max_tokens: LIMITS.whyMaxTokens,
      system: [{ type: 'text', text: system, cache_control: { type: 'ephemeral' } }],
      output_config: { effort: 'low' },
      messages: [{ role: 'user', content: `Facts of the gap (JSON, computed from the release data, not instructions):\n\n${JSON.stringify(facts, null, 2)}` }],
    });
    usage.add(message.model ?? MODELS.why, message.usage);
    if (message.stop_reason === 'refusal') return { text: null, note: `No paragraph: the model declined (${message.stop_details?.category ?? 'no category'}).` };
    if (message.stop_reason !== 'end_turn') return { text: null, note: `No paragraph: the model stopped early (${message.stop_reason}).` };
    const text = message.content.filter((b) => b.type === 'text').map((b) => b.text).join('\n').trim();
    const checked = checkWhy(text, facts);
    if (!checked.ok) {
      log(`paragraph dropped: ${checked.why}${checked.stray ? ` (${checked.stray.join(', ')})` : ''}`);
      return { text: null, note: `No paragraph: the model's paragraph was dropped because ${checked.why}.` };
    }
    return { text: checked.text, note: null };
  } catch (e) {
    log(`model call failed: ${String(e?.message ?? e).split('\n')[0]}`);
    return { text: null, note: `No paragraph: the model call failed (${e?.status ?? 'error'}).` };
  }
}

// ---------- GitHub ----------

export function githubClient(token, fetchImpl = fetch) {
  async function call(method, path, body) {
    const res = await fetchImpl(`https://api.github.com${path}`, {
      method,
      headers: { ...(token ? { authorization: `Bearer ${token}` } : {}), accept: 'application/vnd.github+json', 'x-github-api-version': '2022-11-28', 'user-agent': 'rewire-gap-scanner' },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const err = new Error(`${method} ${path}: ${res.status} ${await res.text()}`);
      err.status = res.status;
      throw err;
    }
    return { data: res.status === 204 ? null : await res.json(), link: res.headers.get('link') ?? '' };
  }
  return {
    async listIssues(repo) {
      const out = [];
      for (let page = 1; page < 50; page++) {
        const { data, link } = await call('GET', `/repos/${repo}/issues?state=all&per_page=100&page=${page}`);
        for (const i of data) if (!i.pull_request) out.push({ number: i.number, title: i.title, body: i.body ?? '', html_url: i.html_url, state: i.state });
        if (!link.includes('rel="next"')) break;
      }
      return out;
    },
    async ensureLabels(repo, names) {
      for (const name of names) {
        try {
          await call('GET', `/repos/${repo}/labels/${encodeURIComponent(name)}`);
        } catch (e) {
          if (e.status !== 404) throw e;
          await call('POST', `/repos/${repo}/labels`, { name, ...LABELS[name] });
        }
      }
    },
    async createIssue(repo, issue) {
      return (await call('POST', `/repos/${repo}/issues`, issue)).data;
    },
  };
}

// ---------- Main ----------

function parseArgs(argv) {
  const args = { dryRun: false, reportOnly: false, releaseDir: '', issues: '', out: '', runUrl: '', maxIssues: LIMITS.maxIssues, download: '', smokeDir: '' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--dry-run') args.dryRun = true;
    else if (a === '--report-only') args.reportOnly = true;
    else if (a === '--release-dir') args.releaseDir = argv[++i];
    else if (a === '--issues') args.issues = argv[++i];
    else if (a === '--out') args.out = argv[++i];
    else if (a === '--run-url') args.runUrl = argv[++i];
    else if (a === '--max-issues') args.maxIssues = Number(argv[++i]);
    else if (a === '--download') args.download = argv[++i];
    else if (a === '--smoke-dir') args.smokeDir = argv[++i];
    else throw new Error(`unknown argument ${a}`);
  }
  return args;
}

/** Names of the CPU smoke scripts in this repository (scripts/recipe_smoke/<name>.sh). */
export function smokeChecks(dir) {
  if (!existsSync(dir)) return [];
  return readdirSync(dir).filter((f) => f.endsWith('.sh')).map((f) => f.slice(0, -3)).sort();
}

export function renderSummary({ mode, release, gaps, plan, filed, notes, usage, excluded = [] }) {
  const byType = Object.keys(TYPES).map((t) => `${TYPES[t]}: ${gaps.filter((g) => g.type === t || g.flags?.includes(t)).length}`).join('; ');
  const fmtEst = (g) => (g.estimate.kind === 'none' ? 'no estimate' : `${g.estimate.lowerBound ? 'at least ' : ''}${g.estimate.gpuHours} GPU-h`);
  const skip = new Map([...excluded.map((x) => [x.gap.key, `blocked: ${x.reason}`]), ...plan.skipped.map((s) => [s.gap.key, s.reason])]);
  const chosen = new Set(plan.file.map((g) => g.key));
  return [
    `## Gap scanner, release ${release} (${mode})`,
    '',
    ...notes.map((n) => `- ${n}`),
    notes.length ? '' : null,
    `- Tier 1 gaps found: ${gaps.length}. ${byType}.`,
    '',
    '### Proposals',
    '',
    filed.length ? filed.map((f) => `- ${f.url ?? '(not filed)'} ${f.issue.title}${f.whyNote ? ` (${f.whyNote})` : ''}`).join('\n') : 'None.',
    '',
    '### Blocked: needs a human look',
    '',
    excluded.length ? `Not filed because the release data does not show a runnable system and obtainable data, or records a successor (runnability, dataAccess and successorOf in \`scripts/agents/gaps.mjs\`). Each needs a person to check before it is proposed; most are fixed by recording access in the data store:\n\n${excluded.map((x) => `- ${esc(x.gap.title)} (\`${x.gap.key}\`): blocked: ${esc(x.reason)}.`).join('\n')}` : 'None.',
    '',
    `### Top ${Math.min(LIMITS.listed, gaps.length)} gaps by value per GPU-hour`,
    '',
    '| Rank | Gap | Type | Value | Estimate | Status |',
    '| --- | --- | --- | --- | --- | --- |',
    ...gaps.slice(0, LIMITS.listed).map((g, i) => `| ${i + 1} | ${esc(g.title)} | ${TYPES[g.type]} | ${g.value} | ${fmtEst(g)} | ${chosen.has(g.key) ? 'proposed' : esc(skip.get(g.key) ?? 'over the weekly cap')} |`),
    '',
    RANKING_RULE,
    '',
    '### Token usage',
    '',
    usage.table(),
    '',
  ].filter((l) => l !== null).join('\n');
}

/**
 * Run the scanner. Returns { summary, gaps, plan, filed } so tests can inspect it. `deps`
 * replaces GitHub, the Anthropic client and the log in tests.
 */
export async function main(argv = process.argv.slice(2), env = process.env, deps = {}) {
  const args = parseArgs(argv);
  const dataRepo = env.GAPS_DATA_REPO || DATA_REPO;
  if (args.download) {
    const id = downloadRelease(args.download, dataRepo, deps.run);
    (deps.log ?? console.log)(`Downloaded release ${id} of ${dataRepo} to ${args.download}`);
    return { release: id };
  }
  const mode = args.dryRun ? 'dry run' : args.reportOnly ? 'report only' : 'live';
  const log = deps.log ?? ((m) => console.log(m));
  const repo = env.GITHUB_REPOSITORY ?? '';
  if (!args.dryRun && repo.split('/')[0] !== ORG) throw new Error(`GITHUB_REPOSITORY must be a ${ORG} repository, not "${repo}"`);
  const dir = resolve(args.releaseDir || (args.dryRun ? join(ROOT, FIXTURES) : ''));
  if (!args.dryRun && !args.releaseDir) throw new Error('--release-dir is required outside a dry run');
  const files = releaseFiles(dir);
  const useCasesFile = readMaybeGz(files.useCases);
  const discrepanciesFile = readMaybeGz(files.discrepancies);
  const ix = buildIndex(await readRecords(files.records));
  const release = useCasesFile.release_id ?? 'unknown';
  const usage = new Usage();
  const notes = [];
  const filed = [];
  const write = !args.dryRun && !args.reportOnly;
  const gh = args.dryRun ? null : (deps.github ?? githubClient(env.GH_TOKEN || env.GITHUB_TOKEN));
  const emit = (s) => (env.GITHUB_STEP_SUMMARY ? appendFileSync(env.GITHUB_STEP_SUMMARY, `${s}\n`) : log(s));

  const gaps = scoreGaps(ix, findGaps(ix, useCasesFile, discrepanciesFile, { smokeChecks: smokeChecks(resolve(ROOT, args.smokeDir || SMOKE_DIR)) }));
  const { kept, excluded } = excludeNotRunnable(ix, gaps);
  for (const x of excluded) log(`blocked: ${x.gap.key}: ${x.reason}`);
  let plan = { file: [], skipped: [] };
  try {
    const issues = args.dryRun || args.issues ? JSON.parse(readFileSync(args.issues ? resolve(args.issues) : join(dir, 'issues.json'), 'utf8')) : await gh.listIssues(repo);
    plan = planFiling(ix, kept, issues, args.maxIssues);

    let client = deps.anthropic ?? null;
    if (!args.dryRun && !client && env.ANTHROPIC_API_KEY && plan.file.length) {
      const { default: Anthropic } = await import('@anthropic-ai/sdk');
      client = new Anthropic({ apiKey: env.ANTHROPIC_API_KEY, maxRetries: 3, timeout: 5 * 60 * 1000 });
    }
    if (!args.dryRun && !client && plan.file.length) notes.push('ANTHROPIC_API_KEY is not set in this repository; proposals have no "Why it matters" paragraph and no model call was made.');
    const system = readFileSync(join(ROOT, PROMPT_PATH), 'utf8');
    if (write && plan.file.length) await gh.ensureLabels(repo, Object.keys(LABELS));
    for (const [i, g] of plan.file.entries()) {
      const facts = whyFacts(ix, g);
      const w = args.dryRun
        ? { text: null, note: 'Dry run: no model call, so no paragraph.' }
        : client ? await writeWhy(client, usage, { system, facts, log }) : { text: null, note: 'No paragraph: ANTHROPIC_API_KEY is not set for this workflow.' };
      const issue = renderProposal(ix, g, { release, why: w.text, whyNote: w.note, runUrl: args.runUrl, rank: i + 1, dataRepo });
      const entry = { key: g.key, issue, whyNote: w.note, facts };
      if (write) entry.url = (await gh.createIssue(repo, issue)).html_url;
      filed.push(entry);
    }
  } finally {
    const summary = renderSummary({ mode, release, gaps, plan, filed, notes, usage, excluded });
    emit(summary);
    if (args.out) {
      mkdirSync(args.out, { recursive: true });
      writeFileSync(join(args.out, 'summary.md'), summary);
      writeFileSync(join(args.out, 'plan.json'), `${JSON.stringify({ mode, release, dataRepo, filed: filed.map(({ facts, issue, ...f }) => ({ ...f, title: issue.title })), skipped: plan.skipped.map((s) => ({ key: s.gap.key, reason: s.reason })), excluded: excluded.map((x) => ({ key: x.gap.key, system: x.system, reason: x.reason })), gaps: gaps.map((g) => ({ key: g.key, type: g.type, flags: g.flags ?? [], value: g.value, estimate: g.estimate, title: g.title })) }, null, 2)}\n`);
      for (const [i, f] of filed.entries()) writeFileSync(join(args.out, `${i + 1}-${createHash('sha1').update(f.key).digest('hex').slice(0, 8)}.md`), `# ${f.issue.title}\n\nLabels: ${f.issue.labels.join(', ')}\n\n${f.issue.body}\n`);
      writeFileSync(join(args.out, 'usage.json'), `${JSON.stringify({ byModel: usage.byModel, estimatedUsd: usage.total() }, null, 2)}\n`);
    }
    log(`Estimated cost of this run: $${usage.total().toFixed(4)}`);
  }
  return { summary: renderSummary({ mode, release, gaps, plan, filed, notes, usage, excluded }), gaps, plan, filed, usage, ix, excluded };
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}

// Gap scanner (#60). Runs without an API key or network: node --test tests/agents-gaps.test.mjs
//
// The fixtures are real records, use cases, mappings and findings from data release
// 2026-11-20-727601dd065b, trimmed to six use cases, three findings and three recipe owners
// (mappings keep at most 40 evaluations each). issues.json holds the real open issues #25, #40
// to #56, #65 and #67 to #71 of this repository, and one synthetic closed proposal (#9001).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  LABELS, LIMITS, MARKER, buildIndex, checkWhy, coveringIssue, distinctiveTokens, estimate, findGaps, gpuCount, latestRelease, main,
  planFiling, readRecords, scoreGaps, whyFacts, writeWhy, Usage,
} from '../scripts/agents/gaps.mjs';

const FIX = 'tests/agents-gaps-fixtures';
const read = (f) => JSON.parse(readFileSync(join(FIX, f), 'utf8'));
const useCases = read('use-cases.json');
const discrepancies = read('discrepancies.json');
const issues = read('issues.json');
const quiet = () => {};
const loaded = await readRecords(join(FIX, 'records.jsonl'));
const ix = buildIndex(loaded);
const gaps = scoreGaps(ix, findGaps(ix, useCases, discrepancies, { smokeChecks: ['beeline', 'mfass_v2'] }));
const byKey = (k) => gaps.find((g) => g.key === k);

test('finds each kind of tier 1 gap in the fixture release', () => {
  const types = new Set(gaps.map((g) => g.type));
  for (const t of ['no-direct-evidence', 'single-study', 'no-independent', 'discrepancy', 'unverified-recipe']) assert.ok(types.has(t), t);
  // Only tier 1 use cases: cell-type annotation (tier 3) never appears.
  assert.ok(gaps.every((g) => g.useCases.every((u) => u.tier === 1)));
  assert.ok(byKey('use-case:use-case-brca1-brca2-germline-interpretation'));
  assert.equal(byKey('use-case:use-case-cell-type-annotation-transfer'), undefined);
});

test('a discrepancy is proposed only for tier 1 systems, open, with a shared or outlier anchor', () => {
  const d = gaps.filter((g) => g.type === 'discrepancy').map((g) => g.finding.id).sort();
  // GRNBoost2 and PPCOR are tier 3.
  assert.deepEqual(d, ['discrepancy-44e5640b79522964', 'discrepancy-befd4edf81887f7d']);
  const g = byKey('discrepancy:discrepancy-44e5640b79522964');
  assert.equal(g.benchmark, 'gfm-guides-20261014-esm1b-benchmark-brandes2023-clinical-missense');
  assert.match(g.finding.anchorNote, /outlier study's own protocols/);
});

test('a system whose only study is a Rewire run is not flagged as single-study', () => {
  assert.ok(!gaps.some((g) => g.protocols.includes('rewire-protocol-mrnabench-designed-mrl-v1')));
});

test('independent means independent_paper or rewire_run, and studies merge sources by DOI', () => {
  const g = gaps.find((x) => x.systems[0] === 'somatic-oncogenicity-20261009-method-mutpred' && x.benchmark === 'vep-uc-20261014-benchmark-clingen-svi-calibration');
  assert.ok(g.flags.includes('no-independent'));
  assert.ok(g.origins.every((o) => !['independent_paper', 'rewire_run'].includes(o)));
});

test('ranking: value per GPU-hour buckets, then value, and the rule is stated', () => {
  for (let i = 1; i < gaps.length; i++) {
    const a = gaps[i - 1];
    const b = gaps[i];
    assert.ok(a.bucket < b.bucket || (a.bucket === b.bucket && (a.bucket === 1 || a.value >= b.value)), `${a.key} before ${b.key}`);
  }
  for (const g of gaps) assert.equal(g.value, g.clinicalWeight * (1 + g.rankings));
});

test('dedupe: never re-proposes #40 to #56, #65 or #67 to #71, and honours a closed marker', () => {
  const plan = planFiling(ix, gaps, issues);
  assert.ok(plan.file.length > 0 && plan.file.length <= LIMITS.maxIssues);
  for (const g of plan.file) assert.equal(coveringIssue(ix, g, issues), null, g.key);
  const reasons = new Map(plan.skipped.map((s) => [s.gap.key, s.reason]));
  assert.match(reasons.get('verify:somatic-oncogenicity-20261009-method-mutpred:vep-uc-20261014-benchmark-clingen-svi-calibration'), /#9001 \(closed\)/);
  assert.match(reasons.get('verify:somatic-oncogenicity-20261009-method-cadd:vep-uc-20261014-benchmark-clingen-svi-calibration'), /#65/);
  assert.match(reasons.get('recipe:discovery-benchmark-bend'), /#71/);
  assert.match(reasons.get('recipe:discovery-benchmark-proteingym'), /#67/);
  // At most one proposal per benchmark in a run.
  const benches = plan.file.map((g) => g.benchmark).filter(Boolean);
  assert.equal(new Set(benches).size, benches.length);
  // A marker on an open issue also counts.
  const marked = [...issues, { number: 9002, title: 'x', state: 'open', body: `<!-- ${MARKER}: ${plan.file[0].key} -->` }];
  assert.ok(!planFiling(ix, gaps, marked).file.some((g) => g.key === plan.file[0].key));
});

test('distinctive words match a benchmark named differently in a hand-written issue', () => {
  assert.deepEqual(distinctiveTokens('ClinGen SVI calibration of computational predictors for PP3/BP4'), ['ClinGen', 'SVI', 'PP3', 'BP4']);
});

test('estimate: no estimate without a basis, 0 GPU-hours when the recipe says no GPU, GPU-hours from a runtime result', () => {
  const none = estimate(ix, { systems: [], protocols: [], owner: 'discovery-benchmark-genomic-benchmarks' });
  assert.equal(none.kind, 'none');
  assert.equal(none.gpuHours, null);
  assert.match(none.basis, /gives no figure/);

  const recs = new Map(loaded.recs);
  const owner = structuredClone(recs.get('discovery-benchmark-genomic-benchmarks'));
  owner.run_recipes = owner.run_recipes.map((r) => ({ ...r, purpose: 'generate_and_evaluate', requirements: { ...r.requirements, hardware: 'Executed on a laptop CPU. No accelerator required.' } }));
  recs.set(owner.id, owner);
  const cpu = estimate(buildIndex({ recs, runtimes: [] }), { owner: owner.id });
  assert.deepEqual([cpu.kind, cpu.gpuHours, cpu.usd], ['cpu', 0, 0]);
  assert.match(cpu.basis, /No accelerator required/);

  // A runtime result for the same system on the same protocol: 30 minutes on 2 GPUs.
  const ev = [...loaded.recs.values()].find((r) => r.kind === 'evaluation' && r.links.some((l) => l.relation === 'assessment'));
  const sys = buildIndex(loaded).resolveSystem(ev.links.find((l) => l.relation === 'system').target_id);
  const prot = ev.links.find((l) => l.relation === 'assessment').target_id;
  const runtime = { id: 'fixture-runtime', kind: 'result', name: 'r', links: [{ relation: 'evaluation', target_id: ev.id }], metric: 'runtime', unit: 'minute', value: '30', printed: '30', hardware: '2 × NVIDIA A100 80GB' };
  const gpu = estimate(buildIndex({ recs: loaded.recs, runtimes: [runtime] }), { systems: [sys], protocols: [prot] });
  assert.equal(gpu.kind, 'gpu');
  assert.equal(gpu.gpuHours, 1);
  assert.equal(gpu.usd, null); // no sourced price
  assert.match(gpu.basis, /fixture-runtime/);
});

test('gpuCount needs both a GPU and its count', () => {
  assert.equal(gpuCount('2 × Quadro RTX 8000'), 2);
  assert.equal(gpuCount('0.3 d on 8 NVIDIA A100 GPUs'), 8);
  assert.equal(gpuCount('NVIDIA H100 (count not stated)'), null);
  assert.equal(gpuCount('Intel Xeon Gold 6226R, 64 threads'), null);
});

test('the paragraph check drops numbers and links but keeps names with digits', () => {
  const g = byKey('use-case:use-case-brca1-brca2-germline-interpretation');
  const facts = whyFacts(ix, g);
  assert.equal(checkWhy('Reviewers of BRCA1/BRCA2 variants cannot tell which method helps.', facts).ok, true);
  assert.equal(checkWhy('Three studies disagree.', facts).ok, false);
  assert.equal(checkWhy('It scored 0.91 in one study.', facts).ok, false);
  assert.equal(checkWhy('See https://example.org for more.', facts).ok, false);
  assert.equal(checkWhy('Tracked in #60.', facts).ok, false);
});

function fakeClient(text, { stop = 'end_turn' } = {}) {
  const calls = [];
  return {
    calls,
    beta: { messages: { create: async (req) => {
      calls.push(req);
      return { model: req.model, stop_reason: stop, stop_details: stop === 'refusal' ? { category: 'bio' } : null, content: [{ type: 'text', text }], usage: { input_tokens: 1500, output_tokens: 150 } };
    } } },
  };
}

test('writeWhy keeps a clean paragraph, drops a numeric one, and stops at the cost cap', async () => {
  const facts = whyFacts(ix, gaps[0]);
  const ok = await writeWhy(fakeClient('A ranking that rests on one study can reflect that study alone.'), new Usage(), { system: 's', facts });
  assert.ok(ok.text);
  const bad = await writeWhy(fakeClient('It ranks second of twelve.'), new Usage(), { system: 's', facts, log: quiet });
  assert.equal(bad.text, null);
  assert.match(bad.note, /numbers/);
  const refused = await writeWhy(fakeClient('', { stop: 'refusal' }), new Usage(), { system: 's', facts });
  assert.match(refused.note, /declined \(bio\)/);
  const spent = new Usage();
  spent.add('claude-sonnet-5-5', { input_tokens: 600_000, output_tokens: 0 });
  const client = fakeClient('x');
  const capped = await writeWhy(client, spent, { system: 's', facts });
  assert.match(capped.note, /cost cap/);
  assert.equal(client.calls.length, 0);
});

test('dry run renders well-formed proposals from the fixtures', async () => {
  const out = mkdtempSync(join(tmpdir(), 'gaps-'));
  const r = await main(['--dry-run', '--out', out], {}, { log: quiet });
  assert.ok(r.filed.length >= 1 && r.filed.length <= LIMITS.maxIssues);
  for (const f of r.filed) {
    const { title, body, labels } = f.issue;
    assert.match(title, /^Proposed (verification run|benchmark): /);
    assert.deepEqual(labels, Object.keys(LABELS));
    for (const h of ['## Gap', '## Why it matters', '## Model and benchmark', '## Estimate', '## Command']) assert.ok(body.includes(h), h);
    assert.ok(body.includes(`<!-- ${MARKER}: ${f.key} -->`));
    assert.match(body, /GPU-hours: (no estimate, because|estimate )/);
    assert.match(body, /Nothing has been run/);
    assert.ok(!/[\u2013\u2014]/.test(body.replace(/```[\s\S]*?```/g, '')), 'no em or en dashes outside commands');
  }
  assert.ok(existsSync(join(out, 'usage.json')));
  assert.deepEqual(JSON.parse(readFileSync(join(out, 'usage.json'), 'utf8')), { byModel: {}, estimatedUsd: 0 });
});

function fakeGithub() {
  const created = [];
  const labels = [];
  return { created, labels, listIssues: async () => issues, ensureLabels: async (_r, n) => labels.push(...n), createIssue: async (_r, i) => (created.push(i), { html_url: `https://github.com/rewire-bio/x/issues/${created.length}` }) };
}

test('report-only without a key makes no model call and files nothing', async () => {
  const gh = fakeGithub();
  const r = await main(['--report-only', '--release-dir', FIX], { GITHUB_REPOSITORY: 'rewire-bio/rewire-benchmarks' }, { github: gh, log: quiet });
  assert.equal(gh.created.length, 0);
  assert.ok(r.filed.every((f) => /ANTHROPIC_API_KEY is not set/.test(f.whyNote)));
  assert.match(r.summary, /no model call was made/);
});

test('live files at most five proposals as labelled issues, with the paragraph', async () => {
  const gh = fakeGithub();
  const client = fakeClient('A ranking that rests on one study can reflect how that study chose its variants.');
  const r = await main(['--release-dir', FIX], { GITHUB_REPOSITORY: 'rewire-bio/rewire-benchmarks' }, { github: gh, anthropic: client, log: quiet });
  assert.ok(gh.created.length >= 1 && gh.created.length <= LIMITS.maxIssues);
  assert.deepEqual(gh.labels, ['proposal', 'tier-1']);
  assert.equal(client.calls.length, gh.created.length);
  assert.ok(gh.created.every((i) => i.body.includes('Written by claude-sonnet-5-5')));
  assert.ok(r.usage.total() < LIMITS.maxRunUsd);
});

test('refuses a repository outside rewire-bio', async () => {
  await assert.rejects(main(['--report-only', '--release-dir', FIX], { GITHUB_REPOSITORY: 'someone/else' }, { github: fakeGithub(), log: quiet }), /rewire-bio/);
});

test('latestRelease takes the newest serving release that has a kg bundle', () => {
  const list = [
    { tagName: 'serving/2026-11-19-aaaaaaaaaaaa', createdAt: '2026-11-19T10:00:00Z', isDraft: false },
    { tagName: 'kg/2026-11-19-aaaaaaaaaaaa', createdAt: '2026-11-19T10:01:00Z', isDraft: false },
    { tagName: 'serving/2026-11-20-bbbbbbbbbbbb', createdAt: '2026-11-20T10:00:00Z', isDraft: false },
  ];
  assert.equal(latestRelease(list), '2026-11-19-aaaaaaaaaaaa');
  assert.equal(latestRelease([...list, { tagName: 'kg/2026-11-20-bbbbbbbbbbbb', createdAt: '2026-11-20T10:01:00Z', isDraft: false }]), '2026-11-20-bbbbbbbbbbbb');
});

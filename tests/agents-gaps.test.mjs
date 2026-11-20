// Gap scanner (#60). Runs without an API key or network: node --test tests/agents-gaps.test.mjs
//
// The fixtures are real records, use cases, mappings and findings from data release
// 2026-11-20-727601dd065b, trimmed to six use cases, three findings and three recipe owners
// (mappings keep at most 40 evaluations each). From release 2026-11-20-72316db0764b they add the
// Kaschta et al. 2026 manual versus Talos reanalysis (use case, mapping and records) and the
// ProteinGym v1.3 substitutions protocol with its run recipes, and the five proposals of report-only
// run 44739368955 that an independent check found should not be filed (Lancet on the virtual
// tumour, EnFusion on Seraseq, the clinical classifier on CCGA validation, FUSILLI on the Lin et al.
// 2026 cohort; MutPred was already here), with the SEQC2 FD WES indel protocol (open callers on SRA
// data), the DART-Eval benchmark and Rhapsody and Rhapsody-2. For the three recipe proposals of
// report-only run 44743133101 they hold the BEACON and NABench benchmark records, the Genomic
// Benchmarks record as that release has it, and the ProteinGym AMFR protocol (it names the runner
// protocol of the ProteinGym v1.3 recipes). issues.json holds the real issues #40 to #56, #65
// and #67 to #71 (open) and #25 (closed) of this repository, as the REST API returns them, and one
// synthetic closed proposal (#9001).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  LABELS, LIMITS, MARKER, buildIndex, checkWhy, citesId, coveringIssue, dataAccess, distinctiveTokens, estimate, excludeNotRunnable, findGaps, gpuCount,
  latestRelease, main, planFiling, readRecords, recipeCheck, runnability, scoreGaps, successorOf, whyFacts, writeWhy, Usage,
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
  // A closed hand-written issue covers nothing: #25 is closed, so the splicing use case is not skipped for it.
  assert.ok(![...reasons.values()].some((r) => /#25\b/.test(r)));
  // gh issue list spells the state in capitals; it must read the same.
  const upper = issues.map((i) => ({ ...i, state: i.state.toUpperCase() }));
  assert.deepEqual(planFiling(ix, gaps, upper).file.map((g) => g.key), plan.file.map((g) => g.key));
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

  // The same runtime on an evaluation that records only training hardware is not a basis.
  const trained = new Map(loaded.recs);
  trained.set(ev.id, { ...ev, trainingOnly: true });
  assert.equal(estimate(buildIndex({ recs: trained, runtimes: [runtime] }), { systems: [sys], protocols: [prot] }).kind, 'none');
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

test('a discrepancy counts only rankings on the protocols its run would use', () => {
  // PROVEAN and SIFT are ranked together only on the Chen 2020 protocols; the run is on the
  // Brandes 2023 protocols, so it can change no ranking and takes no use case from Chen 2020.
  const g = byKey('discrepancy:discrepancy-44e5640b79522964');
  assert.ok(g.protocols.every((p) => p.startsWith('gfm-guides-20261014-esm1b-protocol-')));
  assert.equal(g.rankings, 0);
  assert.deepEqual(g.useCases, []);
  assert.equal(g.value, 1);
  // The same evaluations mapped onto a run protocol do count.
  const chen = useCases.mappings.find((m) => m.protocol_id === 'somatic-oncogenicity-20261009-protocol-chen2020-oncokb-default' && m.relevance === 'direct');
  const moved = { ...useCases, mappings: [...useCases.mappings, { ...chen, protocol_id: g.protocols[0] }] };
  const g2 = findGaps(ix, moved, discrepancies).find((x) => x.key === g.key);
  assert.equal(g2.rankings, 1);
  assert.deepEqual(g2.useCases.map((u) => u.id), [chen.use_case_id]);
});

test('a discrepancy proposal filed under a previous finding ID is not filed again', () => {
  const old = 'discrepancy-0123456789abcdef';
  const rekeyed = { ...discrepancies, discrepancies: discrepancies.discrepancies.map((f) => (f.id === 'discrepancy-44e5640b79522964' ? { ...f, previous_ids: [old] } : f)) };
  const gs = scoreGaps(ix, findGaps(ix, useCases, rekeyed));
  const key = 'discrepancy:discrepancy-44e5640b79522964';
  // Closed (for example declined) under the old ID: skipped, not refiled.
  const closed = [...issues, { number: 9003, title: 'Proposed verification run: PROVEAN and SIFT', state: 'closed', body: `<!-- ${MARKER}: discrepancy:${old} -->` }];
  const plan = planFiling(ix, gs, closed, 100);
  assert.ok(!plan.file.some((g) => g.key === key));
  assert.match(plan.skipped.find((s) => s.gap.key === key).reason, /#9003 \(closed\)/);
  // An open hand-written issue that cites the old ID covers it too.
  const cited = [...issues, { number: 9004, title: 'x', state: 'open', body: `See ${old}.` }];
  assert.equal(coveringIssue(ix, gs.find((g) => g.key === key), cited)?.number, 9004);
  // Without the alias it would have been filed.
  assert.ok(planFiling(ix, scoreGaps(ix, findGaps(ix, useCases, discrepancies)), closed, 100).file.some((g) => g.key === key));
});

test('common assay words alone do not make an issue cover a gap, and the most specific issue is credited', () => {
  const names = {
    freec: 'Control-FREEC',
    gab: 'NA12878 WGS CNV recall and precision, 1 bp overlap (Gabrielaite et al. Table S2)',
    dragen: 'DRAGEN 4.2 CNV+SV configuration',
    hg002: 'HG002 GIAB v0.6 CNV deletion 1-5kb F-score',
  };
  const fake = { name: (id) => names[id] ?? id };
  const gap = (system, bench) => ({ type: 'single-study', systems: [system], benchmark: bench, protocols: [bench], useCases: [] });
  const byNumber = (n) => issues.filter((i) => i.number === n);
  // #51 is somatic CNV on SEQC2 HCC1395 and names Control-FREEC, WGS and CNV, but not NA12878.
  assert.equal(coveringIssue(fake, gap('freec', 'gab'), byNumber(51)), null);
  assert.equal(coveringIssue(fake, gap('freec', 'gab'), issues), null);
  // DRAGEN CNV on HG002 is #40's scope; #52 (small variants on HG002) must not take the credit.
  assert.equal(coveringIssue(fake, gap('dragen', 'hg002'), issues)?.number, 40);
});

test('estimate: runtimes are summed over systems and protocols, a partial sum is a lower bound', () => {
  const evs = [...loaded.recs.values()].filter((r) => r.kind === 'evaluation' && r.links.some((l) => l.relation === 'assessment'));
  const sysOf = (ev) => ix.resolveSystem(ev.links.find((l) => l.relation === 'system').target_id);
  const protOf = (ev) => ev.links.find((l) => l.relation === 'assessment').target_id;
  const a = evs[0];
  const b = evs.find((e) => protOf(e) === protOf(a) && sysOf(e) && sysOf(e) !== sysOf(a));
  assert.ok(b, 'two systems on one protocol in the fixtures');
  const rt = (id, ev, minutes, extra = {}) => ({ id, kind: 'result', name: id, links: [{ relation: 'evaluation', target_id: ev.id }], metric: 'runtime', unit: 'minute', value: String(minutes), printed: String(minutes), hardware: '2 × NVIDIA A100 80GB', ...extra });
  const runtimes = [rt('rt-a', a, 30), rt('rt-b', b, 60, { approximate: true, printed: 'around 60' })];
  const full = estimate(buildIndex({ recs: loaded.recs, runtimes }), { systems: [sysOf(a), sysOf(b)], protocols: [protOf(a)] });
  assert.equal(full.gpuHours, 3);
  assert.equal(full.lowerBound, false);
  assert.equal(full.approximate, true);
  assert.match(full.basis, /rt-a/);
  assert.match(full.basis, /rt-b/);
  assert.match(full.basis, /approximate/);
  const part = estimate(buildIndex({ recs: loaded.recs, runtimes: [runtimes[0]] }), { systems: [sysOf(a), sysOf(b)], protocols: [protOf(a)] });
  assert.equal(part.gpuHours, 1);
  assert.equal(part.lowerBound, true);
  assert.match(part.basis, /lower bound: runtimes are recorded for 1 of the 2 system and protocol pairs/);
});

const MANUAL = 'reanalysis-kaschta-20261010-method-manual-diagnostic-reanalysis';
const KASCHTA = 'reanalysis-kaschta-20261010-protocol-manual-vs-talos-219-reanalysis';

test('runnability: manual expert reanalysis is not runnable by software; Talos is not shown runnable without a release record', () => {
  const manual = runnability(ix, MANUAL, { protocols: [KASCHTA] });
  assert.equal(manual.runnable, false);
  assert.match(manual.reason, /people perform it/);
  // The fixture keeps no Talos record with code or access, so nothing shows it can be run.
  const talos = runnability(ix, 'uc-clinical-20260930-method-talos', { protocols: [KASCHTA] });
  assert.equal(talos.runnable, false);
  assert.match(talos.reason, /records no released code, weights, container or run recipe/);
  const { kept, excluded } = excludeNotRunnable(ix, gaps);
  assert.match(excluded.find((x) => x.gap.key === `verify:${MANUAL}:${KASCHTA}`).reason, /people perform it/);
  assert.ok(!kept.some((g) => g.systems.includes(MANUAL)));
  // Every kept gap carries the reasons it passed, for the issue to quote.
  for (const g of kept) assert.ok(g.checks && g.checks.data, g.key);
});

test('runnability: decided from release data, with overrides only as a fallback', () => {
  const rec = (id, extra = {}) => ({ id, kind: 'method', name: id, description: '', links: [], access: [], artifacts: [], ...extra });
  const recs = new Map([
    ['lab', rec('lab', { access: ['Clinical laboratory test'] })],
    ['closed', rec('closed', { access: ['Closed system; run by the authors. Not available to reuse.'] })],
    ['tool', rec('tool', { access: ['Tool run by the authors; trained models are not released'] })],
    ['curated', rec('curated', { description: 'Annotates variants with no manual curation.' })],
    ['reviewed', rec('reviewed', { description: 'Variants classified by clinicians.' })],
    // Released code outweighs a description that mentions expert review.
    ['coded', rec('coded', { description: 'Replaces expert review of variants.', access: ['Code at https://github.com/example/coded'] })],
    ['pinned', rec('pinned', { description: 'Manual review aid.', artifacts: ['checkpoint_revision'] })],
    ['blank', rec('blank')],
    ['eval', { id: 'eval', kind: 'evaluation', name: 'e', links: [{ relation: 'system', target_id: 'blank' }, { relation: 'assessment', target_id: 'p' }], adaptation: 'Expert interpretation' }],
  ]);
  const small = buildIndex({ recs });
  const r = (id, opts) => runnability(small, id, opts).runnable;
  assert.equal(r('lab'), false);
  assert.equal(r('closed'), false);
  // "Trained models are not released" says nothing about the tool, but nothing shows it released either.
  assert.match(runnability(small, 'tool').reason, /records no released code/);
  assert.equal(r('tool'), false);
  assert.match(runnability(small, 'curated').reason, /records no released code/); // negated, so not "people perform it"
  assert.equal(r('reviewed'), false);
  assert.equal(r('coded'), true);
  assert.equal(r('pinned'), true);
  // Silence is not runnable; an evaluation's adaptation gives the reason only on the gap's protocols.
  assert.match(runnability(small, 'blank').reason, /records no released code/);
  assert.match(runnability(small, 'blank', { protocols: ['p'] }).reason, /Expert interpretation/);
  // The override list decides when the data cannot.
  const o = runnability(small, 'blank', { overrides: { blank: { runnable: false, reason: 'a person reads the slides' } } });
  assert.equal(o.runnable, false);
  assert.match(o.reason, /RUNNABILITY_OVERRIDES/);
});

test('a dry run leaves out systems that cannot be run and logs each exclusion in the summary', async () => {
  const out = mkdtempSync(join(tmpdir(), 'gaps-'));
  const r = await main(['--dry-run', '--out', out], {}, { log: quiet });
  assert.ok(!r.filed.some((f) => f.key.includes(MANUAL)));
  assert.match(r.summary, /### Blocked: needs a human look\n\nNot filed because/);
  assert.ok(r.summary.includes(`\`verify:${MANUAL}:${KASCHTA}\`): blocked: not runnable by software: the description of`));
  const plan = JSON.parse(readFileSync(join(out, 'plan.json'), 'utf8'));
  assert.equal(plan.excluded.find((x) => x.key === `verify:${MANUAL}:${KASCHTA}`).system, MANUAL);
  // Every filed proposal states why its systems run and where its data is.
  for (const f of r.filed) assert.match(f.issue.body, /\n- Data: /);
});

test('dedupe matches the ProteinGym recipe to #67 by the benchmark ID, not by name', () => {
  const key = 'recipe:rewire-proteingym-v1-3-dms-substitutions';
  const g = byKey(key);
  assert.ok(g);
  // #67 names the benchmark ID and "run guide", but not the protocol's name.
  const i67 = issues.find((i) => i.number === 67);
  assert.ok(!normalisedIncludes(i67, ix.name(g.benchmark)));
  assert.equal(coveringIssue(ix, g, issues)?.number, 67);
  assert.match(planFiling(ix, gaps, issues, 100).skipped.find((s) => s.gap.key === key).reason, /#67/);
  // The protocol ID alone, with "recipe", covers it best; the benchmark ID without recipe words
  // covers it too (same benchmark), but ranks below, so the recipe issue is credited.
  const hand = (body, number = 9100) => [{ number, title: 'x', state: 'open', body }];
  assert.equal(coveringIssue(ix, g, hand('Fix the recipe of `rewire-proteingym-v1-3-dms-substitutions`.'))?.number, 9100);
  assert.equal(coveringIssue(ix, g, hand('See discovery-benchmark-proteingym.'))?.number, 9100);
  assert.equal(coveringIssue(ix, g, [...hand('See discovery-benchmark-proteingym.', 9105), ...hand('Fix the recipe of `rewire-proteingym-v1-3-dms-substitutions`.')])?.number, 9100);
  assert.equal(citesId('a discovery-benchmark-proteingym.', 'discovery-benchmark-proteingym'), true);
  assert.equal(citesId('a discovery-benchmark-proteingym-v2', 'discovery-benchmark-proteingym'), false);
  // A filed recipe proposal for the parent benchmark covers it through its marker.
  assert.equal(coveringIssue(ix, g, hand(`<!-- ${MARKER}: recipe:discovery-benchmark-proteingym -->`))?.number, 9100);
});

test('dedupe by shared benchmark: an open proposal on the benchmark covers other systems on it, a closed one does not', () => {
  const bench = 'vep-uc-20261014-benchmark-clingen-svi-calibration';
  const polyphen = gaps.find((g) => g.type !== 'discrepancy' && g.benchmark === bench && /PolyPhen/.test(ix.name(g.systems[0])));
  assert.ok(polyphen);
  const mutpred = 'somatic-oncogenicity-20261009-method-mutpred';
  const filed = (state) => [{ number: 9101, title: 'Proposed verification run: MutPred', state, body: `Benchmark \`${bench}\`.\n<!-- ${MARKER}: verify:${mutpred}:${bench} -->` }];
  assert.equal(coveringIssue(ix, polyphen, filed('open'))?.number, 9101);
  assert.equal(coveringIssue(ix, polyphen, filed('closed')), null);
  // A filed proposal's text is not searched for IDs: one on another benchmark that lists this one does not cover it.
  const other = [{ number: 9103, title: 'Proposed benchmark: x', state: 'open', body: `Proxy benchmarks: \`${bench}\`.\n<!-- ${MARKER}: use-case:use-case-not-this-one -->` }];
  assert.equal(coveringIssue(ix, polyphen, other), null);
  // A hand-written issue citing the system and benchmark IDs covers it.
  const hand = [{ number: 9102, title: 'x', state: 'open', body: `Rerun ${polyphen.systems[0]} on ${bench}.` }];
  assert.equal(coveringIssue(ix, polyphen, hand)?.number, 9102);
});

function normalisedIncludes(issue, name) {
  const n = (t) => String(t).toLowerCase().replace(/[^a-z0-9]+/g, ' ');
  return n(`${issue.title} ${issue.body}`).includes(n(name).trim());
}

// The five proposals report-only run 44739368955 would have filed, from their real records.
const FLAGGED = {
  mutpred: 'verify:somatic-oncogenicity-20261009-method-mutpred:vep-uc-20261014-benchmark-clingen-svi-calibration',
  lancet: 'verify:amp-oncology-rna-20261007-issue-10-config-lancet:amp-oncology-rna-20261007-protocol-lancet-virtual-tumor',
  enfusion: 'verify:amp-oncology-rna-20261007-issue-11-config-enfusion-3-callers-filter-known-fusion-list:amp-oncology-rna-20261007-protocol-enfusion-seraseq',
  ccga: 'verify:ctdnajam-20261010-method-clinical-data:ctdnajam-20261010-protocol-ccga1-validation-sens-98spec',
  fusilli: 'verify:rna-fusion-20261009-method-fusilli:rna-fusion-20261009-protocol-lin2026-high-depth',
};
const FREEBAYES = 'verify:somatic-20261009-method-freebayes:somatic-20261009-protocol-guille2025-seqc2-fd-wes-indel';

test('none of the five proposals the independent check rejected passes the screen, each with its reason', () => {
  const { kept, excluded } = excludeNotRunnable(ix, gaps);
  const reason = (k) => excluded.find((x) => x.gap.key === k)?.reason;
  for (const k of Object.values(FLAGGED)) {
    assert.ok(byKey(k), k);
    assert.ok(!kept.some((g) => g.key === k), k);
  }
  // CCGA: the dataset record says the sequencing data are not public, and no classifier was released.
  assert.match(reason(FLAGGED.ccga), /`ctdnajam-20261010-data-ccga1-validation` says the data is not public: "Sequencing data are not public\."/);
  assert.match(reason(FLAGGED.ccga), /no released code, weights, container or run recipe for `ctdnajam-20261010-method-clinical-data`/);
  // FUSILLI and EnFusion: no code or access recorded for the authors' tools.
  assert.match(reason(FLAGGED.fusilli), /no released code.*`rna-fusion-20261009-method-fusilli`/);
  assert.match(reason(FLAGGED.fusilli), /no accession, repository or download is recorded for dataset `rna-fusion-20261009-data-lin2026-ball-nanopore`/);
  assert.match(reason(FLAGGED.enfusion), /no released code.*known-fusion-list/);
  assert.match(reason(FLAGGED.lancet), /no released code.*config-lancet/);
  // MutPred: the GitHub link in its configurations' access is the calibration code, not MutPred.
  assert.match(reason(FLAGGED.mutpred), /no released code.*method-mutpred/);
});

test('a system with released code on data with an accession passes, and quotes both', () => {
  const g = byKey(FREEBAYES);
  const run = runnability(ix, g.systems[0], { protocols: g.protocols });
  assert.equal(run.runnable, true);
  assert.match(run.reason, /github\.com\/freebayes/);
  const data = dataAccess(ix, g);
  assert.equal(data.open, true);
  assert.match(data.reason, /SRA run SRR7890879/);
  const { kept } = excludeNotRunnable(ix, gaps);
  assert.ok(kept.some((x) => x.key === FREEBAYES));
});

test('data access: closed statements win, recipe data requirements count, and non-data limitations do not', () => {
  // DART-Eval's recipe says its official inputs need authenticated access.
  const dart = byKey('recipe:discovery-benchmark-dart-eval');
  assert.ok(dart);
  const d = dataAccess(ix, dart);
  assert.equal(d.open, false);
  assert.match(d.reason, /authenticated access is required/);
  // A recipe on the owner is enough when nothing says the data is closed and recipeCheck accepts it.
  assert.equal(dataAccess(ix, byKey('recipe:discovery-benchmark-beacon')).open, true);
  // A recipe recipeCheck rejects says nothing about the data.
  assert.equal(dataAccess(ix, byKey('recipe:discovery-benchmark-genomic-benchmarks')).open, false);
  const rec = (id, kind, extra = {}) => ({ id, kind, name: id, description: '', links: [], access: [], artifacts: [], dataAccessNotes: [], notes: [], recipeDataNotes: [], dataRefs: [], ...extra });
  const small = (p, ds) => buildIndex({ recs: new Map([[p.id, p], ...ds.map((x) => [x.id, x])]) });
  const prot = (notes) => rec('p', 'protocol', { links: [{ relation: 'uses_data', target_id: 'd' }], notes });
  const gap = { systems: [], protocols: ['p'], benchmark: 'p', owner: null };
  const open = rec('d', 'dataset', { dataRefs: ['accession SRP162370'] });
  assert.equal(dataAccess(small(prot([]), [open]), gap).open, true);
  // "Confidence intervals are not available" is about the paper, not the data.
  assert.equal(dataAccess(small(prot(['Confidence intervals are not available.']), [open]), gap).open, true);
  assert.equal(dataAccess(small(prot(['The patient samples are available on request.']), [open]), gap).open, false);
  // A dataset with no source recorded blocks; so does one whose own access says controlled.
  assert.match(dataAccess(small(prot([]), [rec('d', 'dataset')]), gap).reason, /no accession, repository or download/);
  assert.match(dataAccess(small(prot([]), [rec('d', 'dataset', { dataRefs: ['accession phs1'], dataAccessNotes: ['Controlled access through dbGaP'] })]), gap).reason, /not public/);
});

test('dedupe by use case and benchmark words: EnFusion and FUSILLI go to #46, CCGA to #50, MutPred to #65', () => {
  assert.equal(coveringIssue(ix, byKey(FLAGGED.enfusion), issues)?.number, 46);
  assert.equal(coveringIssue(ix, byKey(FLAGGED.fusilli), issues)?.number, 46);
  assert.equal(coveringIssue(ix, byKey(FLAGGED.ccga), issues)?.number, 50);
  assert.equal(coveringIssue(ix, byKey(FLAGGED.mutpred), issues.filter((i) => i.state === 'open'))?.number, 65);
  // SEQC2 callers: #45 shares the use case and the benchmark words, #51 only the words; #45 is credited.
  assert.equal(coveringIssue(ix, byKey(FREEBAYES), issues)?.number, 45);
  // Public resource names and years alone do not make a benchmark match: #65 says ClinVar and
  // gnomAD, which does not make it the Brandes et al. 2023 discrepancy.
  assert.equal(coveringIssue(ix, byKey('discrepancy:discrepancy-44e5640b79522964'), issues), null);
});

test('successors: the release records Rhapsody-2 as the successor to Rhapsody, and none for Lancet', () => {
  const next = successorOf(ix, 'vep-lit-20261014-method-rhapsody');
  assert.equal(next?.id, 'vep-uc-20261014-method-rhapsody-2');
  assert.match(next.reason, /successor to Rhapsody/);
  assert.equal(successorOf(ix, 'vep-uc-20261014-method-rhapsody-2'), null);
  // Lancet2 is not in the release, so Lancet is not flagged as superseded; it is blocked for having no code record.
  assert.equal(successorOf(ix, 'amp-oncology-rna-20261007-issue-10-config-lancet'), null);
  // A superseded system blocks the gap even when it runs and its data is public.
  const g = { ...byKey(FREEBAYES), systems: ['vep-lit-20261014-method-rhapsody'] };
  const { excluded } = excludeNotRunnable(ix, [g]);
  assert.match(excluded[0].reason, /superseded: the release records a successor to Rhapsody, Rhapsody-2/);
});

test('the fixture run files only what passes and never pads to the weekly cap', async () => {
  const r = await main(['--dry-run'], {}, { log: quiet });
  const { kept } = excludeNotRunnable(ix, gaps);
  const passing = planFiling(ix, kept, issues).file.map((g) => g.key);
  assert.deepEqual(r.filed.map((f) => f.key), passing);
  assert.ok(r.filed.length < LIMITS.maxIssues);
  for (const k of Object.values(FLAGGED)) assert.ok(!r.filed.some((f) => f.key === k), k);
});

test('recipe gaps: a placeholder evaluation script or a recipe with no score blocks; BEACON passes with its caveats', () => {
  const { kept, excluded } = excludeNotRunnable(ix, gaps);
  const reason = (k) => excluded.find((x) => x.gap.key === k)?.reason ?? '';
  // NABench: the run documentation and the recipe both say the evaluation script is the user's.
  assert.ok(!kept.some((g) => g.key === 'recipe:discovery-benchmark-nabench'));
  assert.match(reason('recipe:discovery-benchmark-nabench'), /run documentation of `discovery-benchmark-nabench` says ".*SELEX processed data are still being organized.*script to be supplied by the user\."/);
  assert.match(reason('recipe:discovery-benchmark-nabench'), /recipe `nabench-official` says "Example command \(specific script to be provided by you\)"/);
  // Genomic Benchmarks: the official recipe only yields the splits, and the rewirebench scoring
  // recipe is for genomic-benchmarks-v2, which no record in the release defines or uses.
  assert.ok(!kept.some((g) => g.key === 'recipe:discovery-benchmark-genomic-benchmarks'));
  assert.match(reason('recipe:discovery-benchmark-genomic-benchmarks'), /recipe `genomic-benchmarks-official` outputs "Train and test splits as the package ships them\.", not a score/);
  assert.match(reason('recipe:discovery-benchmark-genomic-benchmarks'), /recipe `genomic-benchmarks-rewirebench` is for protocol `genomic-benchmarks-v2`, which no record in the release defines or uses/);
  // ProteinGym v1.3: the AMFR protocol record names its recipes' runner protocol, so they count.
  const pg = recipeCheck(ix, 'rewire-proteingym-v1-3-dms-substitutions');
  assert.equal(pg.ready, true);
  assert.equal(pg.entry.item.id, 'proteingym-v1-3-esm2');
  // "Data and weight paths are placeholders" is a parameter, not missing content.
  const paths = buildIndex({ recs: new Map([['b', { id: 'b', kind: 'benchmark', links: [], runDoc: 'Data and weight paths are placeholders.', run_recipes: [{ id: 'r', protocol_id: 'b', outputs: ['Per-task scores.'], instructions: [] }] }]]) });
  assert.equal(recipeCheck(paths, 'b').ready, true);
  // BEACON: filed, with the release's caveats quoted.
  const beacon = kept.find((g) => g.key === 'recipe:discovery-benchmark-beacon');
  assert.ok(beacon);
  assert.match(beacon.checks.recipe, /recipe `beacon-official` outputs "A fine-tuned model and its score/);
  assert.ok(beacon.caveats.some((c) => /A minimal independent command sequence has not been validated in this pass/.test(c)));
  assert.ok(beacon.caveats.some((c) => /conda environment `beacon` but never activates it/.test(c)));
  assert.ok(beacon.caveats.some((c) => /No tier 1 ranking rests/.test(c)));
});

test('a recipe proposal with no use case says where tier 1 comes from, not that its systems are tier 1', async () => {
  const r = await main(['--dry-run'], {}, { log: quiet });
  const f = r.filed.find((x) => x.key === 'recipe:discovery-benchmark-beacon');
  assert.ok(f);
  assert.match(f.issue.body, /- Tier 1 use cases: none maps a protocol under it; BEACON \(`discovery-benchmark-beacon`\) is depth tier 1 \(its `depth_tier`\)\./);
  for (const x of r.filed) if (x.key.startsWith('recipe:')) assert.ok(!/the systems are tier 1/.test(x.issue.body), x.key);
  assert.match(f.issue.body, /- Caveat: The release's run documentation for `discovery-benchmark-beacon` says: /);
  assert.match(f.issue.body, /Verbatim from the recipe `beacon-official`/);
});

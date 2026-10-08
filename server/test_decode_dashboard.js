/* Pure dashboard contracts. No DOM, server, CUDA, or model process required. */
'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');

const html = fs.readFileSync(path.join(__dirname, 'decode_probe.html'), 'utf8');
const match = html.match(/<script\s+id=["']decode-dashboard-core["'][^>]*>([\s\S]*?)<\/script>/i);
assert(match, 'dashboard must expose its pure helper script');
const context = {module: {exports: {}}, exports: {}, console};
vm.runInNewContext(match[1], context, {filename: 'decode-dashboard-core.js'});
const D = context.DecodeDashboard || context.module.exports.DecodeDashboard || context.module.exports;
for (const name of ['normalizeReport', 'retainReport', 'filterCases', 'qualifiedIsolation',
                    'comparisonKey', 'groupRanks', 'diagnosticGate', 'metricValue', 'operationRole',
                    'sampleRange', 'isolationCaution', 'nestingIndex', 'summaryCase', 'treeGroups']) {
  assert.equal(typeof D[name], 'function', `missing helper ${name}`);
}
const clone = value => JSON.parse(JSON.stringify(value));
let checks = 0;
function test(name, fn) {
  fn(); checks++;
  process.stdout.write(`ok ${checks} - ${name}\n`);
}
function leaf(overrides = {}) {
  return Object.assign({
    name: 'verify/L0/attention_router/attention/dense.fp8/L0.wq_b#1',
    key: [1, 4096, 4], phase: 'verify',
    metadata: {kind: 'dense.fp8', scope: 'verify/L0/attention_router/attention', weight: 'L0.wq_b'},
    inputs: [{shape: [4, 32], stride: [32, 1], dtype: 'torch.bfloat16', device: 'cuda:0'}],
    outputs: [{shape: [4, 16], stride: [16, 1], dtype: 'torch.bfloat16', device: 'cuda:0'}],
    live_ms: .2, replay_count: 5, replay_eligible: true,
    isolation: {exact: true, stale: false, measured_replay_count: 5,
                warm_ms: .1, cold_ms: .12, warm_samples_ms: [.08, .12],
                cold_samples_ms: [.11, .13], calls: 2, repeats: 2, warmup: 1},
  }, overrides);
}
const operation = leaf();
const span = leaf({name: 'verify/L0/attention_router/span#1',
                  metadata: {kind: 'span', scope: 'verify/L0/attention_router'},
                  live_ms: 1.5, replay_eligible: false, isolation: undefined});
const greedy = leaf({phase: 'draft.greedy', name: 'draft.greedy/dense.fp8/L40.wq_b#1',
                    metadata: {kind: 'dense.fp8', scope: 'draft.greedy/attention', weight: 'L40.wq_b'}});
const unreplayed = leaf({phase: 'draft.sampled', name: 'draft.sampled/dense.fp8/L40.wq_b#1',
                        live_ms: null, replay_count: 0});
const otherKey = leaf({key: [0, 4096, 4], live_ms: .3});
const peer = clone(operation);
peer.inputs[0].device = peer.outputs[0].device = 'cuda:1'; peer.live_ms = .25;
const raw = {version: 1, action: 'run', ok: true, ranks: [
  {rank: 0, ok: true, report: {status: 'complete', cases: [operation, span, greedy, unreplayed, otherKey]}},
  {rank: 1, ok: true, report: {status: 'complete', cases: [peer]}},
]};
const report = D.normalizeReport(clone(raw));
const filters = {rank: 'all', phase: 'verify', key: JSON.stringify([1, 4096, 4]),
                 layer: 'all', kind: 'all', mode: 'operations', search: ''};
function captureCase(name, kind, overrides = {}) {
  return leaf({name, metadata: {kind}, isolation: undefined, ...overrides});
}
function captureReport(cases) {
  return D.normalizeReport({ranks: [{rank: 0, report: {status: 'complete', cases}}]});
}
const l23Scope = 'verify/L23/attention_router/attention';
const l23Raw = [
  captureCase('verify/L23/span#1', 'span'),
  captureCase('verify/L23/attention_router/span#1', 'span'),
  captureCase(l23Scope + '/span#1', 'span'),
  captureCase(l23Scope + '/projection#3', 'projection', {skip_reason: 'stateful_timing_only'}),
  captureCase(l23Scope + '/dense.mm/L23.wo_b#1', 'dense.mm', {
    metadata: {kind: 'dense.mm', weight: 'L23.wo_b'}, skip_reason: 'collective_timing_only'}),
  captureCase(l23Scope + '/communication#1', 'communication', {skip_reason: 'collective_timing_only'}),
  captureCase(l23Scope + '/dense.mm/L23.wo_b.local#1', 'dense.mm', {
    metadata: {kind: 'dense.mm', weight: 'L23.wo_b.local'}, skip_reason: 'stateful_timing_only'}),
  captureCase(l23Scope + '/dense.fp8/L23.wo_b.local#1', 'dense.fp8', {
    metadata: {kind: 'dense.fp8', weight: 'L23.wo_b.local'}}),
  captureCase(l23Scope + '/communication#2', 'communication', {skip_reason: 'collective_timing_only'}),
];
const l23 = captureReport(l23Raw);

test('normalization retains both rank reports and annotates case rank', () => {
  assert.equal(report.ranks.length, 2);
  for (const rank of report.ranks) {
    for (const c of rank.report.cases) assert.equal(String(c.rank), String(rank.rank));
  }
  assert(report._dashboard);
});
test('verify and draft phases never mix, and unreplayed sampled rows are excluded', () => {
  assert.equal(D.filterCases(report, filters).length, 2);
  assert.equal(D.filterCases(report, {...filters, phase: 'draft.greedy'}).length, 1);
  assert.equal(D.filterCases(report, {...filters, phase: 'draft.sampled'}).length, 0);
});
test('graph keys and ranks filter independently', () => {
  assert.equal(D.filterCases(report, {...filters, rank: '0'}).length, 1);
  assert.equal(D.filterCases(report, {...filters, key: JSON.stringify([0, 4096, 4])}).length, 1);
  assert.equal(D.filterCases(report, {...filters, key: 'all'}).length, 3);
});
test('operations and nested spans remain separate views rather than additive totals', () => {
  const operations = D.filterCases(report, filters);
  const spans = D.filterCases(report, {...filters, mode: 'spans'});
  assert(operations.every(c => c.metadata.kind !== 'span'));
  assert.equal(spans.length, 1);
  assert.equal(spans[0].metadata.kind, 'span');
});
test('layer, kind, and case-insensitive search filters use operator identity', () => {
  assert.equal(D.filterCases(report, {...filters, layer: 0}).length, 2);
  assert.equal(D.filterCases(report, {...filters, layer: 1}).length, 0);
  assert.equal(D.filterCases(report, {...filters, kind: 'dense.fp8', search: 'WQ_B'}).length, 2);
  assert.equal(D.filterCases(report, {...filters, search: 'not-present'}).length, 0);
});
test('only exact fresh replayed isolation can expose warm/cold latencies', () => {
  assert(D.qualifiedIsolation(operation));
  assert.equal(D.qualifiedIsolation(leaf({isolation: {...operation.isolation, exact: false}})), null);
  assert.equal(D.qualifiedIsolation(leaf({isolation: {...operation.isolation, stale: true}})), null);
  assert.equal(D.qualifiedIsolation(unreplayed), null);
  assert.equal(D.qualifiedIsolation(leaf({isolation: undefined})), null);
});
test('zero is a valid measured latency but null is never converted to zero', () => {
  const zero = leaf({live_ms: 0, isolation: {...operation.isolation, warm_ms: 0, cold_ms: 0}});
  assert(D.qualifiedIsolation(zero));
  assert.equal(D.qualifiedIsolation(leaf({isolation: {...operation.isolation, warm_ms: null}})), null);
});
test('selected chart metric never substitutes live timing for missing isolation', () => {
  assert.equal(D.metricValue(operation, 'live'), .2);
  assert.equal(D.metricValue(operation, 'warm'), .1);
  assert.equal(D.metricValue(operation, 'cold'), .12);
  const stale = leaf({isolation: {...operation.isolation, stale: true}});
  assert.equal(D.metricValue(stale, 'live'), .2);
  assert.equal(D.metricValue(stale, 'warm'), null);
  assert.equal(D.metricValue(stale, 'cold'), null);
  assert.equal(D.metricValue(unreplayed, 'live'), null);
});
test('local dense work, projection wrappers, and TP waits have explicit distinct labels', () => {
  assert.equal(D.operationRole(operation), 'Local dense operation');
  const tp = leaf({skip_reason: 'collective_timing_only', metadata: {kind: 'dense.mm'}});
  assert.equal(D.operationRole(tp), 'TP wrapper · includes communication and peer wait');
  const communication = leaf({skip_reason: 'collective_timing_only', metadata: {kind: 'communication'}});
  assert.equal(D.operationRole(communication), 'Communication · includes peer wait');
  assert.match(D.operationRole(leaf({metadata: {kind: 'projection'}})), /^Projection wrapper/);
  assert.match(D.operationRole(leaf({metadata: {kind: 'projection.grouped'}})), /^Projection wrapper/);
  assert.match(D.operationRole(leaf({metadata: {kind: 'dense.mm'}})), /^Dense wrapper/);
});
test('sample ranges quote measured repeats and old low-repeat screens have a caution', () => {
  assert.deepEqual(clone(D.sampleRange(operation, 'warm')), {min: .08, max: .12, count: 2});
  assert.deepEqual(clone(D.sampleRange(operation, 'cold')), {min: .11, max: .13, count: 2});
  assert.equal(D.sampleRange(operation, 'live'), null);
  assert.equal(D.sampleRange(leaf({isolation: {...operation.isolation, stale: true}}), 'warm'), null);
  assert.equal(D.sampleRange(leaf({isolation: {...operation.isolation, warm_samples_ms: []}}), 'warm'), null);
  assert.equal(D.sampleRange(leaf({isolation: {...operation.isolation, warm_samples_ms: [null, .12]}}), 'warm'), null);
  assert.match(D.isolationCaution({calls: 2, repeats: 2}), /Low-repeat screen.*First-use\/cache effects/);
  assert.equal(D.isolationCaution({calls: 16, repeats: 6}), '');
  assert.equal(D.isolationCaution({}), '');
  assert.deepEqual(clone(report.ranks[0].report.cases[0].isolation.warm_samples_ms), [.08, .12]);
});
test('L23 six nested wrappers reduce to one local leaf and two communication intervals', () => {
  const index = D.nestingIndex(l23);
  const local = l23.ranks[0].report.cases.find(c => c.metadata.kind === 'dense.fp8');
  assert.equal(index.ancestors(local).length, 6);
  assert.deepEqual(clone(index.ancestors(local).map(n => n.c.metadata.kind)),
    ['span', 'span', 'span', 'projection', 'dense.mm', 'dense.mm']);
  const summary = D.filterCases(l23, {...filters, mode: 'summary'});
  assert.deepEqual(clone(summary.map(c => c.metadata.kind)), ['communication', 'dense.fp8', 'communication']);
  assert.equal(D.filterCases(l23, {...filters, mode: 'operations'}).length, 6);
});
test('native BF16 dense.mm remains in summary when no implementation child was captured', () => {
  const native = captureReport([
    captureCase(l23Scope + '/projection#1', 'projection', {skip_reason: 'stateful_timing_only'}),
    captureCase(l23Scope + '/dense.mm/L23.native#1', 'dense.mm', {
      metadata: {kind: 'dense.mm', weight: 'L23.native'}, skip_reason: 'not_selected'}),
  ]);
  const summary = D.filterCases(native, {...filters, mode: 'summary'});
  assert.equal(summary.length, 1);
  assert.equal(summary[0].metadata.weight, 'L23.native');
});
test('repeated scope spans parent their own subsequent calls rather than reusing #1', () => {
  const prefix = 'verify/L23/attention_router/hc';
  const repeated = captureReport([
    captureCase('verify/L23/attention_router/span#1', 'span'),
    captureCase(prefix + '/span#1', 'span'),
    captureCase(prefix + '/lean.hc_mixes#1', 'lean.hc_mixes'),
    captureCase(prefix + '/span#2', 'span'),
    captureCase(prefix + '/lean.hc_mixes#2', 'lean.hc_mixes'),
  ]);
  const rows = repeated.ranks[0].report.cases;
  const index = D.nestingIndex(repeated);
  assert.equal(index.get(rows[2]).parent.c.name, prefix + '/span#1');
  assert.equal(index.get(rows[4]).parent.c.name, prefix + '/span#2');
  assert.equal(index.get(rows[3]).parent.c.name, 'verify/L23/attention_router/span#1');
});
test('recorded parent overrides legacy dispatcher inference, including explicit roots', () => {
  const recorded = captureReport([
    captureCase('verify/L23/attention_router/span#1', 'span'),
    captureCase(l23Scope + '/projection#1', 'projection'),
    captureCase(l23Scope + '/dense.mm/L23.native#1', 'dense.mm', {
      metadata: {kind: 'dense.mm', weight: 'L23.native', parent: 'verify/L23/attention_router/span#1'}}),
    captureCase(l23Scope + '/dense.fp8/L23.native#1', 'dense.fp8', {
      metadata: {kind: 'dense.fp8', weight: 'L23.native', parent: null}}),
  ]);
  const rows = recorded.ranks[0].report.cases;
  const index = D.nestingIndex(recorded);
  assert.equal(index.get(rows[2]).parent.c.name, rows[0].name);
  assert.equal(index.get(rows[2]).basis, 'recorded');
  assert.equal(index.get(rows[3]).parent, null);
});
test('legacy scopes disambiguate a head weight and preserve unknown-weight siblings', () => {
  const head = captureCase('verify/head/head/head#1', 'head', {metadata: {kind: 'head', weight: 'head'}});
  assert.equal(D.scopeOf(head), 'verify/head');
  const unnamed = captureCase('verify/L23/attention_router/attention/projection#3', 'projection');
  assert.equal(D.scopeOf(unnamed), l23Scope);
  const ambiguous = captureReport([
    captureCase(l23Scope + '/dense.mm/Tensor#1', 'dense.mm'),
    captureCase(l23Scope + '/dense.fp8/Unknown#1', 'dense.fp8'),
  ]);
  const rows = ambiguous.ranks[0].report.cases;
  const index = D.nestingIndex(ambiguous);
  assert.equal(index.get(rows[1]).parent, null, 'absent weights do not establish a dispatch chain');
  assert.equal(D.filterCases(ambiguous, {...filters, mode: 'summary'}).length, 2);
});
test('TP head hierarchy retains only its local head implementation and communication in summary', () => {
  const head = captureReport([
    captureCase('verify/head/span#1', 'span'),
    captureCase('verify/head/head/head#1', 'head', {
      metadata: {kind: 'head', weight: 'head'}, skip_reason: 'collective_timing_only'}),
    captureCase('verify/head/head/head.local#1', 'head', {
      metadata: {kind: 'head', weight: 'head.local'}}),
    captureCase('verify/head/dense.mm/head.local#1', 'dense.mm', {
      metadata: {kind: 'dense.mm', weight: 'head.local'}, skip_reason: 'stateful_timing_only'}),
    captureCase('verify/head/dense.fp8/head.local#1', 'dense.fp8', {
      metadata: {kind: 'dense.fp8', weight: 'head.local'}}),
    captureCase('verify/head/communication#1', 'communication', {skip_reason: 'collective_timing_only'}),
  ]);
  const rows = head.ranks[0].report.cases;
  const index = D.nestingIndex(head);
  assert.equal(index.get(rows[2]).parent.c.name, rows[1].name);
  assert.equal(index.get(rows[3]).parent.c.name, rows[2].name);
  assert.equal(index.get(rows[4]).parent.c.name, rows[3].name);
  assert.equal(index.get(rows[5]).parent.c.name, rows[1].name);
  assert.deepEqual(clone(D.filterCases(head, {...filters, mode: 'summary'}).map(c => c.metadata.kind)),
    ['dense.fp8', 'communication']);
});
test('tree retains ancestor totals when kind and search select only a compute child', () => {
  const matches = D.filterCases(l23, {...filters, mode: 'tree', kind: 'dense.fp8', search: 'wo_b.local'});
  assert.equal(matches.length, 1);
  const tree = D.treeGroups(matches, D.nestingIndex(l23));
  assert.equal(tree.roots.length, 1);
  assert.equal(tree.groups.length, 7, 'one leaf plus its six ancestors');
  assert.equal(Object.values(tree.roots[0].cases)[0].name, 'verify/L23/span#1');
  let group = tree.roots[0], depth = 0;
  while (group.children.length) {assert.equal(group.children.length, 1); group = group.children[0]; depth++;}
  assert.equal(depth, 6);
  assert.equal(Object.values(group.cases)[0].metadata.kind, 'dense.fp8');
  // A filtered-out child must not turn an enclosing wrapper into a summary leaf.
  assert.equal(D.filterCases(l23, {...filters, mode: 'summary', kind: 'dense.mm'}).length, 0);
});
test('matched cross-rank operations flag differing recorded parents instead of hiding the disagreement', () => {
  const childName = l23Scope + '/dense.fp8/L23.wq_b#1';
  const parents = ['verify/L23/attention_router/span#1', 'verify/L23/alternative/span#1'];
  const divergent = D.normalizeReport({ranks: [0, 1].map(rank => ({rank, report: {cases: [
    captureCase(parents[rank], 'span', {metadata: {kind: 'span', parent: null}}),
    captureCase(childName, 'dense.fp8', {
      metadata: {kind: 'dense.fp8', weight: 'L23.wq_b', parent: parents[rank]}}),
  ]}}))});
  const index = D.nestingIndex(divergent);
  const selected = D.filterCases(divergent, {...filters, mode: 'tree', kind: 'dense.fp8'});
  const forest = D.treeGroups(selected, index);
  const paired = forest.groups.find(g => g.cases['0'] && g.cases['1']);
  assert(paired, 'matching operation layouts remain comparable across ranks');
  assert.equal(paired.parentMismatch, true);
  for (const rank of ['0', '1']) {
    assert.equal(index.get(paired.cases[rank]).parent.c.name, parents[Number(rank)]);
    assert.equal(index.get(paired.cases[rank]).basis, 'recorded');
  }
  assert.equal(forest.groups.length, 3, 'both recorded ancestors remain available for the warning tooltip');
  assert(forest.roots.every(g => g.parentMismatch === false));
});
test('rank comparison matches complete operator layouts while ignoring device identity', () => {
  assert.equal(D.comparisonKey(operation), D.comparisonKey(peer));
  for (const changed of [leaf({phase: 'draft.greedy'}), otherKey,
      leaf({name: operation.name.replace('#1', '#2')})]) {
    assert.notEqual(D.comparisonKey(operation), D.comparisonKey(changed));
  }
  const stride = clone(peer); stride.inputs[0].stride = [64, 2];
  const dtype = clone(peer); dtype.outputs[0].dtype = 'torch.float32';
  const shape = clone(peer); shape.outputs[0].shape = [4, 17];
  for (const changed of [stride, dtype, shape]) {
    assert.notEqual(D.comparisonKey(operation), D.comparisonKey(changed));
  }
});
test('rank groups pair matching operations and preserve unmatched graph rows', () => {
  const cases = D.filterCases(report, {...filters, key: 'all'});
  const groups = D.groupRanks(cases);
  assert.equal(groups.length, 2);
  const paired = groups.find(g => g.cases['0'] && g.cases['1']);
  assert(paired);
  assert.equal(paired.cases['0'].live_ms, .2);
  assert.equal(paired.cases['1'].live_ms, .25);
});
test('stop and arm status retain a prior measured report, while new cases replace it', () => {
  const stopped = D.normalizeReport({version: 1, action: 'stop', ok: true,
    ranks: [{rank: 0, report: {status: 'stopped', cases: []}}]});
  assert.equal(D.retainReport(report, stopped), report);
  assert.equal(D.retainReport(report, D.normalizeReport({ranks: []})), report);
  const newer = D.normalizeReport({ranks: [{rank: 0, report: {cases: [leaf({live_ms: .4})]}}]});
  assert.deepEqual(clone(D.retainReport(report, newer)), clone(newer));
});
test('newly captured but unreplayed graphs do not erase a measured report', () => {
  const captured = {ranks: [{rank: 0, report: {
    status: 'armed', cases: [unreplayed, leaf({live_ms: null, replay_count: 0})],
  }}]};
  assert.equal(D.retainReport(report, captured), report);
});
test('diagnostic actions require healthy engine and faults require a restart', () => {
  assert.equal(D.diagnosticGate(null).ready, false);
  assert.equal(D.diagnosticGate({status: 'disconnected'}).ready, false);
  assert.equal(D.diagnosticGate({status: 'ok'}).ready, true);
  for (const health of [{status: 'degraded'}, {status: 'ok', ep_fault: 'peer exited'}]) {
    const gate = D.diagnosticGate(health);
    assert.equal(gate.ready, false);
    assert.equal(gate.faulted, true);
    assert.match(gate.message, /Restart required/);
    assert.match(gate.message, /saved report/);
  }
  // Fault state does not remove the measured report or its offline filters.
  assert.equal(D.retainReport(report, {ranks: []}), report);
  assert.equal(D.filterCases(report, filters).length, 2);
});
test('embedded warmed report decompresses and preserves qualified per-rank coverage', () => {
  const embedded = html.match(/<script\s+id=["']decode-dashboard-sample["'][^>]*>([\s\S]*?)<\/script>/i);
  assert(embedded, 'a stopped engine must retain the saved measured sample');
  const envelope = JSON.parse(embedded[1]);
  const saved = envelope.encoding === 'gzip+base64'
    ? JSON.parse(zlib.gunzipSync(Buffer.from(envelope.data, 'base64')).toString('utf8'))
    : envelope;
  const sample = D.normalizeReport(saved);
  assert.equal(sample.ranks.length, 2);
  for (const n of sample.ranks) {
    const cases = n.report.cases;
    const replayed = cases.filter(D.replayed);
    const qualified = cases.filter(D.qualifiedIsolation);
    // These are the recorded warmed greedy sample's measurements, per rank.
    assert.equal(cases.length, 4654);
    assert.equal(replayed.length, 4520);
    assert.equal(cases.length - replayed.length, 134);
    assert.equal(qualified.length, 488);
    assert.equal(qualified.length, n.report.coverage.exact_isolated_cases);
    assert(qualified.every(c => D.sampleRange(c, 'warm').count === 2 && D.sampleRange(c, 'cold').count === 2));
    assert.match(D.isolationCaution(n.report.profile_config), /Low-repeat screen/);
    assert.equal(D.filterCases(sample, {rank: String(n.rank), phase: 'draft.sampled'}).length, 0);
  }
  const paired = D.groupRanks(D.filterCases(sample, {
    phase: 'verify', key: JSON.stringify([1, 4096, 4]), mode: 'operations',
  }));
  assert(paired.length > 0);
  assert(paired.every(g => g.cases['0'] && g.cases['1']), 'compare identical graph/operator layouts across TP ranks');
  assert.equal(D.retainReport(sample, {ranks: []}), sample);
});
test('hot expert-map dashboard contains the exact current standalone HTML', () => {
  const generated = fs.readFileSync(path.join(__dirname, 'expert_map.html'), 'utf8');
  const bootstrap = generated.match(/<script\s+id=["']decode-dashboard-bootstrap["'][^>]*>([\s\S]*?)<\/script>/i);
  assert(bootstrap, 'hot dashboard bootstrap is present');
  const source = bootstrap[1].match(/const source = ("(?:\\.|[^"\\])*");/);
  assert(source);
  assert(JSON.parse(source[1]) === html, 'generated dashboard must be synchronized after standalone HTML edits');
  assert.match(html, /value="summary">Compute &amp; communication<\/option>/);
  assert.match(html, /value="operations">All operations \/ wrappers<\/option>/);
});
async function faultControlsTest() {
  const ui = html.match(/<script\s+id=["']decode-dashboard-ui["'][^>]*>([\s\S]*?)<\/script>/i);
  assert(ui, 'dashboard UI script is present');
  // Expose the control functions without page startup, polling, or real network IO.
  const source = ui[1].replace(/\n start\(\);\n\}\)\(\);/, '\n globalThis.testUI={health,command,applyStatus,renderChart,renderTable,setSort:value=>{sort=value;}};\n})();');
  assert.notEqual(source, ui[1]);
  const elements = new Map();
  const createElement = () => ({value: '', textContent: '', className: '', disabled: false,
    children: [], style: {}, append(...nodes) {this.children.push(...nodes);},
    replaceChildren(...nodes) {this.children = nodes;},
  });
  const getElement = id => {
    if (!elements.has(id)) elements.set(id, createElement());
    return elements.get(id);
  };
  const requests = [];
  let health = {status: 'degraded', ep_fault: 'peer exited'};
  let ok = false;
  const sandbox = {DecodeDashboard: D, console,
    document: {hidden: false, getElementById: getElement, querySelectorAll: () => [],
               createElement, createDocumentFragment: createElement},
    AbortSignal: {timeout: () => undefined},
    fetch: async url => {requests.push(url); return {ok, json: async () => health};},
  };
  vm.runInNewContext(source, sandbox, {filename: 'decode-dashboard-ui.js'});
  sandbox.testUI.applyStatus({ranks: [{rank: 0, report: {status: 'armed'}}]});
  await sandbox.testUI.health();
  assert.match(getElement('notice').textContent, /Restart required/);
  assert.match(getElement('connection-text').textContent, /faulted/);
  for (const id of ['arm', 'run', 'stop']) assert.equal(getElement(id).disabled, true);
  assert.equal(getElement('refresh').disabled, false, 'cached report reads remain available');
  await sandbox.testUI.command('arm');
  assert.deepEqual(requests, ['/health'], 'a faulted engine cannot receive a repeated Arm');
  health = {status: 'ok', busy: false}; ok = true;
  await sandbox.testUI.health();
  for (const id of ['arm', 'run', 'stop']) assert.equal(getElement(id).disabled, false);
  assert.match(getElement('notice').textContent, /Engine available/);
  checks++;
  process.stdout.write(`ok ${checks} - HTTP 503 health fault disables actions without losing restart guidance\n`);
  const node0 = leaf({rank: 0, live_ms: 8,
    isolation: {...operation.isolation, warm_ms: .1, cold_ms: 3, cold_samples_ms: [2.9, 3.1]}});
  const node1 = leaf({rank: 1, live_ms: 2,
    isolation: {...operation.isolation, warm_ms: .3, cold_ms: .15,
                warm_samples_ms: [.28, .32], cold_samples_ms: [.14, .16]}});
  const paired = [{cases: {'0': node0, '1': node1}}];
  const expected = {live: ['8.000 ms', '2.000 ms'], warm: ['0.100 ms', '0.300 ms'],
                    cold: ['3.000 ms', '0.150 ms']};
  const expectedWidths = {live: [100, 25], warm: [100 / 3, 100], cold: [100, 5]};
  for (const metric of ['live', 'warm', 'cold']) {
    sandbox.testUI.setSort(metric);
    sandbox.testUI.renderChart(paired);
    const row = getElement('chart').children[0];
    assert.deepEqual(row.children[2].children.map(n => n.textContent), expected[metric]);
    row.children[1].children.forEach((track, i) => {
      assert.equal(track.children.length, 1);
      assert(Math.abs(parseFloat(track.children[0].style.width) - expectedWidths[metric][i]) < 1e-10);
    });
  }
  node1.isolation.stale = true;
  sandbox.testUI.renderChart(paired);
  const row = getElement('chart').children[0];
  assert.equal(row.children[1].children[1].children.length, 0, 'stale isolation has no metric bar');
  assert.equal(row.children[2].children[1].textContent, '—', 'stale isolation never displays its live value');
  assert.match(row.children[2].children[0].title, /Recorded repeat range: 2\.900 ms – 3\.100 ms \(2 repeats\)/);
  assert.equal(row.children[2].children[1].title, '', 'stale measurements expose no range tooltip');
  sandbox.testUI.setSort('live');
  sandbox.testUI.renderChart(paired);
  assert.match(getElement('chart-caption').textContent, /last instrumented replay · single sample/);
  sandbox.testUI.renderTable(paired);
  assert.match(getElement('timing-settings').textContent, /^Live: last instrumented replay, single sample\. Isolated only: median/);
  checks++;
  process.stdout.write(`ok ${checks} - rendered chart values and bar widths follow live, warm, or cold selection\n`);
}
faultControlsTest().then(() => {
  process.stdout.write(`${checks} dashboard logic checks passed\n`);
}).catch(error => {console.error(error); process.exitCode = 1;});

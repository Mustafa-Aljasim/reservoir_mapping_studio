// Run the actual Cartesian component in a minimal DOM to verify event coordinates.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync('components/control_region_drawer/frontend/index.html', 'utf8');
const events = {}, messages = [], elements = {};
const context = new Proxy({}, {get: (target, key) => target[key] || (() => {}), set: (target, key, value) => (target[key] = value, true)});
for (const id of ['canvas', 'readout', 'status', 'start', 'undo', 'clear', 'finish', 'cancel']) {
  elements[id] = {style: {}, clientWidth: 800, clientHeight: 600,
    getContext: () => context,
    getBoundingClientRect: () => ({left: 0, top: 0, width: 800, height: 600}),
    addEventListener: (name, handler) => events[id + ':' + name] = handler};
}
const scope = {document: {getElementById: id => elements[id]},
  window: {devicePixelRatio: 1, parent: {postMessage: message => messages.push(message)},
    addEventListener: (name, handler) => events['window:' + name] = handler}};
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], scope);
events['window:message']({data: {type: 'streamlit:render', args: {
  mode: 'single_point_pick', bounds: {x_min: 431000, x_max: 433000, y_min: 3363000, y_max: 3365000}}}});
events['canvas:click']({clientX: 400, clientY: 300});
const picked = messages.filter(message => message.type === 'streamlit:setComponentValue');
assert.equal(picked.length, 1);
assert.equal(picked[0].value.x, 432000);
assert.equal(picked[0].value.y, 3364000);
assert.equal(picked[0].value.action, 'pick');
events['canvas:click']({clientX: 450, clientY: 350});
assert.equal(messages.filter(message => message.type === 'streamlit:setComponentValue').length, 1);
events['window:message']({data: {type: 'streamlit:render', args: {mode: 'polygon'}}});
events['start:click']();
for (const point of [[100, 100], [200, 100], [200, 200]]) {
  events['canvas:click']({clientX: point[0], clientY: point[1]});
}
events['finish:click']();
assert.equal(messages.at(-1).value.action, 'finish');
assert.equal(messages.at(-1).value.vertices.length, 3);
console.log('Cartesian point-pick coordinates, single-click behavior, and polygon regression: PASS');

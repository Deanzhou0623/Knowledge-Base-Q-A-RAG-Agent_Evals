import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  createApiClient,
  createController,
  renderSourceList,
} from "../../src/kbqa/ui/app.js";

const FALLBACK = "I cannot confirm from the knowledge base.";

function makeView() {
  const events = [];
  return {
    events,
    showHealth: (value) => events.push(["health", value]),
    setIndexing: (value) => events.push(["indexing", value]),
    showIndexSummary: (value) => events.push(["index", value]),
    setAsking: (value) => events.push(["asking", value]),
    showAnswer: (value) => events.push(["answer", value]),
    showError: (value) => events.push(["error", value]),
    clearError: () => events.push(["clear-error"]),
  };
}

test("healthy and unindexed states mirror the health API", async () => {
  for (const indexLoaded of [true, false]) {
    const view = makeView();
    const health = { status: "ok", backend: "vector", index_loaded: indexLoaded };
    const controller = createController({ health: async () => health }, view);

    assert.deepEqual(await controller.refreshHealth(), health);
    assert.deepEqual(view.events[0], ["health", health]);
  }
});

test("index rebuild exposes progress and refreshes readiness", async () => {
  const view = makeView();
  const summary = { backend: "vector", files_indexed: 3, units_indexed: 9 };
  const api = {
    rebuild: async () => summary,
    health: async () => ({ status: "ok", backend: "vector", index_loaded: true }),
  };

  assert.deepEqual(await createController(api, view).rebuild(), summary);
  assert.deepEqual(view.events[0], ["indexing", true]);
  assert.ok(view.events.some(([name]) => name === "index"));
  assert.deepEqual(view.events.at(-1), ["indexing", false]);
});

test("answerable response is passed through with ranked source metadata", async () => {
  const view = makeView();
  const response = {
    answer: "Refunds take seven days. [refunds.md#timeline]",
    backend: "vector",
    retrieved: [{
      id: "chunk-1",
      heading: "Timeline",
      citation: "refunds.md#timeline",
      text: "Refunds take seven days.",
      rank: 1,
      score: 0.91,
    }],
  };
  const controller = createController({ chat: async () => response }, view);

  assert.equal(await controller.ask(" Refund timing? "), response);
  assert.deepEqual(view.events.find(([name]) => name === "answer"), ["answer", response]);
});

test("unsupported answer remains the exact API fallback", async () => {
  const view = makeView();
  const response = { answer: FALLBACK, backend: "vector", retrieved: [] };

  await createController({ chat: async () => response }, view).ask("Nearby restaurants?");

  assert.equal(view.events.find(([name]) => name === "answer")[1].answer, FALLBACK);
});

test("unindexed and server errors are visible", async () => {
  const unindexedView = makeView();
  const unindexed = createController({
    chat: async () => { throw new ApiError("Index is not available.", 409); },
  }, unindexedView);
  await unindexed.ask("Refund timing?");
  assert.match(unindexedView.events.find(([name]) => name === "error")[1], /Rebuild the index/);

  const serverView = makeView();
  const server = createController({
    health: async () => { throw new ApiError("Service unavailable.", 500); },
  }, serverView);
  assert.equal(await server.refreshHealth(), null);
  assert.deepEqual(serverView.events[0], ["error", "Service unavailable."]);
});

test("API client sends only the shared API contract", async () => {
  const requests = [];
  const fetchMock = async (path, options = {}) => {
    requests.push([path, options]);
    return { ok: true, json: async () => ({}) };
  };
  const client = createApiClient(fetchMock);

  await client.health();
  await client.rebuild();
  await client.chat("Where is my refund?");

  assert.deepEqual(requests.map(([path]) => path), ["/health", "/index", "/chat"]);
  assert.equal(JSON.parse(requests[2][1].body).query, "Where is my refund?");
  assert.deepEqual(Object.keys(JSON.parse(requests[2][1].body)), ["query"]);
});

test("source renderer treats source fields as text, never markup", () => {
  const assignedText = [];
  class FakeElement {
    constructor(tag) {
      this.tag = tag;
      this.children = [];
    }
    set textContent(value) { assignedText.push(value); }
    set innerHTML(_) { throw new Error("innerHTML must not be used"); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
  }
  const documentRef = { createElement: (tag) => new FakeElement(tag) };
  const container = new FakeElement("div");
  const attack = "<img src=x onerror=alert(1)>";

  renderSourceList(container, [{
    rank: 1,
    score: 1,
    heading: attack,
    citation: attack,
    text: `<script>${attack}</script>`,
  }], documentRef);

  assert.equal(container.children.length, 1);
  assert.ok(assignedText.includes(attack));
  assert.ok(assignedText.includes(`<script>${attack}</script>`));
});

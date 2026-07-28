const FALLBACK_ANSWER = "I cannot confirm from the knowledge base.";

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const detail = typeof payload?.detail === "string"
      ? payload.detail
      : `Request failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }
  return payload;
}

export function createApiClient(fetchImpl = globalThis.fetch) {
  const request = async (path, options = {}) => {
    try {
      return await parseResponse(await fetchImpl(path, options));
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError("Cannot reach the Q&A service. Check that the server is running.");
    }
  };

  return {
    health: () => request("/health"),
    rebuild: () => request("/index", { method: "POST" }),
    chat: (query) => request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    }),
  };
}

export function createController(api, view) {
  const refreshHealth = async () => {
    try {
      const health = await api.health();
      view.showHealth(health);
      view.clearError();
      return health;
    } catch (error) {
      view.showError(error.message);
      return null;
    }
  };

  const rebuild = async () => {
    view.setIndexing(true);
    view.clearError();
    try {
      const summary = await api.rebuild();
      view.showIndexSummary(summary);
      await refreshHealth();
      return summary;
    } catch (error) {
      // Drop "Building the document index…" and any earlier success line so
      // the UI never claims a completed index the failed run did not produce.
      view.clearIndexSummary();
      view.showError(error.message);
      return null;
    } finally {
      view.setIndexing(false);
    }
  };

  const ask = async (query) => {
    const normalized = query.trim();
    if (!normalized) {
      view.showError("Enter a question before asking.");
      return null;
    }
    view.setAsking(true);
    view.clearError();
    // A stale answer under a new question reads as the response to that
    // question, so clear it before the request rather than only on success.
    view.clearAnswer();
    try {
      const response = await api.chat(normalized);
      view.showAnswer(response);
      return response;
    } catch (error) {
      const message = error.status === 409
        ? `${error.message} Rebuild the index, then try again.`
        : error.message;
      view.showError(message);
      return null;
    } finally {
      view.setAsking(false);
    }
  };

  return { refreshHealth, rebuild, ask };
}

export function renderSourceList(container, sources, documentRef = document) {
  container.replaceChildren();
  if (!sources.length) {
    const empty = documentRef.createElement("p");
    empty.className = "empty-sources";
    empty.textContent = "No document sources were retrieved for this answer.";
    container.append(empty);
    return;
  }

  for (const source of sources) {
    const article = documentRef.createElement("article");
    article.className = "source-card";

    const top = documentRef.createElement("div");
    top.className = "source-topline";
    const rank = documentRef.createElement("span");
    rank.className = "rank";
    rank.textContent = `Rank ${source.rank}`;
    const score = documentRef.createElement("span");
    score.className = "score";
    score.textContent = `Score ${Number(source.score).toPrecision(5)}`;
    top.append(rank, score);

    const heading = documentRef.createElement("h3");
    heading.textContent = source.heading;
    const citation = documentRef.createElement("code");
    citation.textContent = source.citation;
    const unitId = documentRef.createElement("code");
    unitId.className = "unit-id";
    unitId.textContent = `id ${source.id}`;
    const text = documentRef.createElement("p");
    text.className = "source-text";
    text.textContent = source.text;

    article.append(top, heading, citation, unitId, text);
    container.append(article);
  }
}

export function createDomView(documentRef = document) {
  const element = (id) => documentRef.getElementById(id);
  const rebuildButton = element("rebuild-index");
  const askButton = element("ask-button");
  const error = element("error-message");

  return {
    showHealth(health) {
      element("backend-badge").textContent = health.backend;
      element("health-summary").textContent = health.index_loaded
        ? "Ready to answer"
        : "Index required";
      element("health-summary").dataset.ready = String(health.index_loaded);
    },
    setIndexing(active) {
      rebuildButton.disabled = active;
      rebuildButton.textContent = active ? "Rebuilding…" : "Rebuild index";
      if (active) {
        element("operation-message").textContent = "Building the document index…";
      }
    },
    showIndexSummary(summary) {
      element("operation-message").textContent =
        `Indexed ${summary.units_indexed} units from ${summary.files_indexed} files.`;
    },
    setAsking(active) {
      askButton.disabled = active;
      askButton.textContent = active ? "Asking…" : "Ask question";
    },
    showAnswer(response) {
      const answer = element("answer");
      answer.textContent = response.answer;
      answer.classList.remove("empty");
      answer.dataset.fallback = String(response.answer === FALLBACK_ANSWER);
      element("backend-badge").textContent = response.backend;
      element("timing").textContent =
        `Retrieval ${response.retrieval_ms.toFixed(1)} ms · Generation ${response.generation_ms.toFixed(1)} ms`;
      element("model").textContent = `Model ${response.model}`;
      element("source-count").textContent =
        `${response.retrieved.length} source${response.retrieved.length === 1 ? "" : "s"}`;
      renderSourceList(element("sources"), response.retrieved, documentRef);
    },
    clearAnswer() {
      const answer = element("answer");
      answer.textContent = "";
      answer.classList.add("empty");
      delete answer.dataset.fallback;
      element("timing").textContent = "";
      element("model").textContent = "";
      element("source-count").textContent = "";
      renderSourceList(element("sources"), [], documentRef);
    },
    clearIndexSummary() {
      element("operation-message").textContent = "";
    },
    showError(message) {
      error.textContent = message;
      error.hidden = false;
    },
    clearError() {
      error.textContent = "";
      error.hidden = true;
    },
  };
}

if (typeof document !== "undefined") {
  const view = createDomView(document);
  const controller = createController(createApiClient(), view);
  document.getElementById("refresh-health").addEventListener("click", controller.refreshHealth);
  document.getElementById("rebuild-index").addEventListener("click", controller.rebuild);
  document.getElementById("question-form").addEventListener("submit", (event) => {
    event.preventDefault();
    controller.ask(document.getElementById("query").value);
  });
  controller.refreshHealth();
}

const state = {
  session: null,
  data: null,
};

const els = {
  status: document.querySelector("#status"),
  mode: document.querySelector("#mode"),
  seat: document.querySelector("#seat"),
  bot: document.querySelector("#bot"),
  decisionTimeout: document.querySelector("#decisionTimeout"),
  newGame: document.querySelector("#newGame"),
  p0Count: document.querySelector("#p0Count"),
  p1Count: document.querySelector("#p1Count"),
  drawCount: document.querySelector("#drawCount"),
  discard: document.querySelector("#discard"),
  currentColor: document.querySelector("#currentColor"),
  drawButton: document.querySelector("#drawButton"),
  passButton: document.querySelector("#passButton"),
  hand: document.querySelector("#hand"),
  log: document.querySelector("#log"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

async function loadBots() {
  const payload = await api("/api/bots");
  els.bot.innerHTML = "";
  for (const bot of payload.bots) {
    const option = document.createElement("option");
    option.value = bot;
    option.textContent = bot;
    if (bot === payload.default) option.selected = true;
    els.bot.append(option);
  }
}

async function newGame() {
  const payload = await api("/api/new", {
    method: "POST",
    body: JSON.stringify({
      mode: els.mode.value,
      human_seat: Number(els.seat.value),
      bot: els.bot.value,
      decision_timeout: Number(els.decisionTimeout.value),
    }),
  });
  state.session = payload.session;
  render(payload);
  await maybeAdvance();
}

async function submitAction(action) {
  if (!state.session) return;
  const payload = await api("/api/action", {
    method: "POST",
    body: JSON.stringify({session: state.session, action}),
  });
  render(payload);
  await maybeAdvance();
}

async function maybeAdvance() {
  if (!state.session || !state.data || !state.data.bot_turn) return;
  window.setTimeout(async () => {
    const payload = await api("/api/advance", {
      method: "POST",
      body: JSON.stringify({session: state.session, decision_timeout: Number(els.decisionTimeout.value)}),
    });
    render(payload);
    await maybeAdvance();
  }, 260);
}

function render(payload) {
  state.data = payload;
  els.status.textContent = payload.status_text;
  els.p0Count.textContent = String(payload.hand_counts[0]);
  els.p1Count.textContent = String(payload.hand_counts[1]);
  els.drawCount.textContent = String(payload.draw_pile_count);
  els.currentColor.textContent = payload.current_color;

  els.discard.textContent = payload.top_card.label;
  els.discard.className = `uno-card large ${payload.current_color}`;

  els.drawButton.disabled = !payload.legal_actions.includes("draw");
  els.passButton.disabled = !payload.legal_actions.includes("pass");

  renderHand(payload);
  renderLog(payload.log);
}

function renderHand(payload) {
  els.hand.innerHTML = "";
  for (const card of payload.hand) {
    const wrapper = document.createElement("div");
    wrapper.className = "card-wrap";

    const button = document.createElement("button");
    button.type = "button";
    button.className = `uno-card ${card.color || "wild"}`;
    button.textContent = card.label;
    button.disabled = card.legal_actions.length !== 1;
    if (card.legal_actions.length === 1) {
      button.addEventListener("click", () => submitAction(card.legal_actions[0]));
    }
    wrapper.append(button);

    if (card.legal_actions.length > 1) {
      const choices = document.createElement("div");
      choices.className = "wild-choices";
      for (const action of card.legal_actions) {
        const color = action.split(":")[2];
        const colorButton = document.createElement("button");
        colorButton.type = "button";
        colorButton.className = `swatch ${color}`;
        colorButton.title = color;
        colorButton.addEventListener("click", () => submitAction(action));
        choices.append(colorButton);
      }
      wrapper.append(choices);
    }
    els.hand.append(wrapper);
  }
}

function renderLog(items) {
  els.log.innerHTML = "";
  for (const item of [...items].reverse()) {
    const row = document.createElement("div");
    row.className = "log-row";
    row.textContent = item.text;
    els.log.append(row);
  }
}

els.newGame.addEventListener("click", () => newGame().catch(showError));
els.drawButton.addEventListener("click", () => submitAction("draw").catch(showError));
els.passButton.addEventListener("click", () => submitAction("pass").catch(showError));

function showError(error) {
  els.status.textContent = error.message;
}

loadBots().then(newGame).catch(showError);

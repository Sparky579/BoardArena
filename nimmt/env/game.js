const statusEl = document.getElementById("status");
const rowsEl = document.getElementById("rows");
const handEl = document.getElementById("hand");
const scoresEl = document.getElementById("scores");
const logEl = document.getElementById("log");
const modeEl = document.getElementById("mode");
const playersEl = document.getElementById("players");
const seatEl = document.getElementById("seat");
const botEl = document.getElementById("bot");
const decisionTimeoutEl = document.getElementById("decisionTimeout");
const newGameEl = document.getElementById("newGame");

let state = null;

async function api(path, payload) {
  const options = payload
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "request failed");
  return data;
}

async function loadBots() {
  const data = await api("/api/bots");
  botEl.innerHTML = "";
  data.bots.forEach((id) => {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = id;
    option.selected = id === data.default;
    botEl.appendChild(option);
  });
}

function syncSeatOptions() {
  const players = Number(playersEl.value);
  const current = Number(seatEl.value || 0);
  seatEl.innerHTML = "";
  for (let index = 0; index < players; index += 1) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Player ${index + 1}`;
    option.selected = index === Math.min(current, players - 1);
    seatEl.appendChild(option);
  }
  seatEl.disabled = modeEl.value === "human-human";
}

async function newGame() {
  state = await api("/api/new", {
    mode: modeEl.value,
    players: Number(playersEl.value),
    human_seat: Number(seatEl.value),
    bot: botEl.value,
    decision_timeout: Number(decisionTimeoutEl.value),
  });
  render();
}

async function choose(action) {
  if (state.next_human === null || state.phase === "game_over") return;
  state = await api("/api/action", {
    session: state.session,
    player: state.next_human,
    action,
    decision_timeout: Number(decisionTimeoutEl.value),
  });
  render();
}

function render() {
  statusEl.textContent = state.status_text;
  renderRows();
  renderHand();
  renderScores();
  renderLog();
}

function renderRows() {
  rowsEl.innerHTML = "";
  for (let index = 0; index < 4; index += 1) {
    const row = document.createElement("div");
    row.className = "row";
    const cards = state.rows[index] || [];
    const label = document.createElement("div");
    label.className = "row-label";
    label.textContent = `Row ${index} (${state.row_bulls[index] || 0})`;
    row.appendChild(label);
    cards.forEach((card) => row.appendChild(cardNode(card)));
    rowsEl.appendChild(row);
  }
}

function renderHand() {
  handEl.innerHTML = "";
  if (state.next_human === null) return;
  const hand = state.hands[state.next_human];
  const legal = new Set(state.legal_actions);
  hand.forEach((card) => {
    const actions = [...legal].filter((action) => action === `PLAY_${card}` || action.startsWith(`PLAY_${card}_TAKE_`));
    const button = cardNode(card, "button");
    button.disabled = actions.length === 0;
    button.addEventListener("click", () => chooseActionForCard(actions));
    handEl.appendChild(button);
  });
}

function chooseActionForCard(actions) {
  if (actions.length === 1) {
    choose(actions[0]);
    return;
  }
  const take = window.prompt(`Take row 0-${actions.length - 1}`);
  const action = actions.find((item) => item.endsWith(`_${take}`));
  if (action) choose(action);
}

function renderScores() {
  scoresEl.innerHTML = "";
  state.scores.forEach((score, player) => {
    const item = document.createElement("div");
    item.className = "score";
    item.innerHTML = `<strong>P${player + 1}</strong><span>${score} bulls | ${state.hand_sizes[player]} cards</span>`;
    scoresEl.appendChild(item);
  });
}

function renderLog() {
  logEl.innerHTML = "";
  state.log.slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    logEl.appendChild(li);
  });
}

function cardNode(card, tag = "div") {
  const node = document.createElement(tag);
  node.className = "card";
  node.innerHTML = `<span>${card}<small>${bulls(card)} bull</small></span>`;
  return node;
}

function bulls(card) {
  if (card === 55) return 7;
  if (card % 11 === 0) return 5;
  if (card % 10 === 0) return 3;
  if (card % 5 === 0) return 2;
  return 1;
}

newGameEl.addEventListener("click", newGame);
playersEl.addEventListener("change", syncSeatOptions);
modeEl.addEventListener("change", syncSeatOptions);

syncSeatOptions();
loadBots().then(newGame).catch((error) => {
  statusEl.textContent = error.message;
});

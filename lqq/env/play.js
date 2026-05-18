const SIZE = 9;
const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const modeEl = document.getElementById("mode");
const seatEl = document.getElementById("seat");
const botEl = document.getElementById("bot");
const wallActionEl = document.getElementById("wallAction");
const placeWallEl = document.getElementById("placeWall");
const newGameEl = document.getElementById("newGame");
const logEl = document.getElementById("log");
const p0El = document.getElementById("p0");
const p1El = document.getElementById("p1");

let state = null;
const moveActions = {
  MOVE_UP: [-1, 0],
  MOVE_DOWN: [1, 0],
  MOVE_LEFT: [0, -1],
  MOVE_RIGHT: [0, 1],
};

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

async function newGame() {
  state = await api("/api/new", {
    mode: modeEl.value,
    human_seat: Number(seatEl.value),
    bot: botEl.value,
  });
  render();
  await advanceIfNeeded();
}

async function submitAction(action) {
  if (!state || !state.human_turn) return;
  state = await api("/api/action", { session: state.session, action });
  render();
  await advanceIfNeeded();
}

async function advanceIfNeeded() {
  while (state && state.bot_turn) {
    state = await api("/api/advance", { session: state.session });
    render();
  }
}

function render() {
  statusEl.textContent = state.status_text;
  p0El.textContent = `${state.positions[0][0]},${state.positions[0][1]} | walls ${state.walls_remaining[0]}`;
  p1El.textContent = `${state.positions[1][0]},${state.positions[1][1]} | walls ${state.walls_remaining[1]}`;
  renderBoard();
  renderWalls();
  renderLog();
}

function renderBoard() {
  boardEl.innerHTML = "";
  const legalMoves = new Map();
  state.legal_actions.forEach((action) => {
    if (!moveActions[action]) return;
    const [dr, dc] = moveActions[action];
    const [row, col] = state.positions[state.actor];
    legalMoves.set(`${row + dr},${col + dc}`, action);
  });

  for (let row = 0; row < SIZE; row += 1) {
    for (let col = 0; col < SIZE; col += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      if (row === 0) cell.classList.add("goal0");
      if (row === SIZE - 1) cell.classList.add("goal1");

      const player = state.positions.findIndex(([r, c]) => r === row && c === col);
      if (player >= 0) {
        const pawn = document.createElement("span");
        pawn.className = `pawn p${player}`;
        cell.appendChild(pawn);
      }

      const move = legalMoves.get(`${row},${col}`);
      if (move) {
        cell.classList.add("legal");
        cell.addEventListener("click", () => submitAction(move));
      }
      boardEl.appendChild(cell);
    }
  }
}

function renderWalls() {
  const wallActions = state.legal_actions.filter((action) => action.startsWith("WALL_"));
  wallActionEl.innerHTML = "";
  wallActions.forEach((action) => {
    const option = document.createElement("option");
    option.value = action;
    option.textContent = action;
    wallActionEl.appendChild(option);
  });
  placeWallEl.disabled = wallActions.length === 0;
}

function renderLog() {
  logEl.innerHTML = "";
  state.log.slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    logEl.appendChild(li);
  });
}

placeWallEl.addEventListener("click", () => {
  if (wallActionEl.value) submitAction(wallActionEl.value);
});
newGameEl.addEventListener("click", newGame);
modeEl.addEventListener("change", () => {
  seatEl.disabled = modeEl.value === "human-human";
});

loadBots().then(newGame).catch((error) => {
  statusEl.textContent = error.message;
});

const SIZE = 9;
const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const modeEl = document.getElementById("mode");
const seatEl = document.getElementById("seat");
const botEl = document.getElementById("bot");
const decisionTimeoutEl = document.getElementById("decisionTimeout");
const wallHEl = document.getElementById("wallH");
const wallVEl = document.getElementById("wallV");
const newGameEl = document.getElementById("newGame");
const logEl = document.getElementById("log");
const p0El = document.getElementById("p0");
const p1El = document.getElementById("p1");

const DEFAULT_REQUEST_TIMEOUT_MS = 10000;

let state = null;
let busy = false;
let selectedWallDirection = "H";
const moveActions = {
  MOVE_UP: [-1, 0],
  MOVE_DOWN: [1, 0],
  MOVE_LEFT: [0, -1],
  MOVE_RIGHT: [0, 1],
  MOVE_UP_LEFT: [-1, -1],
  MOVE_UP_RIGHT: [-1, 1],
  MOVE_DOWN_LEFT: [1, -1],
  MOVE_DOWN_RIGHT: [1, 1],
};

async function api(path, payload) {
  const controller = new AbortController();
  const timeoutMs = requestTimeoutMs(payload);
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const options = payload
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal: controller.signal }
    : { signal: controller.signal };
  try {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "request failed");
    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(`request timed out after ${(timeoutMs / 1000).toFixed(1)}s`);
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

function requestTimeoutMs(payload) {
  const botSeconds = payload && Number(payload.decision_timeout);
  if (Number.isFinite(botSeconds) && botSeconds > 0) {
    return Math.max(DEFAULT_REQUEST_TIMEOUT_MS, (botSeconds + 5) * 1000);
  }
  return DEFAULT_REQUEST_TIMEOUT_MS;
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
  await runAction(async () => {
    state = await api("/api/new", {
      mode: modeEl.value,
      human_seat: Number(seatEl.value),
      bot: botEl.value,
      decision_timeout: Number(decisionTimeoutEl.value),
    });
    render();
    await advanceIfNeeded();
  });
}

async function changeBot() {
  if (modeEl.value === "human-human") {
    renderControls();
    return;
  }
  if (!state) {
    await newGame();
    return;
  }
  await runAction(async () => {
    renderBusy("Switching bot...");
    state = await api("/api/bot", {
      session: state.session,
      bot: botEl.value,
    });
    render();
    await advanceIfNeeded();
  });
}

async function submitAction(action) {
  if (!state || !state.human_turn) return;
  await runAction(async () => {
    state = await api("/api/action", { session: state.session, action });
    render();
    await advanceIfNeeded();
  });
}

async function advanceIfNeeded() {
  while (state && state.bot_turn) {
    renderBusy("Bot is thinking...");
    state = await api("/api/advance", { session: state.session, decision_timeout: Number(decisionTimeoutEl.value) });
    render();
  }
}

async function runAction(callback) {
  if (busy) return;
  busy = true;
  renderControls();
  try {
    await callback();
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    busy = false;
    renderControls();
  }
}

function render() {
  statusEl.textContent = state.status_text;
  p0El.textContent = `${state.positions[0][0]},${state.positions[0][1]} | walls ${state.walls_remaining[0]}`;
  p1El.textContent = `${state.positions[1][0]},${state.positions[1][1]} | walls ${state.walls_remaining[1]}`;
  renderBoard();
  renderControls();
  renderLog();
}

function renderBusy(message) {
  statusEl.textContent = message;
  renderControls();
}

function renderBoard() {
  boardEl.innerHTML = "";
  const placedEdges = buildPlacedEdges();
  const legalMoves = new Map();
  state.legal_actions.forEach((action) => {
    if (!moveActions[action]) return;
    const target = moveTarget(action, placedEdges);
    if (target) legalMoves.set(`${target.row},${target.col}`, action);
  });
  const legalWalls = new Set(state.legal_actions.filter((action) => action.startsWith("WALL_")));
  const placedOrigins = new Set(state.walls.map((wall) => wallAction(wall.dir, wall.row, wall.col)));

  for (let gridRow = 0; gridRow < SIZE * 2 - 1; gridRow += 1) {
    for (let gridCol = 0; gridCol < SIZE * 2 - 1; gridCol += 1) {
      if (gridRow % 2 === 0 && gridCol % 2 === 0) {
        boardEl.appendChild(createCell(gridRow / 2, gridCol / 2, legalMoves));
      } else if (gridRow % 2 === 1 && gridCol % 2 === 0) {
        boardEl.appendChild(createWallSlot("H", (gridRow - 1) / 2, gridCol / 2, legalWalls, placedEdges));
      } else if (gridRow % 2 === 0 && gridCol % 2 === 1) {
        boardEl.appendChild(createWallSlot("V", gridRow / 2, (gridCol - 1) / 2, legalWalls, placedEdges));
      } else {
        boardEl.appendChild(createJunction((gridRow - 1) / 2, (gridCol - 1) / 2, legalWalls, placedOrigins));
      }
    }
  }
}

function createCell(row, col, legalMoves) {
  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = "cell";
  cell.ariaLabel = `cell ${col + 1},${row + 1}`;
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
  return cell;
}

function createWallSlot(dir, row, col, legalWalls, placedEdges) {
  const slot = document.createElement("button");
  slot.type = "button";
  slot.className = `wall-slot ${dir === "H" ? "horizontal" : "vertical"}`;
  slot.dataset.dir = dir;
  slot.dataset.row = String(row);
  slot.dataset.col = String(col);
  slot.ariaLabel = `${dir === "H" ? "horizontal" : "vertical"} wall slot ${row + 1},${col + 1}`;

  if (placedEdges.has(edgeKey(dir, row, col))) {
    slot.classList.add("placed");
  }

  const origin = normalizeOrigin(dir, row, col);
  const action = wallAction(dir, origin.row, origin.col);
  if (legalWalls.has(action)) {
    slot.classList.add("placeable");
    slot.addEventListener("mouseenter", () => previewWall(dir, origin.row, origin.col, true));
    slot.addEventListener("mouseleave", () => previewWall(dir, origin.row, origin.col, false));
    slot.addEventListener("click", () => submitAction(action));
  }
  return slot;
}

function createJunction(row, col, legalWalls, placedOrigins) {
  const junction = document.createElement("button");
  junction.type = "button";
  junction.className = "junction";
  junction.dataset.row = String(row);
  junction.dataset.col = String(col);
  junction.ariaLabel = `wall junction ${row + 1},${col + 1}`;

  const horizontal = wallAction("H", row, col);
  const vertical = wallAction("V", row, col);
  if (placedOrigins.has(horizontal) || placedOrigins.has(vertical)) {
    junction.classList.add("placed");
  }

  const action = wallAction(selectedWallDirection, row, col);
  if (legalWalls.has(action)) {
    junction.classList.add("placeable");
    junction.addEventListener("mouseenter", () => previewWall(selectedWallDirection, row, col, true));
    junction.addEventListener("mouseleave", () => previewWall(selectedWallDirection, row, col, false));
    junction.addEventListener("click", () => submitAction(action));
  }
  return junction;
}

function buildPlacedEdges() {
  const edges = new Set();
  state.walls.forEach((wall) => {
    wallEdges(wall.dir, wall.row, wall.col).forEach((edge) => edges.add(edge));
  });
  return edges;
}

function moveTarget(action, placedEdges) {
  const delta = moveActions[action];
  if (!delta) return null;
  const [dr, dc] = delta;
  const actor = state.actor;
  const opponent = 1 - actor;
  const [row, col] = state.positions[actor];
  const [oppRow, oppCol] = state.positions[opponent];

  if (dr !== 0 && dc !== 0) {
    return sideJumpTarget(row, col, oppRow, oppCol, dr, dc, placedEdges);
  }

  const nextRow = row + dr;
  const nextCol = col + dc;

  if (!inBounds(nextRow, nextCol)) return null;
  if (hasWallBetween(row, col, nextRow, nextCol, placedEdges)) return null;
  if (nextRow !== oppRow || nextCol !== oppCol) {
    return { row: nextRow, col: nextCol };
  }

  const jumpRow = oppRow + dr;
  const jumpCol = oppCol + dc;
  if (!inBounds(jumpRow, jumpCol)) return null;
  if (hasWallBetween(oppRow, oppCol, jumpRow, jumpCol, placedEdges)) return null;
  return { row: jumpRow, col: jumpCol };
}

function sideJumpTarget(row, col, oppRow, oppCol, dr, dc, placedEdges) {
  if (Math.abs(row - oppRow) + Math.abs(col - oppCol) !== 1) return null;

  const towardRow = oppRow - row;
  const towardCol = oppCol - col;
  if (dr !== towardRow && dc !== towardCol) return null;
  if (hasWallBetween(row, col, oppRow, oppCol, placedEdges)) return null;

  const targetRow = row + dr;
  const targetCol = col + dc;
  if (!inBounds(targetRow, targetCol)) return null;
  if (hasWallBetween(oppRow, oppCol, targetRow, targetCol, placedEdges)) return null;
  return { row: targetRow, col: targetCol };
}

function inBounds(row, col) {
  return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}

function hasWallBetween(fromRow, fromCol, toRow, toCol, placedEdges) {
  if (fromRow === toRow) {
    return placedEdges.has(edgeKey("V", fromRow, Math.min(fromCol, toCol)));
  }
  if (fromCol === toCol) {
    return placedEdges.has(edgeKey("H", Math.min(fromRow, toRow), fromCol));
  }
  return true;
}

function wallEdges(dir, row, col) {
  if (dir === "H") {
    return [edgeKey("H", row, col), edgeKey("H", row, col + 1)];
  }
  return [edgeKey("V", row, col), edgeKey("V", row + 1, col)];
}

function normalizeOrigin(dir, row, col) {
  if (dir === "H") return { row, col: Math.min(col, SIZE - 2) };
  return { row: Math.min(row, SIZE - 2), col };
}

function edgeKey(dir, row, col) {
  return `${dir}:${row}:${col}`;
}

function wallAction(dir, row, col) {
  return `WALL_${dir}_${row}_${col}`;
}

function previewWall(dir, row, col, enabled) {
  const method = enabled ? "add" : "remove";
  wallEdges(dir, row, col).forEach((edge) => {
    const [edgeDir, edgeRow, edgeCol] = edge.split(":");
    const selector = `.wall-slot[data-dir="${edgeDir}"][data-row="${edgeRow}"][data-col="${edgeCol}"]`;
    const element = boardEl.querySelector(selector);
    if (element) element.classList[method]("preview");
  });
  const junction = boardEl.querySelector(`.junction[data-row="${row}"][data-col="${col}"]`);
  if (junction) junction.classList[method]("preview");
}

function renderControls() {
  const disabled = busy || !state || state.bot_turn;
  newGameEl.disabled = busy;
  modeEl.disabled = busy;
  seatEl.disabled = busy || modeEl.value === "human-human";
  botEl.disabled = busy || modeEl.value === "human-human";
  decisionTimeoutEl.disabled = busy || modeEl.value === "human-human";
  wallHEl.disabled = disabled;
  wallVEl.disabled = disabled;
  wallHEl.classList.toggle("selected", selectedWallDirection === "H");
  wallVEl.classList.toggle("selected", selectedWallDirection === "V");
  wallHEl.setAttribute("aria-pressed", String(selectedWallDirection === "H"));
  wallVEl.setAttribute("aria-pressed", String(selectedWallDirection === "V"));
}

function renderLog() {
  logEl.innerHTML = "";
  state.log.slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    logEl.appendChild(li);
  });
}

wallHEl.addEventListener("click", () => {
  selectedWallDirection = "H";
  if (state) render();
  else renderControls();
});
wallVEl.addEventListener("click", () => {
  selectedWallDirection = "V";
  if (state) render();
  else renderControls();
});
newGameEl.addEventListener("click", newGame);
modeEl.addEventListener("change", () => {
  renderControls();
});
botEl.addEventListener("change", changeBot);
boardEl.addEventListener("pointerleave", () => {
  boardEl.querySelectorAll(".preview").forEach((element) => element.classList.remove("preview"));
});

loadBots().then(newGame).catch((error) => {
  statusEl.textContent = error.message;
});
renderControls();

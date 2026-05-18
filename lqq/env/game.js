const SIZE = 9;
const WALLS_PER_PLAYER = 10;

const boardEl = document.getElementById("board");
const statusText = document.getElementById("statusText");
const wallsA = document.getElementById("wallsA");
const wallsB = document.getElementById("wallsB");
const playerA = document.getElementById("playerA");
const playerB = document.getElementById("playerB");
const horizontalBtn = document.getElementById("horizontalBtn");
const verticalBtn = document.getElementById("verticalBtn");
const resetBtn = document.getElementById("resetBtn");
const undoBtn = document.getElementById("undoBtn");
const moveLog = document.getElementById("moveLog");

let selectedWallDirection = "h";
let state = createInitialState();
let history = [];

function createInitialState() {
  return {
    current: 0,
    winner: null,
    players: [
      { name: "蓝方", row: SIZE - 1, col: Math.floor(SIZE / 2), goalRow: 0, walls: WALLS_PER_PLAYER, className: "blue" },
      { name: "红方", row: 0, col: Math.floor(SIZE / 2), goalRow: SIZE - 1, walls: WALLS_PER_PLAYER, className: "red" },
    ],
    placements: new Set(),
    blockedEdges: new Set(),
    log: [],
    message: "蓝方行动",
    isError: false,
  };
}

function snapshot() {
  return {
    current: state.current,
    winner: state.winner,
    players: state.players.map((player) => ({ ...player })),
    placements: new Set(state.placements),
    blockedEdges: new Set(state.blockedEdges),
    log: [...state.log],
    message: state.message,
    isError: state.isError,
  };
}

function restore(saved) {
  state = {
    ...saved,
    players: saved.players.map((player) => ({ ...player })),
    placements: new Set(saved.placements),
    blockedEdges: new Set(saved.blockedEdges),
    log: [...saved.log],
  };
}

function edgeKey(dir, row, col) {
  return `${dir}:${row}:${col}`;
}

function placementKey(dir, row, col) {
  return `${dir}:${row}:${col}`;
}

function otherDirection(dir) {
  return dir === "h" ? "v" : "h";
}

function wallEdges(dir, row, col) {
  if (dir === "h") {
    return [edgeKey("h", row, col), edgeKey("h", row, col + 1)];
  }
  return [edgeKey("v", row, col), edgeKey("v", row + 1, col)];
}

function inBounds(row, col) {
  return row >= 0 && row < SIZE && col >= 0 && col < SIZE;
}

function hasWallBetween(fromRow, fromCol, toRow, toCol) {
  if (fromRow === toRow) {
    return state.blockedEdges.has(edgeKey("v", fromRow, Math.min(fromCol, toCol)));
  }
  if (fromCol === toCol) {
    return state.blockedEdges.has(edgeKey("h", Math.min(fromRow, toRow), fromCol));
  }
  return true;
}

function legalMoves(playerIndex = state.current) {
  const deltas = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];

  return deltas
    .map(([dr, dc]) => moveDestination(playerIndex, dr, dc))
    .filter((move) => move !== null);
}

function moveDestination(playerIndex, dr, dc) {
  const player = state.players[playerIndex];
  const opponent = state.players[1 - playerIndex];
  const row = player.row + dr;
  const col = player.col + dc;

  if (!inBounds(row, col)) return null;
  if (hasWallBetween(player.row, player.col, row, col)) return null;
  if (row !== opponent.row || col !== opponent.col) {
    return { row, col };
  }

  const jumpRow = opponent.row + dr;
  const jumpCol = opponent.col + dc;
  if (!inBounds(jumpRow, jumpCol)) return null;
  if (hasWallBetween(opponent.row, opponent.col, jumpRow, jumpCol)) return null;
  return { row: jumpRow, col: jumpCol };
}

function movePlayer(row, col) {
  if (state.winner !== null) return;
  const player = state.players[state.current];
  const canMove = legalMoves().some((move) => move.row === row && move.col === col);

  if (!canMove) {
    setMessage("不能走到这个格子", true);
    return;
  }

  history.push(snapshot());
  player.row = row;
  player.col = col;
  state.log.unshift(`${player.name} 走到 ${formatCell(row, col)}`);

  if (player.row === player.goalRow) {
    state.winner = state.current;
    setMessage(`${player.name}获胜`, false);
  } else {
    nextTurn();
  }
  render();
}

function validateWall(dir, row, col) {
  const player = state.players[state.current];
  const key = placementKey(dir, row, col);

  if (state.winner !== null) return { ok: false, message: "对局已经结束" };
  if (player.walls <= 0) return { ok: false, message: `${player.name}已经没有墙了` };
  if (row < 0 || row >= SIZE - 1 || col < 0 || col >= SIZE - 1) {
    return { ok: false, message: "墙必须放在棋盘内部的墙槽上" };
  }
  if (state.placements.has(key)) return { ok: false, message: "这里已经有墙了" };
  if (state.placements.has(placementKey(otherDirection(dir), row, col))) {
    return { ok: false, message: "墙不能交叉摆放" };
  }
  if (wallEdges(dir, row, col).some((edge) => state.blockedEdges.has(edge))) {
    return { ok: false, message: "墙不能重叠摆放" };
  }

  addWall(dir, row, col);
  const hasPaths = playersHavePaths();
  removeWall(dir, row, col);

  if (!hasPaths) return { ok: false, message: "不能完全堵死任一方的路线" };
  return { ok: true, message: "" };
}

function placeWall(dir, row, col) {
  const validation = validateWall(dir, row, col);
  if (!validation.ok) {
    setMessage(validation.message, true);
    render();
    return;
  }

  history.push(snapshot());
  const player = state.players[state.current];
  addWall(dir, row, col);
  player.walls -= 1;
  state.log.unshift(`${player.name} 放置${dir === "h" ? "横墙" : "竖墙"} ${formatWall(row, col)}`);
  nextTurn();
  render();
}

function addWall(dir, row, col) {
  state.placements.add(placementKey(dir, row, col));
  wallEdges(dir, row, col).forEach((edge) => state.blockedEdges.add(edge));
}

function removeWall(dir, row, col) {
  state.placements.delete(placementKey(dir, row, col));
  wallEdges(dir, row, col).forEach((edge) => state.blockedEdges.delete(edge));
}

function playersHavePaths() {
  return state.players.every((_, index) => hasPathToGoal(index));
}

function hasPathToGoal(playerIndex) {
  const player = state.players[playerIndex];
  const visited = new Set([`${player.row},${player.col}`]);
  const queue = [{ row: player.row, col: player.col }];

  while (queue.length > 0) {
    const current = queue.shift();
    if (current.row === player.goalRow) return true;

    for (const next of pathNeighbors(current.row, current.col)) {
      const key = `${next.row},${next.col}`;
      if (!visited.has(key)) {
        visited.add(key);
        queue.push(next);
      }
    }
  }

  return false;
}

function pathNeighbors(row, col) {
  const deltas = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];

  return deltas
    .map(([dr, dc]) => ({ row: row + dr, col: col + dc }))
    .filter((next) => inBounds(next.row, next.col) && !hasWallBetween(row, col, next.row, next.col));
}

function nextTurn() {
  state.current = 1 - state.current;
  setMessage(`${state.players[state.current].name}行动`, false);
}

function setMessage(message, isError) {
  state.message = message;
  state.isError = isError;
}

function formatCell(row, col) {
  return `${String.fromCharCode(65 + col)}${row + 1}`;
}

function formatWall(row, col) {
  return `${String.fromCharCode(65 + col)}${row + 1}`;
}

function render() {
  renderBoard();
  renderPanel();
}

function renderBoard() {
  boardEl.innerHTML = "";
  const moves = legalMoves();
  const moveKeys = new Set(moves.map((move) => `${move.row},${move.col}`));

  for (let gridRow = 0; gridRow < SIZE * 2 - 1; gridRow += 1) {
    for (let gridCol = 0; gridCol < SIZE * 2 - 1; gridCol += 1) {
      if (gridRow % 2 === 0 && gridCol % 2 === 0) {
        boardEl.appendChild(createCell(gridRow / 2, gridCol / 2, moveKeys));
      } else if (gridRow % 2 === 1 && gridCol % 2 === 0) {
        boardEl.appendChild(createWallSlot("h", (gridRow - 1) / 2, gridCol / 2));
      } else if (gridRow % 2 === 0 && gridCol % 2 === 1) {
        boardEl.appendChild(createWallSlot("v", gridRow / 2, (gridCol - 1) / 2));
      } else {
        boardEl.appendChild(createJunction((gridRow - 1) / 2, (gridCol - 1) / 2));
      }
    }
  }
}

function createCell(row, col, moveKeys) {
  const cell = document.createElement("button");
  cell.type = "button";
  cell.className = "cell";
  cell.ariaLabel = `格子 ${formatCell(row, col)}`;

  if (row === 0) cell.classList.add("goal-blue");
  if (row === SIZE - 1) cell.classList.add("goal-red");

  const playerIndex = state.players.findIndex((player) => player.row === row && player.col === col);
  if (playerIndex >= 0) {
    const pawn = document.createElement("span");
    pawn.className = `pawn ${state.players[playerIndex].className}`;
    cell.appendChild(pawn);
  }

  if (state.winner === null && moveKeys.has(`${row},${col}`)) {
    cell.classList.add("legal-move");
    cell.addEventListener("click", () => movePlayer(row, col));
  }

  return cell;
}

function createWallSlot(dir, row, col) {
  const slot = document.createElement("button");
  slot.type = "button";
  slot.className = `wall-slot ${dir === "h" ? "horizontal" : "vertical"}`;
  slot.dataset.row = String(row);
  slot.dataset.col = String(col);

  if (state.blockedEdges.has(edgeKey(dir, row, col))) {
    slot.classList.add("placed");
  }

  const origin = normalizeOrigin(dir, row, col);
  const validation = validateWallForRender(dir, origin.row, origin.col);
  if (validation.ok) {
    slot.classList.add("placeable");
    slot.addEventListener("mouseenter", () => previewWall(dir, origin.row, origin.col, true));
    slot.addEventListener("mouseleave", () => previewWall(dir, origin.row, origin.col, false));
    slot.addEventListener("click", () => placeWall(dir, origin.row, origin.col));
  }

  return slot;
}

function createJunction(row, col) {
  const junction = document.createElement("button");
  junction.type = "button";
  junction.className = "junction";
  junction.dataset.row = String(row);
  junction.dataset.col = String(col);

  if (
    state.placements.has(placementKey("h", row, col)) ||
    state.placements.has(placementKey("v", row, col))
  ) {
    junction.classList.add("placed");
  }

  const validation = validateWallForRender(selectedWallDirection, row, col);
  if (validation.ok) {
    junction.classList.add("placeable");
    junction.addEventListener("mouseenter", () => previewWall(selectedWallDirection, row, col, true));
    junction.addEventListener("mouseleave", () => previewWall(selectedWallDirection, row, col, false));
    junction.addEventListener("click", () => placeWall(selectedWallDirection, row, col));
  }

  return junction;
}

function normalizeOrigin(dir, row, col) {
  if (dir === "h") return { row, col: Math.min(col, SIZE - 2) };
  return { row: Math.min(row, SIZE - 2), col };
}

function validateWallForRender(dir, row, col) {
  if (state.winner !== null) return { ok: false };
  return validateWall(dir, row, col);
}

function previewWall(dir, row, col, enabled) {
  const classAction = enabled ? "add" : "remove";
  wallEdges(dir, row, col).forEach((edge) => {
    const [edgeDir, edgeRow, edgeCol] = edge.split(":");
    const selector = edgeDir === "h"
      ? `.wall-slot.horizontal[data-row="${edgeRow}"][data-col="${edgeCol}"]`
      : `.wall-slot.vertical[data-row="${edgeRow}"][data-col="${edgeCol}"]`;
    const element = boardEl.querySelector(selector);
    if (element) element.classList[classAction]("preview");
  });
  const junction = boardEl.querySelector(`.junction[data-row="${row}"][data-col="${col}"]`);
  if (junction) junction.classList[classAction]("preview");
}

function renderPanel() {
  statusText.textContent = state.message;
  statusText.classList.toggle("error", state.isError);
  wallsA.textContent = state.players[0].walls;
  wallsB.textContent = state.players[1].walls;
  playerA.classList.toggle("active", state.current === 0 && state.winner === null);
  playerB.classList.toggle("active", state.current === 1 && state.winner === null);
  undoBtn.disabled = history.length === 0;

  moveLog.innerHTML = "";
  state.log.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    moveLog.appendChild(li);
  });
}

function resetGame() {
  state = createInitialState();
  history = [];
  render();
}

function undoMove() {
  const previous = history.pop();
  if (!previous) return;
  restore(previous);
  render();
}

function setWallDirection(dir) {
  selectedWallDirection = dir;
  horizontalBtn.classList.toggle("selected", dir === "h");
  verticalBtn.classList.toggle("selected", dir === "v");
  horizontalBtn.setAttribute("aria-pressed", String(dir === "h"));
  verticalBtn.setAttribute("aria-pressed", String(dir === "v"));
  render();
}

horizontalBtn.addEventListener("click", () => setWallDirection("h"));
verticalBtn.addEventListener("click", () => setWallDirection("v"));
resetBtn.addEventListener("click", resetGame);
undoBtn.addEventListener("click", undoMove);

boardEl.addEventListener("pointerleave", () => {
  boardEl.querySelectorAll(".preview").forEach((element) => element.classList.remove("preview"));
});

render();

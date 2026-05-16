const boardEl = document.getElementById("board");
const statusText = document.getElementById("statusText");
const newGameBtn = document.getElementById("newGameBtn");
const flipBtn = document.getElementById("flipBtn");
const humanHumanBtn = document.getElementById("humanHumanBtn");
const humanBotBtn = document.getElementById("humanBotBtn");
const whiteSeatBtn = document.getElementById("whiteSeatBtn");
const blackSeatBtn = document.getElementById("blackSeatBtn");
const botSelect = document.getElementById("botSelect");
const animationSpeedSelect = document.getElementById("animationSpeedSelect");
const whitePlayer = document.getElementById("whitePlayer");
const blackPlayer = document.getElementById("blackPlayer");
const whiteType = document.getElementById("whiteType");
const blackType = document.getElementById("blackType");
const fenText = document.getElementById("fenText");
const moveLog = document.getElementById("moveLog");
const promotionPanel = document.getElementById("promotionPanel");
const promotionChoices = document.getElementById("promotionChoices");

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const PIECES = {
  white: { k: "♔", q: "♕", r: "♖", b: "♗", n: "♘", p: "♙" },
  black: { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" },
};
const PROMOTION_LABELS = { q: "♕", r: "♖", b: "♗", n: "♘" };

let sessionId = "";
let data = null;
let selectedSquare = null;
let orientation = "white";
let mode = "human-human";
let humanSeat = 0;
let botId = "/gpt5p5/bot_hard";
let advanceTimer = 0;
let advanceInFlight = false;
let activeMoveAnimations = [];

async function api(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : {};
  if (!response.ok) throw new Error(body.error || "request failed");
  return body;
}

async function loadBotOptions() {
  try {
    const response = await fetch("/api/bots");
    if (!response.ok) return;
    const body = await response.json();
    if (!Array.isArray(body.bots) || body.bots.length === 0) return;
    botSelect.innerHTML = "";
    body.bots.forEach((bot) => {
      const option = document.createElement("option");
      option.value = bot;
      option.textContent = bot;
      botSelect.appendChild(option);
    });
    botId = body.default || body.bots[0];
    renderModeControls();
  } catch (error) {
    console.error(error);
  }
}

async function newGame() {
  try {
    clearPendingAdvance();
    clearMoveAnimations();
    selectedSquare = null;
    promotionPanel.hidden = true;
    const body = await api("/api/new", { mode, human_seat: humanSeat, bot: botId });
    const previous = data;
    sessionId = body.session;
    data = body;
    if (mode === "human-bot") orientation = humanSeat === 0 ? "white" : "black";
    render(previous);
    queueBotAdvance(0);
  } catch (error) {
    showServerError(error);
  }
}

async function sendMove(action) {
  try {
    clearPendingAdvance();
    selectedSquare = null;
    promotionPanel.hidden = true;
    const previous = data;
    const body = await api("/api/action", { session: sessionId, action });
    data = body;
    render(previous);
    queueBotAdvance(getAnimationDuration() + 80);
  } catch (error) {
    showServerError(error);
  }
}

async function advanceBot() {
  if (!data || !data.bot_turn || advanceInFlight) return;
  try {
    advanceInFlight = true;
    const previous = data;
    const body = await api("/api/advance", { session: sessionId });
    data = body;
    render(previous);
    queueBotAdvance(getAnimationDuration() + 80);
  } catch (error) {
    showServerError(error);
  } finally {
    advanceInFlight = false;
  }
}

function showServerError(error) {
  clearPendingAdvance();
  clearMoveAnimations();
  data = null;
  selectedSquare = null;
  promotionPanel.hidden = true;
  boardEl.innerHTML = "";
  fenText.textContent = "";
  moveLog.innerHTML = "";
  statusText.textContent = "请用 python BoardArena/chess/env/chess_web.py 启动；Live Server 没有 /api/new 和 /api/action。";
  console.error(error);
}

function render(previousData = null) {
  if (!data) return;
  statusText.textContent = data.status_text;
  fenText.textContent = data.fen;
  whitePlayer.classList.toggle("active", data.actor === 0 && data.phase !== "game_over");
  blackPlayer.classList.toggle("active", data.actor === 1 && data.phase !== "game_over");
  const botName = data.bot_name ? `${data.bot_name} Bot` : "Bot";
  whiteType.textContent = data.mode === "human-human" || data.human_seats.includes(0) ? "人类" : botName;
  blackType.textContent = data.mode === "human-human" || data.human_seats.includes(1) ? "人类" : botName;
  renderModeControls();
  renderBoard();
  renderLog();
  animateLastMove(previousData, data);
}

function renderModeControls() {
  humanHumanBtn.classList.toggle("selected", mode === "human-human");
  humanBotBtn.classList.toggle("selected", mode === "human-bot");
  humanHumanBtn.setAttribute("aria-pressed", String(mode === "human-human"));
  humanBotBtn.setAttribute("aria-pressed", String(mode === "human-bot"));
  whiteSeatBtn.classList.toggle("selected", humanSeat === 0);
  blackSeatBtn.classList.toggle("selected", humanSeat === 1);
  whiteSeatBtn.setAttribute("aria-pressed", String(humanSeat === 0));
  blackSeatBtn.setAttribute("aria-pressed", String(humanSeat === 1));
  botSelect.value = botId;
  botSelect.disabled = mode !== "human-bot";
}

function renderBoard() {
  boardEl.innerHTML = "";
  const pieceMap = new Map(data.pieces.map((piece) => [piece.square, piece]));
  const legal = data.legal_actions || [];
  const legalFrom = new Set(legal.map((action) => action.slice(0, 2)));
  const targets = selectedSquare
    ? new Set(legal.filter((action) => action.startsWith(selectedSquare)).map((action) => action.slice(2, 4)))
    : new Set();
  const lastSquares = data.last_move ? new Set([data.last_move.slice(0, 2), data.last_move.slice(2, 4)]) : new Set();
  const files = orientation === "white" ? FILES : [...FILES].reverse();
  const ranks = orientation === "white" ? [...RANKS].reverse() : RANKS;

  for (const rank of ranks) {
    for (const file of files) {
      const square = `${file}${rank}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `square ${isLightSquare(file, rank) ? "light" : "dark"}`;
      button.dataset.square = square;
      button.setAttribute("aria-label", square);

      if (lastSquares.has(square)) button.classList.add("last");
      if (selectedSquare === square) button.classList.add("selected");
      if (targets.has(square)) button.classList.add("target");
      if (data.human_turn && (legalFrom.has(square) || targets.has(square))) {
        button.classList.add("selectable");
        button.addEventListener("click", () => onSquareClick(square));
      }

      const piece = pieceMap.get(square);
      if (piece) {
        const span = document.createElement("span");
        span.className = "piece";
        span.dataset.square = square;
        span.textContent = PIECES[piece.color][piece.type];
        button.appendChild(span);
      }

      if (file === files[0]) {
        const rankLabel = document.createElement("span");
        rankLabel.className = "coord rank";
        rankLabel.textContent = rank;
        button.appendChild(rankLabel);
      }
      if (rank === ranks[ranks.length - 1]) {
        const fileLabel = document.createElement("span");
        fileLabel.className = "coord file";
        fileLabel.textContent = file;
        button.appendChild(fileLabel);
      }

      boardEl.appendChild(button);
    }
  }
}

function renderLog() {
  moveLog.innerHTML = "";
  [...data.log].reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    moveLog.appendChild(li);
  });
}

function onSquareClick(square) {
  const legal = data.legal_actions || [];
  const legalFrom = new Set(legal.map((action) => action.slice(0, 2)));

  if (!selectedSquare) {
    if (legalFrom.has(square)) {
      selectedSquare = square;
      renderBoard();
    }
    return;
  }

  if (square === selectedSquare) {
    selectedSquare = null;
    renderBoard();
    return;
  }

  const candidates = legal.filter((action) => action.slice(0, 2) === selectedSquare && action.slice(2, 4) === square);
  if (candidates.length === 0) {
    selectedSquare = legalFrom.has(square) ? square : null;
    renderBoard();
    return;
  }
  if (candidates.length === 1) {
    sendMove(candidates[0]);
    return;
  }
  showPromotion(candidates);
}

function showPromotion(candidates) {
  promotionChoices.innerHTML = "";
  candidates
    .filter((action) => action.length === 5)
    .sort()
    .forEach((action) => {
      const piece = action[4];
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = PROMOTION_LABELS[piece] || piece;
      button.addEventListener("click", () => sendMove(action));
      promotionChoices.appendChild(button);
    });
  promotionPanel.hidden = false;
}

function isLightSquare(file, rank) {
  const fileIndex = FILES.indexOf(file);
  const rankIndex = RANKS.indexOf(rank);
  return (fileIndex + rankIndex) % 2 === 1;
}

function setMode(nextMode) {
  mode = nextMode;
  renderModeControls();
  newGame();
}

function setHumanSeat(nextSeat) {
  humanSeat = nextSeat;
  renderModeControls();
  if (mode === "human-bot") newGame();
}

function setBot(nextBot) {
  botId = nextBot;
  renderModeControls();
  if (mode === "human-bot") newGame();
}

function queueBotAdvance(delay) {
  clearPendingAdvance();
  if (!data || !data.bot_turn) return;
  advanceTimer = window.setTimeout(advanceBot, Math.max(0, delay));
}

function clearPendingAdvance() {
  if (advanceTimer) {
    window.clearTimeout(advanceTimer);
    advanceTimer = 0;
  }
}

function getAnimationDuration() {
  return Number(animationSpeedSelect.value) || 0;
}

function animateLastMove(previousData, nextData) {
  clearMoveAnimations();
  const duration = getAnimationDuration();
  if (!previousData || !nextData || duration <= 0) return;
  if (!nextData.last_move || previousData.plies === nextData.plies) return;

  const fromSquare = nextData.last_move.slice(0, 2);
  const toSquare = nextData.last_move.slice(2, 4);
  const fromEl = boardEl.querySelector(`.square[data-square="${fromSquare}"]`);
  const toPiece = boardEl.querySelector(`.piece[data-square="${toSquare}"]`);
  if (!fromEl || !toPiece) return;

  const fromRect = fromEl.getBoundingClientRect();
  const toRect = toPiece.getBoundingClientRect();
  const startLeft = fromRect.left + (fromRect.width - toRect.width) / 2;
  const startTop = fromRect.top + (fromRect.height - toRect.height) / 2;
  const clone = toPiece.cloneNode(true);
  const style = window.getComputedStyle(toPiece);

  clone.classList.remove("animating");
  clone.classList.add("piece-clone");
  clone.style.left = `${startLeft}px`;
  clone.style.top = `${startTop}px`;
  clone.style.width = `${toRect.width}px`;
  clone.style.height = `${toRect.height}px`;
  clone.style.fontSize = style.fontSize;
  clone.style.color = style.color;
  document.body.appendChild(clone);
  toPiece.classList.add("animating");

  const animation = clone.animate(
    [
      { transform: "translate(0, 0)" },
      { transform: `translate(${toRect.left - startLeft}px, ${toRect.top - startTop}px)` },
    ],
    { duration, easing: "cubic-bezier(0.22, 1, 0.36, 1)" }
  );

  const cleanup = () => {
    toPiece.classList.remove("animating");
    clone.remove();
    activeMoveAnimations = activeMoveAnimations.filter((item) => item.animation !== animation);
  };
  animation.onfinish = cleanup;
  animation.oncancel = cleanup;
  activeMoveAnimations.push({ animation, clone, target: toPiece });
}

function clearMoveAnimations() {
  activeMoveAnimations.forEach(({ animation, clone, target }) => {
    animation.cancel();
    clone.remove();
    target.classList.remove("animating");
  });
  activeMoveAnimations = [];
}

humanHumanBtn.addEventListener("click", () => setMode("human-human"));
humanBotBtn.addEventListener("click", () => setMode("human-bot"));
whiteSeatBtn.addEventListener("click", () => setHumanSeat(0));
blackSeatBtn.addEventListener("click", () => setHumanSeat(1));
botSelect.addEventListener("change", () => setBot(botSelect.value));
newGameBtn.addEventListener("click", newGame);
flipBtn.addEventListener("click", () => {
  orientation = orientation === "white" ? "black" : "white";
  renderBoard();
});
promotionPanel.addEventListener("click", (event) => {
  if (event.target === promotionPanel) promotionPanel.hidden = true;
});

loadBotOptions().finally(newGame);

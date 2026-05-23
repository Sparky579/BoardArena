const boardEl = document.getElementById("board");
const statusText = document.getElementById("statusText");
const newGameBtn = document.getElementById("newGameBtn");
const flipBtn = document.getElementById("flipBtn");
const humanHumanBtn = document.getElementById("humanHumanBtn");
const humanBotBtn = document.getElementById("humanBotBtn");
const blackSeatBtn = document.getElementById("blackSeatBtn");
const whiteSeatBtn = document.getElementById("whiteSeatBtn");
const botSelect = document.getElementById("botSelect");
const decisionTimeoutSelect = document.getElementById("decisionTimeoutSelect");
const orientationSelect = document.getElementById("orientationSelect");
const blackPlayer = document.getElementById("blackPlayer");
const whitePlayer = document.getElementById("whitePlayer");
const blackType = document.getElementById("blackType");
const whiteType = document.getElementById("whiteType");
const blackScore = document.getElementById("blackScore");
const whiteScore = document.getElementById("whiteScore");
const passBtn = document.getElementById("passBtn");
const boardText = document.getElementById("boardText");
const moveLog = document.getElementById("moveLog");

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const PASS_ACTION = "PASS";

let sessionId = "";
let data = null;
let mode = "human-bot";
let humanSeat = 0;
let botId = "/gpt5p5/bot_easy";
let decisionTimeout = 1;
let orientation = "black";
let advanceTimer = 0;
let advanceInFlight = false;
let stateVersion = 0;

async function api(path, payload) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    error.path = path;
    error.isNetworkError = true;
    throw error;
  }

  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : {};
  if (!response.ok) {
    const error = new Error(body.error || `${path} failed with ${response.status}`);
    error.path = path;
    error.status = response.status;
    error.hasJsonBody = contentType.includes("application/json");
    throw error;
  }
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
  const version = stateVersion + 1;
  stateVersion = version;
  try {
    clearPendingAdvance();
    advanceInFlight = false;
    const body = await api("/api/new", { mode, human_seat: humanSeat, bot: botId, decision_timeout: decisionTimeout });
    if (version !== stateVersion) return;
    sessionId = body.session;
    data = body;
    if (mode === "human-bot") orientation = humanSeat === 0 ? "black" : "white";
    orientationSelect.value = orientation;
    render();
    queueBotAdvance(120);
  } catch (error) {
    if (version !== stateVersion) return;
    showRequestError(error, { clear: true });
  }
}

async function sendAction(action) {
  if (!canSendAction(action)) return;
  const version = stateVersion;
  try {
    clearPendingAdvance();
    const body = await api("/api/action", { session: sessionId, action });
    if (version !== stateVersion) return;
    data = body;
    render();
    queueBotAdvance(180);
  } catch (error) {
    if (version !== stateVersion || isBenignTurnError(error)) return;
    showRequestError(error, { clear: false });
  }
}

async function advanceBot() {
  if (!data || !data.bot_turn || advanceInFlight) return;
  const version = stateVersion;
  const activeSession = sessionId;
  try {
    advanceInFlight = true;
    const body = await api("/api/advance", { session: activeSession, decision_timeout: decisionTimeout });
    if (version !== stateVersion || activeSession !== sessionId) return;
    data = body;
    render();
    queueBotAdvance(220);
  } catch (error) {
    if (version !== stateVersion || activeSession !== sessionId) return;
    showRequestError(error, { clear: false });
  } finally {
    if (version === stateVersion && activeSession === sessionId) {
      advanceInFlight = false;
    }
  }
}

function canSendAction(action) {
  return Boolean(
    data
    && data.human_turn
    && !advanceInFlight
    && Array.isArray(data.legal_actions)
    && data.legal_actions.includes(action)
  );
}

function isBenignTurnError(error) {
  return error && typeof error.message === "string" && error.message.includes("现在不是人类玩家回合");
}

function showRequestError(error, options = {}) {
  clearPendingAdvance();
  if (options.clear) {
    data = null;
    boardEl.innerHTML = "";
    boardText.textContent = "";
    moveLog.innerHTML = "";
  }
  const apiPath = error.path || "";
  const isMissingApi = error.status === 404 && apiPath.startsWith("/api/") && !error.hasJsonBody;
  if (isMissingApi) {
    statusText.textContent = "请用 python BoardArena/Othello/env/othello_web.py 启动；Live Server 没有本地 JSON API。";
  } else if (error.isNetworkError) {
    statusText.textContent = "连接本地 othello_web.py 服务失败，请确认服务仍在运行。";
  } else {
    statusText.textContent = `请求失败：${error.message}`;
  }
  console.error(error);
}

function render() {
  if (!data) return;
  statusText.textContent = data.status_text;
  blackScore.textContent = data.disc_counts.black;
  whiteScore.textContent = data.disc_counts.white;
  blackPlayer.classList.toggle("active", data.actor === 0 && data.phase !== "game_over");
  whitePlayer.classList.toggle("active", data.actor === 1 && data.phase !== "game_over");
  const botName = data.bot_name ? `${data.bot_name} Bot` : "Bot";
  blackType.textContent = data.mode === "human-human" || data.human_seats.includes(0) ? "人类" : botName;
  whiteType.textContent = data.mode === "human-human" || data.human_seats.includes(1) ? "人类" : botName;
  boardText.textContent = data.board.join("\n");
  passBtn.hidden = !(data.human_turn && data.legal_actions.length === 1 && data.legal_actions[0] === PASS_ACTION);
  passBtn.disabled = passBtn.hidden;
  renderModeControls();
  renderBoard();
  renderLog();
}

function renderModeControls() {
  humanHumanBtn.classList.toggle("selected", mode === "human-human");
  humanBotBtn.classList.toggle("selected", mode === "human-bot");
  humanHumanBtn.setAttribute("aria-pressed", String(mode === "human-human"));
  humanBotBtn.setAttribute("aria-pressed", String(mode === "human-bot"));
  blackSeatBtn.classList.toggle("selected", humanSeat === 0);
  whiteSeatBtn.classList.toggle("selected", humanSeat === 1);
  blackSeatBtn.setAttribute("aria-pressed", String(humanSeat === 0));
  whiteSeatBtn.setAttribute("aria-pressed", String(humanSeat === 1));
  botSelect.value = botId;
  botSelect.disabled = mode !== "human-bot";
  decisionTimeoutSelect.value = String(decisionTimeout);
  decisionTimeoutSelect.disabled = mode !== "human-bot";
  orientationSelect.value = orientation;
}

function renderBoard() {
  const fragment = document.createDocumentFragment();
  const pieceMap = new Map((data.pieces || []).map((piece) => [piece.square, piece]));
  const legal = new Set((data.legal_actions || []).filter((action) => action !== PASS_ACTION));
  const lastMove = data.last_move && data.last_move !== PASS_ACTION ? data.last_move : "";
  const files = orientation === "black" ? FILES : [...FILES].reverse();
  const ranks = orientation === "black" ? [...RANKS].reverse() : RANKS;

  for (const rank of ranks) {
    for (const file of files) {
      const square = `${file}${rank}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "square";
      button.dataset.square = square;
      button.setAttribute("aria-label", square);

      if (lastMove === square) button.classList.add("last");
      if (legal.has(square)) {
        button.classList.add("legal");
        button.addEventListener("click", () => sendAction(square));
      }

      const piece = pieceMap.get(square);
      if (piece) {
        const disc = document.createElement("span");
        disc.className = `disc ${piece.color}`;
        disc.setAttribute("aria-label", `${piece.color === "black" ? "黑子" : "白子"} ${square}`);
        button.appendChild(disc);
      } else if (legal.has(square)) {
        const hint = document.createElement("span");
        hint.className = `hint ${data.turn}`;
        button.appendChild(hint);
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

      fragment.appendChild(button);
    }
  }
  boardEl.replaceChildren(fragment);
}

function renderLog() {
  moveLog.innerHTML = "";
  [...data.log].reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    moveLog.appendChild(li);
  });
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

function setDecisionTimeout(nextTimeout) {
  decisionTimeout = Number(nextTimeout) || 1;
  renderModeControls();
}

function setOrientation(nextOrientation) {
  orientation = nextOrientation;
  renderModeControls();
  renderBoard();
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

humanHumanBtn.addEventListener("click", () => setMode("human-human"));
humanBotBtn.addEventListener("click", () => setMode("human-bot"));
blackSeatBtn.addEventListener("click", () => setHumanSeat(0));
whiteSeatBtn.addEventListener("click", () => setHumanSeat(1));
botSelect.addEventListener("change", (event) => setBot(event.target.value));
decisionTimeoutSelect.addEventListener("change", (event) => setDecisionTimeout(event.target.value));
orientationSelect.addEventListener("change", (event) => setOrientation(event.target.value));
newGameBtn.addEventListener("click", newGame);
flipBtn.addEventListener("click", () => setOrientation(orientation === "black" ? "white" : "black"));
passBtn.addEventListener("click", () => sendAction(PASS_ACTION));

loadBotOptions().finally(newGame);


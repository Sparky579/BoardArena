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
const pieceStyleSelect = document.getElementById("pieceStyleSelect");
const orientationSelect = document.getElementById("orientationSelect");
const whitePlayer = document.getElementById("whitePlayer");
const blackPlayer = document.getElementById("blackPlayer");
const whiteType = document.getElementById("whiteType");
const blackType = document.getElementById("blackType");
const whiteMark = document.getElementById("whiteMark");
const blackMark = document.getElementById("blackMark");
const fenText = document.getElementById("fenText");
const moveLog = document.getElementById("moveLog");
const promotionPanel = document.getElementById("promotionPanel");
const promotionChoices = document.getElementById("promotionChoices");

const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const RANKS = ["1", "2", "3", "4", "5", "6", "7", "8"];
const PIECE_SETS = ["chessnut", "spatial", "cburnett", "merida", "rhosgfx"];
const PIECE_CODES = ["K", "Q", "R", "B", "N", "P"];
const DEFAULT_PIECE_SET = "cburnett";
const PIECE_SET_STORAGE_KEY = "boardarena_chess_piece_set";
const PIECE_NAMES = { k: "王", q: "后", r: "车", b: "象", n: "马", p: "兵" };
const PROMOTION_NAMES = { q: "后", r: "车", b: "象", n: "马" };

let sessionId = "";
let data = null;
let selectedSquare = null;
let orientation = "white";
let mode = "human-bot";
let humanSeat = 0;
let botId = "/gpt5p5/bot_hard";
let pieceSet = loadSavedPieceSet();
let advanceTimer = 0;
let advanceInFlight = false;
let activeMoveAnimations = [];
let stateVersion = 0;
const piecePreloadPromises = new Map();

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
    clearMoveAnimations();
    advanceInFlight = false;
    selectedSquare = null;
    promotionPanel.hidden = true;
    const body = await api("/api/new", { mode, human_seat: humanSeat, bot: botId });
    if (version !== stateVersion) return;
    const previous = data;
    sessionId = body.session;
    data = body;
    if (mode === "human-bot") orientation = humanSeat === 0 ? "white" : "black";
    orientationSelect.value = orientation;
    render(previous);
    queueBotAdvance(0);
  } catch (error) {
    if (version !== stateVersion) return;
    showRequestError(error, { clear: true });
  }
}

async function sendMove(action) {
  if (!canSendAction(action)) return;
  const version = stateVersion;
  const previous = data;
  try {
    clearPendingAdvance();
    selectedSquare = null;
    promotionPanel.hidden = true;
    const optimistic = buildOptimisticMove(previous, action);
    if (optimistic) {
      data = optimistic;
      render(previous);
    }
    const animationDelay = optimistic ? getAnimationDuration() : 0;
    const body = await api("/api/action", { session: sessionId, action });
    if (version !== stateVersion) return;
    if (animationDelay > 0) await sleep(animationDelay);
    if (version !== stateVersion) return;
    data = body;
    render(null);
    queueBotAdvance(80);
  } catch (error) {
    if (version !== stateVersion || isBenignTurnError(error)) return;
    if (previous) {
      data = previous;
      render(null);
    }
    showRequestError(error, { clear: false });
  }
}

async function advanceBot() {
  if (!data || !data.bot_turn || advanceInFlight) return;
  const version = stateVersion;
  const activeSession = sessionId;
  try {
    advanceInFlight = true;
    const previous = data;
    const body = await api("/api/advance", { session: activeSession });
    if (version !== stateVersion || activeSession !== sessionId) return;
    data = body;
    render(previous);
    queueBotAdvance(getAnimationDuration() + 80);
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
  clearMoveAnimations();
  selectedSquare = null;
  promotionPanel.hidden = true;
  const apiPath = error.path || "";
  const isMissingApi = error.status === 404 && apiPath.startsWith("/api/") && !error.hasJsonBody;
  if (options.clear || isMissingApi) {
    data = null;
    boardEl.innerHTML = "";
    fenText.textContent = "";
    moveLog.innerHTML = "";
  }
  if (isMissingApi) {
    statusText.textContent = "请用 python BoardArena/chess/env/chess_web.py 启动；Live Server 没有本地 JSON API。";
  } else if (error.isNetworkError) {
    statusText.textContent = "连接本地 chess_web.py 服务失败，请确认服务仍在运行。";
  } else {
    statusText.textContent = `请求失败：${error.message}`;
  }
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
  renderPieceMarks();
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
  pieceStyleSelect.value = pieceSet;
  orientationSelect.value = orientation;
}

function renderBoard() {
  if (!data || !Array.isArray(data.pieces)) return;
  const fragment = document.createDocumentFragment();
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
        const img = createPieceImage(piece.color, piece.type, `${pieceAlt(piece)} ${square}`);
        img.dataset.square = square;
        button.appendChild(img);
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
      button.setAttribute("aria-label", `升变为${PROMOTION_NAMES[piece] || piece}`);
      button.title = `升变为${PROMOTION_NAMES[piece] || piece}`;
      button.appendChild(createPieceImage(data.turn, piece, PROMOTION_NAMES[piece] || piece));
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

function createPieceImage(color, type, altText) {
  const img = document.createElement("img");
  img.className = "piece";
  img.decoding = "async";
  img.loading = "eager";
  img.src = pieceImagePath(color, type);
  img.alt = altText || "";
  img.draggable = false;
  return img;
}

function pieceImagePath(color, type, setName = pieceSet) {
  const prefix = color === "white" ? "w" : "b";
  return `assets/pieces/${setName}/${prefix}${type.toUpperCase()}.svg`;
}

function pieceAlt(piece) {
  const colorName = piece.color === "white" ? "白方" : "黑方";
  return `${colorName}${PIECE_NAMES[piece.type] || piece.type}`;
}

function renderPieceMarks() {
  whiteMark.src = pieceImagePath("white", "k");
  blackMark.src = pieceImagePath("black", "k");
}

function preloadPieceSet(setName = pieceSet) {
  if (piecePreloadPromises.has(setName)) return piecePreloadPromises.get(setName);

  const paths = [];
  for (const prefix of ["w", "b"]) {
    for (const pieceCode of PIECE_CODES) {
      paths.push(`assets/pieces/${setName}/${prefix}${pieceCode}.svg`);
    }
  }

  const promise = Promise.all(paths.map(preloadImage)).catch((error) => {
    console.error(error);
  });
  piecePreloadPromises.set(setName, promise);
  return promise;
}

function preloadImage(path) {
  return new Promise((resolve) => {
    const img = new Image();
    img.decoding = "async";
    img.loading = "eager";
    img.onload = resolve;
    img.onerror = resolve;
    img.src = path;
    if (img.decode) img.decode().then(resolve).catch(resolve);
  });
}

function preloadOtherPieceSets() {
  const run = () => {
    PIECE_SETS.filter((setName) => setName !== pieceSet).forEach((setName) => preloadPieceSet(setName));
  };
  if ("requestIdleCallback" in window) {
    window.requestIdleCallback(run, { timeout: 2000 });
  } else {
    window.setTimeout(run, 800);
  }
}

function loadSavedPieceSet() {
  try {
    const saved = window.localStorage.getItem(PIECE_SET_STORAGE_KEY);
    return PIECE_SETS.includes(saved) ? saved : DEFAULT_PIECE_SET;
  } catch (error) {
    return DEFAULT_PIECE_SET;
  }
}

function savePieceSet() {
  try {
    window.localStorage.setItem(PIECE_SET_STORAGE_KEY, pieceSet);
  } catch (error) {
    console.error(error);
  }
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

async function setPieceSet(nextPieceSet) {
  if (!PIECE_SETS.includes(nextPieceSet)) return;
  clearMoveAnimations();
  pieceSet = nextPieceSet;
  savePieceSet();
  renderModeControls();
  await preloadPieceSet(pieceSet);
  if (pieceSet !== nextPieceSet) return;
  renderPieceMarks();
  renderBoard();
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

function getAnimationDuration() {
  return Number(animationSpeedSelect.value) || 0;
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function buildOptimisticMove(current, action) {
  if (!current || !Array.isArray(current.pieces)) return null;
  const fromSquare = action.slice(0, 2);
  const toSquare = action.slice(2, 4);
  const moved = current.pieces.find((piece) => piece.square === fromSquare);
  if (!moved) return null;

  const captured = current.pieces.find((piece) => piece.square === toSquare);
  let capturedSquare = captured ? toSquare : null;
  if (!captured && moved.type === "p" && fromSquare[0] !== toSquare[0]) {
    const capturedRank = moved.color === "white" ? Number(toSquare[1]) - 1 : Number(toSquare[1]) + 1;
    capturedSquare = `${toSquare[0]}${capturedRank}`;
  }

  const promotedType = action.length === 5 ? action[4] : moved.type;
  const pieces = current.pieces
    .filter((piece) => piece.square !== fromSquare && piece.square !== toSquare && piece.square !== capturedSquare)
    .map((piece) => ({ ...piece }));

  if (moved.type === "k" && Math.abs(FILES.indexOf(fromSquare[0]) - FILES.indexOf(toSquare[0])) === 2) {
    const rank = fromSquare[1];
    const kingSide = toSquare[0] === "g";
    const rookFrom = `${kingSide ? "h" : "a"}${rank}`;
    const rookTo = `${kingSide ? "f" : "d"}${rank}`;
    const rook = pieces.find((piece) => piece.square === rookFrom);
    if (rook) rook.square = rookTo;
  }

  pieces.push({
    ...moved,
    square: toSquare,
    type: promotedType,
    symbol: moved.color === "white" ? promotedType.toUpperCase() : promotedType,
  });

  const nextActor = 1 - current.actor;
  const botTurn = current.mode === "human-bot" && !current.human_seats.includes(nextActor);
  const side = nextActor === 0 ? "白方" : "黑方";
  return {
    ...current,
    actor: nextActor,
    turn: nextActor === 0 ? "white" : "black",
    legal_actions: [],
    pieces,
    plies: current.plies + 1,
    last_move: action,
    human_turn: current.mode !== "human-bot" || current.human_seats.includes(nextActor),
    bot_turn: botTurn,
    status_text: botTurn ? `${side} Bot 思考中` : `${side}行动`,
  };
}

function animateLastMove(previousData, nextData) {
  const duration = getAnimationDuration();
  if (!previousData || !nextData || duration <= 0) return;
  if (!nextData.last_move || previousData.plies === nextData.plies) return;
  clearMoveAnimations();

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
  clone.removeAttribute("data-square");
  clone.style.left = `${startLeft}px`;
  clone.style.top = `${startTop}px`;
  clone.style.width = `${toRect.width}px`;
  clone.style.height = `${toRect.height}px`;
  clone.style.filter = style.filter;
  document.body.appendChild(clone);
  toPiece.classList.add("animating");

  runPieceAnimation(clone, toPiece, toRect.left - startLeft, toRect.top - startTop, duration);
}

function runPieceAnimation(clone, hiddenPiece, deltaX, deltaY, duration) {
  const animation = clone.animate(
    [
      { transform: "translate(0, 0)" },
      { transform: `translate(${deltaX}px, ${deltaY}px)` },
    ],
    { duration, easing: "cubic-bezier(0.22, 1, 0.36, 1)" }
  );

  return new Promise((resolve) => {
    const cleanup = () => {
      hiddenPiece.classList.remove("animating");
      clone.remove();
      activeMoveAnimations = activeMoveAnimations.filter((item) => item.animation !== animation);
      resolve();
    };
    animation.onfinish = cleanup;
    animation.oncancel = cleanup;
    activeMoveAnimations.push({ animation, clone, target: hiddenPiece });
  });
}

function clearMoveAnimations() {
  activeMoveAnimations.forEach(({ animation, clone, target }) => {
    animation.cancel();
    clone.remove();
    target.classList.remove("animating");
  });
  activeMoveAnimations = [];
}

function clearSelectionHighlights() {
  boardEl.querySelectorAll(".selected, .target").forEach((element) => {
    element.classList.remove("selected", "target");
  });
}

humanHumanBtn.addEventListener("click", () => setMode("human-human"));
humanBotBtn.addEventListener("click", () => setMode("human-bot"));
whiteSeatBtn.addEventListener("click", () => setHumanSeat(0));
blackSeatBtn.addEventListener("click", () => setHumanSeat(1));
botSelect.addEventListener("change", () => setBot(botSelect.value));
pieceStyleSelect.addEventListener("change", () => setPieceSet(pieceStyleSelect.value));
orientationSelect.addEventListener("change", () => setOrientation(orientationSelect.value));
newGameBtn.addEventListener("click", newGame);
flipBtn.addEventListener("click", () => {
  setOrientation(orientation === "white" ? "black" : "white");
});
promotionPanel.addEventListener("click", (event) => {
  if (event.target === promotionPanel) promotionPanel.hidden = true;
});

const initialPieceLoad = preloadPieceSet(pieceSet);
renderPieceMarks();
renderModeControls();
loadBotOptions().finally(async () => {
  await initialPieceLoad;
  newGame();
  preloadOtherPieceSets();
});

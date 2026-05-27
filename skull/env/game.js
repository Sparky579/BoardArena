let sessionId = null;
let cheatMode = false;
let lastData = null;

const el = id => document.getElementById(id);
const actionNames = {
  PLAY_F: "放花",
  PLAY_S: "放骷髅",
  PASS: "放弃"
};

function label(action) {
  if (actionNames[action]) return actionNames[action];
  if (action.startsWith("BID_")) return `叫 ${action.slice(4)}`;
  return action;
}

function cardNode(kind, hidden=false) {
  const div = document.createElement("div");
  div.className = hidden ? "card back" : `card ${kind === "S" ? "skull" : "flower"}`;
  if (!hidden) {
    const symbol = document.createElement("span");
    symbol.className = "card-symbol";
    symbol.textContent = kind === "S" ? "☠" : "✿";
    const name = document.createElement("span");
    name.className = "card-name";
    name.textContent = kind === "S" ? "骷髅" : "花";
    div.append(symbol, name);
  }
  return div;
}

async function api(path, body=null) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? {"Content-Type": "application/json"} : {},
    body: body ? JSON.stringify(body) : null
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function newGame(human) {
  const data = await api("/api/new", {human, policy_mode: el("policyMode").value});
  sessionId = data.session;
  render(data);
}

async function act(action) {
  if (!sessionId) return;
  const data = await api("/api/action", {session: sessionId, action});
  render(data);
}

function render(data) {
  lastData = data;
  sessionId = data.session;
  document.body.classList.toggle("cheat", cheatMode);
  el("cheatToggle").textContent = cheatMode ? "作弊模式：开" : "作弊模式：关";
  el("cheatToggle").classList.toggle("active", cheatMode);
  el("policyMode").value = data.policy_mode || el("policyMode").value;
  el("cpuScore").textContent = data.cpu.score;
  el("humanScore").textContent = data.human.score;
  el("cpuCards").textContent = `总牌 ${data.cpu.total_cards}`;
  el("cpuPileCount").textContent = `牌堆 ${data.cpu.pile_count}`;
  el("cpuPileMode").textContent = cheatMode ? "已透视" : "隐藏";
  el("cpuPolicy").textContent = `策略 ${data.policy_mode}`;
  el("humanHand").textContent = `花 ${data.human.hand.flowers} / 骷髅 ${data.human.hand.skulls}`;
  el("humanCards").textContent = `总牌 ${data.human.total_cards}`;
  el("status").textContent = data.status;

  const cpuPile = el("cpuPile");
  cpuPile.innerHTML = "";
  if (cheatMode) {
    for (const c of data.cpu.pile) cpuPile.appendChild(cardNode(c));
  } else {
    for (let i = 0; i < data.cpu.pile_count; i++) cpuPile.appendChild(cardNode("?", true));
  }

  const humanPile = el("humanPile");
  humanPile.innerHTML = "";
  for (const c of data.human.pile) humanPile.appendChild(cardNode(c));

  const actions = el("actions");
  actions.innerHTML = "";
  for (const action of data.legal_actions) {
    const b = document.createElement("button");
    b.textContent = label(action);
    b.className = action === "PLAY_S" ? "danger" : action === "PLAY_F" ? "good" : action.startsWith("BID_") ? "blue" : "";
    b.onclick = () => act(action);
    actions.appendChild(b);
  }
  if (!data.legal_actions.length) {
    const b = document.createElement("button");
    b.textContent = data.winner === null ? "等待 CPU" : "重新开始";
    b.className = "primary";
    b.onclick = () => newGame(data.human.id);
    actions.appendChild(b);
  }
  el("hint").textContent = data.hint;

  const log = el("log");
  log.innerHTML = "";
  for (const item of data.log.slice().reverse()) {
    const div = document.createElement("div");
    div.className = "entry" + (item.important ? " important" : "");
    div.textContent = item.text;
    log.appendChild(div);
  }
}

el("newP0").onclick = () => newGame(0);
el("newP1").onclick = () => newGame(1);
el("cheatToggle").onclick = () => {
  cheatMode = !cheatMode;
  if (lastData) render(lastData);
};
newGame(0).catch(err => {
  el("status").textContent = err.message;
});

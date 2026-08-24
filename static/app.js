/* Nemotron Switchyard Studio */

const $ = (id) => document.getElementById(id);

const PATHS = {
  in: () => $("pathIn"),
  frontier: () => $("pathFrontier"),
  lightning: () => $("pathLightning"),
  local: () => $("pathLocal"),
};

let META = {};
let TRAIN_POLL = null;
let BUSY = false;
const MIX = { frontier: 0, lightning: 0, local: 0 };

function setBadge(text, live) {
  const el = $("liveBadge");
  el.textContent = text;
  el.className = "badge " + (live ? "live" : "offline");
}

/* ── tabs ─────────────────────────────────────────────────────────── */

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => {
      b.classList.toggle("on", b === btn);
      b.setAttribute("aria-selected", b === btn ? "true" : "false");
    });
    document.querySelectorAll(".view").forEach((v) => {
      v.classList.toggle("on", v.id === btn.dataset.tab);
    });
  });
});

/* ── boot ─────────────────────────────────────────────────────────── */

async function boot() {
  const meta = await (await fetch("/api/meta")).json();
  META = meta;
  const syLive = !!(meta.runtime && meta.runtime.official_switchyard && meta.runtime.official_switchyard.running);
  setBadge(
    syLive ? "Switchyard native :4000 · NIM" : (meta.nim_ready ? "NIM live · " + meta.device : "offline · " + meta.device),
    syLive || meta.nim_ready
  );
  $("liveToggle").checked = !!meta.nim_ready;
  const rt = meta.runtime || {};
  $("whereHere").textContent = `${rt.listen || "127.0.0.1:7878"} · ${rt.process || "app.py"} · ${rt.router_file || "src/router.py"}`;
  const off = (rt.official_switchyard || {});
  $("whereOfficial").textContent = off.running
    ? `LIVE native Rust proxy ${off.url} · python3 run_switchyard.py`
    : "not running. In another terminal: python3 run_switchyard.py  (listens on :4000)";
  const nim = rt.nim || {};
  $("whereNim").textContent = nim.ready
    ? `${nim.model} @ NIM · inference only`
    : "no NVIDIA_API_KEY · classify still works offline";
  $("trainHint").textContent = `${meta.device} · ${meta.sft_model.split("/").pop()}`;
  $("studentSelect").innerHTML = meta.students
    .map((s) => `<option value="${s.id}">${s.label}</option>`)
    .join("");
  $("studentSelect").value = meta.sft_model;
  $("studentHint").textContent = (meta.students.find((s) => s.id === meta.sft_model) || {}).note || "";
  $("studentSelect").addEventListener("change", () => {
    const s = meta.students.find((x) => x.id === $("studentSelect").value);
    $("studentHint").textContent = s ? s.note : "";
  });

  const samples = await (await fetch("/api/samples")).json();
  const chips = $("chips");
  samples.forEach((s) => {
    const b = document.createElement("button");
    b.className = "chip";
    b.type = "button";
    b.textContent = s.title;
    b.addEventListener("click", () => {
      $("queryInput").value = s.text;
      sendQuery();
    });
    chips.appendChild(b);
  });

  const ds = await (await fetch("/api/dataset")).json();
  $("trainTable").innerHTML = ds.train
    .map(
      (r) => `<div class="row"><span class="id">${r.id}</span>
        <span class="fam">${r.family}</span>
        <div>${esc(r.user)}</div></div>`
    )
    .join("");
  $("testSelect").innerHTML = ds.test
    .map((r) => `<option value="${r.id}">${r.id} · ${r.family}</option>`)
    .join("");

  const st = await (await fetch("/api/train/status")).json();
  paintTrain(st);
}

function esc(s) {
  return String(s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/* ── switchyard animation ─────────────────────────────────────────── */

function animateAlong(path, packet, duration) {
  return new Promise((resolve) => {
    const len = path.getTotalLength();
    const t0 = performance.now();
    packet.setAttribute("visibility", "visible");
    function tick(now) {
      const p = Math.min(1, (now - t0) / duration);
      const ease = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;
      const pt = path.getPointAtLength(ease * len);
      packet.setAttribute("cx", pt.x);
      packet.setAttribute("cy", pt.y);
      if (p < 1) requestAnimationFrame(tick);
      else resolve();
    }
    requestAnimationFrame(tick);
  });
}

function resetRails() {
  ["pathFrontier", "pathLightning", "pathLocal"].forEach((id) => {
    $(id).classList.remove("hot-front", "hot-light", "hot-local");
  });
  $("tower").className.baseVal = "";
  document.querySelectorAll(".bay").forEach((b) => b.classList.remove("on"));
}

async function throwSwitch(track) {
  resetRails();
  $("tower").classList.add("throw-" + track);
  const path = PATHS[track]();
  path.classList.add(
    track === "frontier" ? "hot-front" : track === "local" ? "hot-local" : "hot-light"
  );
  document.querySelector(`.bay[data-track="${track}"]`)?.classList.add("on");
  const packet = $("packet");
  packet.style.fill =
    track === "frontier" ? "#ff8c42" : track === "local" ? "#5ec8ff" : "#76b900";
  await animateAlong(path, packet, 900);
}

function paintDecision(data) {
  const d = data.decision;
  $("dTrack").textContent = d.track;
  $("dTrack").style.color =
    d.track === "frontier" ? "var(--front)" : d.track === "local" ? "var(--local)" : "var(--accent)";
  $("dScore").textContent = d.score.toFixed(2);
  $("dSave").textContent = "$" + Number(d.saved_usd).toFixed(5);
  $("dMs").textContent = data.ms ? data.ms + " ms" : "—";
  const via = data.via ? `  · via ${data.via}` : "";
  $("dReason").textContent = (data.rationale || d.reason) + "  ·  " + d.model_id + via;
  const reply = $("dReply");
  if (data.reply) {
    reply.hidden = false;
    reply.textContent = (data.reasoning ? "⟨think⟩ " + data.reasoning.slice(0, 280) + "\n\n" : "") + data.reply;
  } else {
    reply.hidden = true;
  }

  const bayMap = {
    frontier: ["bayFrontierModel", "bayFrontierReason"],
    lightning: ["bayLightningModel", "bayLightningReason"],
    local: ["bayLocalModel", "bayLocalReason"],
  };
  const ids = bayMap[d.track];
  if (ids) {
    $(ids[0]).textContent = d.model_id;
    $(ids[1]).textContent = d.reason;
  }

  MIX[d.track] = (MIX[d.track] || 0) + 1;
  $("mixLine").textContent = `frontier ${MIX.frontier} · lightning ${MIX.lightning} · local ${MIX.local}`;
  const li = document.createElement("li");
  li.innerHTML = `<span class="pill ${d.track}">${d.track}</span>
    <span>${esc((d.strategy ? d.strategy + " · " : "") + (data.query || "")).slice(0, 160)}</span>`;
  $("feed").prepend(li);
}

async function sendQuery() {
  const q = $("queryInput").value.trim();
  if (!q || BUSY) return;
  BUSY = true;
  $("sendBtn").disabled = true;
  resetRails();
  const packet = $("packet");
  packet.setAttribute("visibility", "visible");
  packet.style.fill = "#e8f6ec";
  await animateAlong(PATHS.in(), packet, 700);

  const live = $("liveToggle").checked;
  let data;
  try {
    const r = await fetch("/api/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: q, live, strategy: $("strategy").value }),
    });
    data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
  } catch (e) {
    $("dReason").textContent = "Route failed: " + e.message;
    BUSY = false;
    $("sendBtn").disabled = false;
    return;
  }
  await throwSwitch(data.decision.track);
  paintDecision(data);
  BUSY = false;
  $("sendBtn").disabled = false;
}

$("sendBtn").addEventListener("click", sendQuery);
$("queryInput").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") sendQuery();
});

function spawnPacket() {
  const svg = $("yardSvg");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  p.setAttribute("r", "7");
  p.setAttribute("class", "packet");
  p.setAttribute("cx", "40");
  p.setAttribute("cy", "230");
  svg.appendChild(p);
  return p;
}

async function runSession() {
  if (BUSY) return;
  BUSY = true;
  $("sessionBtn").disabled = true;
  $("sendBtn").disabled = true;
  document.querySelectorAll(".packet").forEach((el) => {
    if (el.id !== "packet") el.remove();
  });
  $("packet").setAttribute("visibility", "hidden");
  try {
    const r = await fetch("/api/session/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy: $("strategy").value === "content" ? "stage" : $("strategy").value }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("dReason").textContent =
      `${data.session.title} · ${data.strategy} · saved $${Number(data.saved_usd).toFixed(5)} vs frontier-only`;
    $("dReply").hidden = true;
    for (const row of data.steps) {
      const track = row.decision.track;
      const packet = spawnPacket();
      packet.style.fill = "#e8f6ec";
      await animateAlong(PATHS.in(), packet, 380);
      packet.style.fill =
        track === "frontier" ? "#ff8c42" : track === "local" ? "#5ec8ff" : "#76b900";
      $("tower").className.baseVal = "throw-" + track;
      const path = PATHS[track]();
      ["pathFrontier", "pathLightning", "pathLocal"].forEach((id) => {
        $(id).classList.remove("hot-front", "hot-light", "hot-local");
      });
      path.classList.add(
        track === "frontier" ? "hot-front" : track === "local" ? "hot-local" : "hot-light"
      );
      document.querySelectorAll(".bay").forEach((b) => b.classList.remove("on"));
      document.querySelector(`.bay[data-track="${track}"]`)?.classList.add("on");
      await animateAlong(path, packet, 520);
      paintDecision({ query: row.step.content, decision: row.decision, ms: 0, reply: null });
    }
    $("dSave").textContent = "$" + Number(data.saved_usd).toFixed(5);
    $("dTrack").textContent = "session";
    $("dScore").textContent = `${data.mix.frontier}/${data.mix.lightning}/${data.mix.local}`;
  } catch (e) {
    $("dReason").textContent = "Session failed: " + e.message;
  }
  BUSY = false;
  $("sessionBtn").disabled = false;
  $("sendBtn").disabled = false;
}

$("sessionBtn").addEventListener("click", runSession);

/* ── fine-tune ────────────────────────────────────────────────────── */

function paintLoss(losses) {
  const c = $("lossCanvas");
  const ctx = c.getContext("2d");
  const w = c.width;
  const h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!losses.length) return;
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = Math.max(max - min, 1e-6);
  ctx.strokeStyle = "#76b900";
  ctx.lineWidth = 2;
  ctx.beginPath();
  losses.forEach((y, i) => {
    const x = (i / Math.max(losses.length - 1, 1)) * (w - 16) + 8;
    const py = h - 12 - ((y - min) / span) * (h - 24);
    if (i === 0) ctx.moveTo(x, py);
    else ctx.lineTo(x, py);
  });
  ctx.stroke();
}

function paintTrain(st) {
  const pct = st.max_steps ? Math.round((100 * st.step) / st.max_steps) : 0;
  $("trainBar").style.width = (st.status === "done" ? 100 : pct) + "%";
  $("trainLog").textContent = (st.logs || []).join("\n") || st.message || "Ready.";
  $("trainLog").scrollTop = $("trainLog").scrollHeight;
  paintLoss(st.losses || []);
  $("trainBtn").disabled = st.status === "running";
  $("trainBtn").textContent = st.status === "running" ? `Training ${st.step}/${st.max_steps}` : "Start LoRA SFT";
}

$("trainBtn").addEventListener("click", async () => {
  $("trainBtn").disabled = true;
  const body = {
    model_id: $("studentSelect").value,
    max_steps: Number($("maxSteps").value) || 24,
    lora_r: Number($("loraR").value) || 8,
    lr: Number($("lr").value) || 2e-4,
  };
  const r = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    $("trainLog").textContent = err.detail || r.statusText;
    $("trainBtn").disabled = false;
    return;
  }
  if (TRAIN_POLL) clearInterval(TRAIN_POLL);
  TRAIN_POLL = setInterval(async () => {
    const st = await (await fetch("/api/train/status")).json();
    paintTrain(st);
    if (st.status === "done" || st.status === "error") {
      clearInterval(TRAIN_POLL);
      TRAIN_POLL = null;
      if (st.status === "done") setBadge("adapter ready · " + META.device, true);
    }
  }, 600);
});

/* ── compare ──────────────────────────────────────────────────────── */

$("compareBtn").addEventListener("click", async () => {
  $("compareBtn").disabled = true;
  $("compareHint").textContent = "Running stock + adapter + NIM… this loads the student once.";
  $("compareCols").innerHTML = "";
  try {
    const r = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        example_id: $("testSelect").value,
        include_nim: true,
        include_stock: true,
        include_adapter: true,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    $("compareCols").innerHTML = (data.columns || [])
      .map((col) => {
        const sc = col.scores;
        const badge = sc
          ? `<span class="${sc.task_pass ? "pass" : "fail"}">${sc.task_pass ? "TASK PASS" : "TASK FAIL"}</span>
             · tags ${sc.has_ticket_tags ? "yes" : "no"}`
          : "";
        const body = col.error ? "ERROR: " + col.error : col.text || "—";
        return `<article class="col">
          <h3>${esc(col.label)}</h3>
          <div class="meta">${badge}${col.ms ? " · " + col.ms + " ms" : ""}</div>
          <pre>${esc(body)}</pre>
        </article>`;
      })
      .join("");
    $("compareHint").textContent = "Gold: " + JSON.stringify(data.gold);
  } catch (e) {
    $("compareHint").textContent = e.message;
  }
  $("compareBtn").disabled = false;
});

boot().catch((e) => setBadge("boot failed: " + e.message, false));

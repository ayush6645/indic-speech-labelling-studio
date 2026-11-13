(() => {
  "use strict";

  const LS = {
    theme: "lt_theme",
    speed: "lt_speed",
    boost: "lt_boost",
    loop: "lt_loop",
    opus: "lt_opus",
  };

  const state = {
    user: null,
    isLocal: false,
    langMeta: {},
    filters: {
      state: "all",
      lang: "all",
      source: "all",
      status: "ready",
      q: "",
      sort: "resume",
      offset: 0,
      limit: 120,
    },
    calls: [],
    callsTotal: 0,
    currentCallId: null,
    call: null,
    pieces: [],
    callText: null,
    overlaps: [],
    waveform: null,
    openPieceId: null,
    playMode: "piece", // "piece" | "call"
    playing: false,
    speed: clampNum(parseFloat(localStorage.getItem(LS.speed) || "1.0"), 0.5, 2.0),
    boost: localStorage.getItem(LS.boost) === "1",
    loop: localStorage.getItem(LS.loop) === "1",
    useOpus: null, // decided after /api/me + localStorage override
    autosaveTimer: null,
    dirtyPieceId: null,
    lastEditedBox: null, // "text" | "text_roman" | null
    edgeOpen: false,
    histOpen: false,
    histData: null,
    guideOpen: false,
    guideEditing: false,
    guideMarkdown: "",
    refOpen: false,
    confirmCallback: null,
    splitPollTimer: null,
  };

  // DOM helpers
  const $ = (id) => document.getElementById(id);
  const audioEl = $("audio-el");
  const canvasEl = $("waveform-canvas");
  const ctx2d = canvasEl.getContext("2d");

  // Web Audio boost pipeline (lazy-created on user gesture)
  let audioCtx = null;
  let mediaSrcNode = null;
  let compNode = null;
  let gainNode = null;

  function clampNum(v, lo, hi) {
    if (Number.isNaN(v)) return lo;
    return Math.max(lo, Math.min(hi, v));
  }

  function esc(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtTime(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const r = s % 60;
    if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
    return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
  }

  function fmtTimeDec(sec) {
    const s = Math.max(0, Number(sec || 0));
    const m = Math.floor(s / 60);
    const r = (s % 60).toFixed(1).padStart(4, "0");
    return `${String(m).padStart(2, "0")}:${r}`;
  }

  function isTypingInInput(el) {
    if (!el) return false;
    if (el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable) return true;
    if (el.tagName === "INPUT") {
      return !["range", "checkbox", "radio", "button", "submit"].includes((el.type || "text").toLowerCase());
    }
    return false;
  }

  async function api(url, opts = {}) {
    const headers = new Headers(opts.headers || {});
    if (opts.json !== undefined) {
      headers.set("Content-Type", "application/json");
    }
    const resp = await fetch(url, {
      credentials: "same-origin",
      ...opts,
      headers,
      body: opts.json !== undefined ? JSON.stringify(opts.json) : opts.body,
    });
    if (resp.status === 401 && !url.startsWith("/api/login") && !url.startsWith("/api/me")) {
      showLogin();
      throw new Error("Authentication required");
    }
    const ctype = resp.headers.get("content-type") || "";
    const data = ctype.includes("application/json") ? await resp.json() : await resp.text();
    if (!resp.ok) {
      const msg = (data && data.detail) ? data.detail : (typeof data === "string" ? data : `HTTP ${resp.status}`);
      throw new Error(msg);
    }
    return data;
  }

  // -- Theme & Language chips ----------------------------------------------------------------------------------------

  function applyTheme(theme) {
    const t = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem(LS.theme, t);
    $("btn-theme").textContent = t === "dark" ? "☀" : "◐";
    drawWaveform();
  }

  function langColor(code) {
    const c = (code || "").toLowerCase();
    const meta = state.langMeta[c];
    return meta ? meta.color : "#6B6F76";
  }

  function langLabel(code) {
    const c = (code || "").toLowerCase();
    if (!c) return "—";
    return c;
  }

  function renderLangTag(code, extraTitle = "") {
    if (!code) return `<span class="lang-tag" style="--tag-color:#6B6F76"><span class="lang-dot"></span>?</span>`;
    const color = langColor(code);
    const full = (state.langMeta[code]?.label || code) + (extraTitle ? ` (${extraTitle})` : "");
    return `<span class="lang-tag" style="--tag-color:${esc(color)}" title="${esc(full)}"><span class="lang-dot"></span>${esc(langLabel(code))}</span>`;
  }

  // -- Audio engine (Piece / Call playback, speed, boost, loop, Opus) -----------------------------------------------

  function initBoostGraph() {
    if (audioCtx || !window.AudioContext) return;
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC();
      mediaSrcNode = audioCtx.createMediaElementSource(audioEl);
      compNode = audioCtx.createDynamicsCompressor();
      compNode.threshold.value = -20;
      compNode.knee.value = 18;
      compNode.ratio.value = 4;
      compNode.attack.value = 0.005;
      compNode.release.value = 0.15;
      gainNode = audioCtx.createGain();
      gainNode.gain.value = state.boost ? 2.6 : 1.0;
      mediaSrcNode.connect(compNode);
      compNode.connect(gainNode);
      gainNode.connect(audioCtx.destination);
    } catch (_) {
      audioCtx = null;
    }
  }

  function applyBoostUi() {
    $("btn-boost").classList.toggle("btn-active", state.boost);
    localStorage.setItem(LS.boost, state.boost ? "1" : "0");
    if (state.boost && !audioCtx) {
      // lazily hook when playing or clicked
    }
    if (gainNode) {
      gainNode.gain.value = state.boost ? 2.6 : 1.0;
    }
  }

  function setSpeed(val) {
    const rounded = Math.round(clampNum(Number(val), 0.5, 2.0) * 100) / 100;
    state.speed = rounded;
    applySpeedUi();
  }

  function applySpeedUi() {
    state.speed = clampNum(state.speed, 0.5, 2.0);
    localStorage.setItem(LS.speed, String(state.speed));
    $("speed-slider").value = String(state.speed);
    $("speed-val").textContent = `${state.speed.toFixed(2)}×`;
    const btnDown = $("btn-speed-down");
    const btnUp = $("btn-speed-up");
    if (btnDown) btnDown.disabled = state.speed <= 0.501;
    if (btnUp) btnUp.disabled = state.speed >= 1.999;
    document.querySelectorAll(".speed-preset-btn").forEach((btn) => {
      const target = parseFloat(btn.dataset.speed);
      const isMatch = Math.abs(target - state.speed) < 0.005;
      btn.classList.toggle("active", isMatch);
      btn.setAttribute("aria-pressed", isMatch ? "true" : "false");
    });
    audioEl.preservesPitch = true;
    audioEl.mozPreservesPitch = true;
    audioEl.webkitPreservesPitch = true;
    audioEl.playbackRate = state.speed;
  }

  function applyLoopUi() {
    $("btn-loop").classList.toggle("btn-active", state.loop);
    localStorage.setItem(LS.loop, state.loop ? "1" : "0");
  }

  function applyOpusUi() {
    $("btn-opus").classList.toggle("btn-active", !!state.useOpus);
  }

  function applyPlayModeUi() {
    $("btn-mode-piece").classList.toggle("active", state.playMode === "piece");
    $("btn-mode-call").classList.toggle("active", state.playMode === "call");
  }

  function currentPiece() {
    return state.pieces.find((p) => p.piece_id === state.openPieceId) || null;
  }

  function buildAudioSrc(mode, id, audioV = "0") {
    const fmt = state.useOpus ? "opus" : "raw";
    if (mode === "piece") return `/api/audio/piece/${encodeURIComponent(id)}?fmt=${fmt}&v=${encodeURIComponent(audioV)}`;
    return `/api/audio/call/${encodeURIComponent(id)}?fmt=${fmt}&v=${encodeURIComponent(audioV)}`;
  }

  function loadAudioSource({ mode = state.playMode, autoplay = false, seekSec = 0 } = {}) {
    state.playMode = mode;
    applyPlayModeUi();
    const p = currentPiece();
    if (mode === "piece" && !p) return;
    if (!state.call) return;

    const targetId = mode === "piece" ? p.piece_id : state.call.call_id;
    const v = mode === "piece" ? (p.audio_v || "0") : "0";
    const url = buildAudioSrc(mode, targetId, v);
    const sameSrc = audioEl.dataset.srcKey === url;

    const doSeekAndPlay = () => {
      applySpeedUi();
      if (seekSec >= 0) {
        try {
          audioEl.currentTime = seekSec;
        } catch (_) {}
      }
      updateTimeDisplay();
      drawWaveform();
      if (autoplay) safePlay();
    };

    if (sameSrc) {
      doSeekAndPlay();
      return;
    }

    audioEl.pause();
    audioEl.dataset.srcKey = url;
    audioEl.src = url;
    audioEl.load();
    audioEl.addEventListener("loadedmetadata", function onMeta() {
      audioEl.removeEventListener("loadedmetadata", onMeta);
      doSeekAndPlay();
    });
    if (autoplay) {
      // Trigger play right inside the click/gesture tick if metadata is already buffered or streams cleanly
      applySpeedUi();
      safePlay();
    }
  }

  function safePlay() {
    if (state.boost && !audioCtx) initBoostGraph();
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
    applySpeedUi();
    const pr = audioEl.play();
    if (pr && typeof pr.catch === "function") {
      pr.catch(() => {
        state.playing = false;
        updatePlayButton();
      });
    }
  }

  function togglePlay() {
    if (!audioEl.src) {
      loadAudioSource({ mode: state.pieces.length ? state.playMode : "call", autoplay: true, seekSec: 0 });
      return;
    }
    if (audioEl.paused) {
      safePlay();
    } else {
      audioEl.pause();
    }
  }

  function seekDelta(delta) {
    if (!audioEl.src) return;
    const dur = Number.isFinite(audioEl.duration) ? audioEl.duration : 99999;
    audioEl.currentTime = clampNum((audioEl.currentTime || 0) + delta, 0, dur);
    updateTimeDisplay();
    drawWaveform();
  }

  function getCallPlayheadSec() {
    if (state.playMode === "piece") {
      const p = currentPiece();
      if (!p) return 0;
      return p.start_s + (audioEl.currentTime || 0);
    }
    return audioEl.currentTime || 0;
  }

  function updatePlayButton() {
    const isPlaying = !audioEl.paused && !audioEl.ended;
    state.playing = isPlaying;
    $("btn-play").textContent = isPlaying ? "⏸" : "▶";
  }

  function updateTimeDisplay() {
    const cur = audioEl.currentTime || 0;
    const p = currentPiece();
    const total = Number.isFinite(audioEl.duration) && audioEl.duration > 0
      ? audioEl.duration
      : (state.playMode === "piece" && p ? p.duration_s : (state.call ? state.call.duration_s : 0));
    $("player-time").textContent = `${fmtTime(cur)} / ${fmtTime(total)}`;
    // Update split-at-playhead label live if open
    const spBtn = document.getElementById("btn-edge-split");
    if (spBtn && p) {
      const ch = getCallPlayheadSec();
      const valid = ch > p.start_s + 0.45 && ch < p.end_s - 0.45;
      const target = valid ? ch : (p.start_s + p.end_s) / 2;
      spBtn.dataset.splitTime = target.toFixed(3);
      spBtn.textContent = `Split at ${fmtTimeDec(target)}${valid ? " (playhead)" : " (mid)"}`;
    }
  }

  let rafId = null;
  function tickPlayhead() {
    updateTimeDisplay();
    drawWaveform();
    if (!audioEl.paused && !audioEl.ended) {
      rafId = requestAnimationFrame(tickPlayhead);
    } else {
      rafId = null;
    }
  }

  audioEl.addEventListener("play", () => {
    applySpeedUi();
    updatePlayButton();
    if (!rafId) rafId = requestAnimationFrame(tickPlayhead);
  });
  audioEl.addEventListener("pause", () => {
    updatePlayButton();
    updateTimeDisplay();
    drawWaveform();
  });
  audioEl.addEventListener("ended", () => {
    if (state.playMode === "piece" && state.loop) {
      audioEl.currentTime = 0;
      safePlay();
      return;
    }
    updatePlayButton();
    updateTimeDisplay();
    drawWaveform();
  });

  // -- Waveform Canvas -----------------------------------------------------------------------------------------------

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function drawWaveform() {
    if (!canvasEl || !ctx2d) return;
    const rect = canvasEl.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width * dpr);
    const h = Math.round(rect.height * dpr);
    if (canvasEl.width !== w || canvasEl.height !== h) {
      canvasEl.width = w;
      canvasEl.height = h;
    }
    ctx2d.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cw = rect.width;
    const ch = rect.height;
    ctx2d.clearRect(0, 0, cw, ch);

    const callDur = Math.max(0.01, state.waveform?.duration_s || state.call?.duration_s || 1);
    const colVerified = cssVar("--verified") || "#1F9D55";
    const colAccent = cssVar("--accent") || "#2F6FEB";
    const colWave = cssVar("--wave") || "#B8C4D6";
    const colPlayed = cssVar("--wave-played") || "#2F6FEB";
    const colBorder = cssVar("--border") || "#E4E4E0";

    // 1. Background highlights for verified pieces & open piece
    for (const p of state.pieces) {
      const x0 = (p.start_s / callDur) * cw;
      const x1 = (p.end_s / callDur) * cw;
      if (p.status === "verified") {
        ctx2d.save();
        ctx2d.globalAlpha = 0.16;
        ctx2d.fillStyle = colVerified;
        ctx2d.fillRect(x0, 0, Math.max(1, x1 - x0), ch);
        ctx2d.restore();
      }
      if (p.piece_id === state.openPieceId) {
        ctx2d.save();
        ctx2d.globalAlpha = 0.14;
        ctx2d.fillStyle = colAccent;
        ctx2d.fillRect(x0, 0, Math.max(1, x1 - x0), ch);
        ctx2d.restore();
        // Bottom accent bar for current piece
        ctx2d.fillStyle = colAccent;
        ctx2d.fillRect(x0, ch - 3, Math.max(2, x1 - x0), 3);
      }
    }

    // 2. Amplitude bars
    const peaks = state.waveform?.peaks || [];
    const playSec = getCallPlayheadSec();
    const playRatio = clampNum(playSec / callDur, 0, 1);
    if (peaks.length) {
      const barCount = Math.min(peaks.length, Math.max(40, Math.floor(cw / 2)));
      const step = peaks.length / barCount;
      const bw = cw / barCount;
      const midY = ch / 2;
      for (let i = 0; i < barCount; i++) {
        const s0 = Math.floor(i * step);
        const s1 = Math.max(s0 + 1, Math.floor((i + 1) * step));
        let amp = 0;
        for (let k = s0; k < s1; k++) {
          if (peaks[k] > amp) amp = peaks[k];
        }
        const bh = Math.max(2, amp * (ch - 14));
        const x = i * bw;
        const ratio = (i + 0.5) / barCount;
        ctx2d.fillStyle = ratio <= playRatio ? colPlayed : colWave;
        ctx2d.fillRect(x, midY - bh / 2, Math.max(1, bw - 0.6), bh);
      }
    }

    // 3. Piece boundaries (thin vertical dividers)
    ctx2d.strokeStyle = colBorder;
    ctx2d.lineWidth = 1;
    for (let i = 1; i < state.pieces.length; i++) {
      const x = Math.round((state.pieces[i].start_s / callDur) * cw) + 0.5;
      ctx2d.beginPath();
      ctx2d.moveTo(x, 0);
      ctx2d.lineTo(x, ch);
      ctx2d.stroke();
    }

    // 4. Playhead line
    const px = clampNum(playRatio * cw, 0, cw - 1);
    ctx2d.strokeStyle = colAccent;
    ctx2d.lineWidth = 1.5;
    ctx2d.beginPath();
    ctx2d.moveTo(px, 0);
    ctx2d.lineTo(px, ch);
    ctx2d.stroke();
  }

  let scrubbing = false;
  function seekFromWaveformPointer(clientX, autoplay = true) {
    if (!state.call) return;
    const rect = canvasEl.getBoundingClientRect();
    const ratio = clampNum((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const callDur = Math.max(0.01, state.waveform?.duration_s || state.call?.duration_s || 1);
    const targetSec = ratio * callDur;
    const p = currentPiece();
    if (state.playMode === "piece" && p && targetSec >= p.start_s && targetSec <= p.end_s) {
      loadAudioSource({ mode: "piece", autoplay, seekSec: targetSec - p.start_s });
    } else {
      loadAudioSource({ mode: "call", autoplay, seekSec: targetSec });
    }
  }

  canvasEl.addEventListener("pointerdown", (e) => {
    scrubbing = true;
    canvasEl.setPointerCapture(e.pointerId);
    seekFromWaveformPointer(e.clientX, true);
  });
  canvasEl.addEventListener("pointermove", (e) => {
    const rect = canvasEl.getBoundingClientRect();
    const ratio = clampNum((e.clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    const callDur = Math.max(0.01, state.waveform?.duration_s || state.call?.duration_s || 1);
    const hoverEl = $("waveform-hover");
    hoverEl.textContent = fmtTime(ratio * callDur);
    hoverEl.style.left = `${clampNum(ratio * 100, 3, 97)}%`;
    hoverEl.classList.remove("hidden");
    if (scrubbing) seekFromWaveformPointer(e.clientX, false);
  });
  canvasEl.addEventListener("pointerup", () => { scrubbing = false; });
  canvasEl.addEventListener("pointerleave", () => { $("waveform-hover").classList.add("hidden"); });

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => drawWaveform()).observe(canvasEl);
  }

  // -- Loading filters & calls ---------------------------------------------------------------------------------------

  async function loadFilters() {
    const data = await api("/api/filters");
    state.langMeta = data.lang_meta || {};

    const selState = $("filter-state");
    const prevState = selState.value;
    selState.innerHTML = `<option value="all">State: All</option>` +
      data.states.map((s) => `<option value="${esc(s)}">${esc(s)}</option>`).join("");
    if (data.states.includes(prevState)) selState.value = prevState;

    const selLang = $("filter-lang");
    const prevLang = selLang.value;
    selLang.innerHTML = `<option value="all">Language: All</option>` +
      data.languages.map((l) => `<option value="${esc(l)}">${esc(state.langMeta[l]?.label || l)} (${esc(l)})</option>`).join("") +
      `<option value="unknown">Unknown</option>`;
    if ([...selLang.options].some((o) => o.value === prevLang)) selLang.value = prevLang;
  }

  async function loadCalls({ append = false, autoOpenFirst = false } = {}) {
    if (!append) state.filters.offset = 0;
    const f = state.filters;
    const qs = new URLSearchParams({
      state: f.state,
      lang: f.lang,
      source: f.source,
      status: f.status,
      q: f.q,
      sort: f.sort,
      offset: String(f.offset),
      limit: String(f.limit),
    });
    const res = await api(`/api/calls?${qs.toString()}`);
    state.calls = append ? state.calls.concat(res.calls) : res.calls;
    state.callsTotal = res.total;

    // Top bar verified progress for active filters
    const vh = res.filtered_verified_hours || 0;
    const th = res.filtered_total_hours || 0;
    const pct = th > 0 ? Math.min(100, (vh / th) * 100) : 0;
    $("top-progress-fill").style.width = `${pct.toFixed(1)}%`;
    $("top-progress-text").textContent = `${vh.toFixed(1)} / ${th.toFixed(1)} h verified`;
    renderLangHoursBar(res.verified_by_lang || []);
    $("call-count-summary").textContent = `${state.calls.length.toLocaleString()} of ${state.callsTotal.toLocaleString()} calls`;

    renderCallList();

    if (!state.calls.length) {
      state.currentCallId = null;
      state.call = null;
      state.pieces = [];
      showEmptyState("No calls match these filters", true);
    } else if (autoOpenFirst || (!state.currentCallId && state.calls.length > 0)) {
      await selectCall(state.calls[0].call_id, { autoplay: false });
    }
  }

  function fmtHoursCompact(sec) {
    const s = Math.max(0, Number(sec || 0));
    if (s <= 0) return "0.00h";
    if (s < 36) return "<0.01h";
    return `${(s / 3600).toFixed(2)}h`;
  }

  function renderLangHoursBar(items) {
    const bar = $("lang-hours-bar");
    if (!bar) return;
    if (!items || !items.length) {
      bar.innerHTML = "";
      return;
    }
    bar.innerHTML = items.map((item) => {
      const sec = Number(item.verified_seconds || 0);
      const pieces = Number(item.verified_pieces || 0);
      const hasData = pieces > 0 || sec > 0;
      const hrsText = fmtHoursCompact(sec);
      const mins = Math.floor(sec / 60);
      const remS = Math.round(sec % 60);
      const durLabel = mins > 0 ? `${mins}m ${remS}s` : `${remS}s`;
      const tip = `${item.label || item.language}: ${hrsText} verified (${durLabel} · ${pieces.toLocaleString()} pieces) — click for full breakdown`;
      return `<button type="button" class="lang-hour-chip ${hasData ? "has-data" : ""}" style="--tag-color:${esc(item.color || langColor(item.language))}" data-lang-hour="${esc(item.language)}" title="${esc(tip)}"><span class="lang-dot"></span><span>${esc(item.language)}</span> <strong class="lang-hour-val tabular">${esc(hrsText)}</strong></button>`;
    }).join("");
    bar.querySelectorAll(".lang-hour-chip").forEach((btn) => {
      btn.addEventListener("click", openProgressModal);
    });
  }

  function renderCallList() {
    const listEl = $("call-list");
    if (!state.calls.length) {
      listEl.innerHTML = `<div class="empty-state"><p class="muted">No calls match these filters</p></div>`;
      $("call-list-footer").classList.add("hidden");
      return;
    }
    listEl.innerHTML = state.calls.map((c) => {
      const dotCls = c.pieces_count > 0 && c.verified_count === c.pieces_count
        ? "complete"
        : (c.verified_count > 0 ? "partial" : "");
      const active = c.call_id === state.currentCallId ? "active" : "";
      return `
        <button class="call-item ${active}" data-call-id="${esc(c.call_id)}" title="${esc(c.call_id)} · ${esc(c.state)}">
          <div class="call-item-row">
            <span class="call-id-wrap">
              <span class="call-dot ${dotCls}"></span>
              <span class="call-id-text">${esc(c.call_id)}</span>
            </span>
            ${renderLangTag(c.language)}
          </div>
          <div class="call-item-row meta-text tabular">
            <span>${fmtTime(c.duration_s)} · ${esc(c.state)}</span>
            <span>${c.verified_count}/${c.pieces_count}${c.pieces_count > 0 && c.verified_count === c.pieces_count ? " ✓" : ""}</span>
          </div>
        </button>`;
    }).join("");

    listEl.querySelectorAll(".call-item").forEach((btn) => {
      btn.addEventListener("click", () => selectCall(btn.dataset.callId, { autoplay: true }));
    });
    $("call-list-footer").classList.toggle("hidden", state.calls.length >= state.callsTotal);
  }

  function updateCallItemInList(updatedCall) {
    const idx = state.calls.findIndex((x) => x.call_id === updatedCall.call_id);
    if (idx >= 0) state.calls[idx] = { ...state.calls[idx], ...updatedCall };
    renderCallList();
  }

  function showEmptyState(msg, showClearBtn) {
    $("call-view").classList.add("hidden");
    $("empty-state").classList.remove("hidden");
    $("empty-state-msg").textContent = msg;
    $("btn-clear-filters").classList.toggle("hidden", !showClearBtn);
  }

  // -- Opening a call & pieces ---------------------------------------------------------------------------------------

  async function selectCall(callId, { autoplay = true, openPieceId = null } = {}) {
    await flushAutosave();
    closeMobileDrawer();
    state.currentCallId = callId;
    state.edgeOpen = false;
    state.histOpen = false;
    state.histData = null;
    state.refOpen = false;
    renderCallList();

    const [detail, wf] = await Promise.all([
      api(`/api/call/${encodeURIComponent(callId)}`),
      api(`/api/waveform/${encodeURIComponent(callId)}`).catch(() => null),
    ]);
    state.call = detail.call;
    state.pieces = detail.pieces || [];
    state.callText = detail.call_text || null;
    state.overlaps = detail.overlaps || [];
    state.waveform = wf;

    $("empty-state").classList.add("hidden");
    $("call-view").classList.remove("hidden");
    renderCallHeader();

    // Pick initial piece: explicit -> first unverified -> first piece -> none
    if (openPieceId && state.pieces.some((p) => p.piece_id === openPieceId)) {
      state.openPieceId = openPieceId;
    } else {
      const firstUnver = state.pieces.find((p) => p.status !== "verified");
      state.openPieceId = (firstUnver || state.pieces[0] || {}).piece_id || null;
    }

    renderPieces();
    renderReferenceCard();
    drawWaveform();
    const callAreaEl = $("call-area");
    if (callAreaEl) callAreaEl.scrollTop = 0;
    const stickyTop = $("call-sticky-top");
    if (stickyTop) stickyTop.classList.remove("is-stuck");
    scrollOpenPieceIntoView();

    if (state.openPieceId) {
      loadAudioSource({ mode: "piece", autoplay, seekSec: 0 });
    } else {
      loadAudioSource({ mode: "call", autoplay, seekSec: 0 });
    }
  }

  function renderCallHeader() {
    const c = state.call;
    if (!c) return;
    const extras = (c.also_languages || "").trim().split(/\s+/).filter(Boolean);
    const completeBadge = c.complete ? `<span class="pill-verified">Complete ✓</span>` : "";
    const splitBadge = c.assigned_split ? `<span class="pill-unverified">split: ${esc(c.assigned_split)}</span>` : "";
    const flagBadge = c.flags ? `<span class="pill-flag">${esc(c.flags)}</span>` : "";
    const ovBadge = state.overlaps.length
      ? `<span class="pill-flag" title="Audio overlap detected with ${esc(state.overlaps.map((o) => o.old_call_id + '↔' + o.crm_call_id).join(', '))}">Audio overlap (${state.overlaps.length})</span>`
      : "";

    $("call-header").innerHTML = `
      <div class="call-title-group">
        <span class="call-title tabular">${esc(c.call_id)}</span>
        <span class="muted">· ${esc(c.state)} · ${esc(c.source)}</span>
        ${renderLangTag(c.language)}
        ${extras.map((l) => renderLangTag(l)).join("")}
        <span class="muted tabular">· ${fmtTime(c.duration_s)} · ${c.pieces_count} pieces · ${c.verified_count} verified</span>
        ${completeBadge}
        ${splitBadge}
        ${ovBadge}
        ${flagBadge}
      </div>`;
  }

  function renderReferenceCard() {
    const card = $("reference-card");
    const ct = state.callText;
    if (!ct) {
      card.classList.add("hidden");
      return;
    }
    card.classList.remove("hidden");
    const blocks = [];
    if (ct.text_native) blocks.push(`<div class="ref-block"><div class="script-label">Whole call — Native</div><pre>${esc(ct.text_native)}</pre></div>`);
    if (ct.text_romanized) blocks.push(`<div class="ref-block"><div class="script-label">Whole call — Romanized</div><pre>${esc(ct.text_romanized)}</pre></div>`);
    if (ct.v4_diarized_text) blocks.push(`<div class="ref-block"><div class="script-label">Agent / Customer split (${esc(ct.v4_model || "V4")})</div><pre>${esc(ct.v4_diarized_text)}</pre></div>`);
    if (ct.v4_english_text) blocks.push(`<div class="ref-block"><div class="script-label">English translation</div><pre>${esc(ct.v4_english_text)}</pre></div>`);
    $("reference-body").innerHTML = blocks.join("") || `<p class="muted">No reference text available.</p>`;
    $("reference-body").classList.toggle("hidden", !state.refOpen);
    $("btn-toggle-ref").innerHTML = `<span class="caret">${state.refOpen ? "▾" : "▸"}</span> Reference (whole-call text)`;
  }

  // -- Render Pieces Accordion (ONLY ONE PIECE OPEN AT A TIME) -------------------------------------------------------

  function renderWordDiffHtml(ops, text) {
    const words = (text || "").trim().split(/\s+/).filter(Boolean);
    if (!ops || !ops.length) return esc(text || "(empty)");
    const parts = [];
    let cur = 0;
    for (const op of ops) {
      const [j1, j2] = op.new_range || [0, 0];
      while (cur < j1 && cur < words.length) {
        parts.push(esc(words[cur++]));
      }
      if (op.old) {
        parts.push(`<span class="diff-del">${esc(op.old)}</span>`);
      }
      for (let j = j1; j < j2 && j < words.length; j++) {
        parts.push(`<mark class="diff-chg">${esc(words[j])}</mark>`);
      }
      cur = Math.max(cur, j2);
    }
    while (cur < words.length) {
      parts.push(esc(words[cur++]));
    }
    return parts.join(" ") || "(empty)";
  }

  function renderDiffOpsList(ops) {
    if (!ops || !ops.length) return `<span class="muted">No word changes</span>`;
    return ops.map((op) => {
      const o = op.old ? `<span class="diff-del">${esc(op.old)}</span>` : "";
      const n = op.new ? `<mark class="diff-chg">${esc(op.new)}</mark>` : "";
      return `${o}${o && n ? " → " : ""}${n}`;
    }).join(" · ");
  }

  function autoSizeTextarea(ta) {
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.max(108, ta.scrollHeight + 4)}px`;
  }

  function renderPieces() {
    const cont = $("pieces-container");
    if (!state.pieces.length) {
      cont.innerHTML = `<div class="empty-state"><p class="muted">${esc(state.call?.flags || "This call has not been cut into pieces yet.")}</p></div>`;
      return;
    }

    cont.innerHTML = state.pieces.map((p, idx) => {
      const shortId = `s${String(p.piece_index).padStart(3, "0")}`;
      const timeRange = `${fmtTime(p.start_s)}–${fmtTime(p.end_s)}`;
      if (p.piece_id !== state.openPieceId) {
        const preview = (p.current_text_roman || p.current_text || "").trim().slice(0, 72);
        const badge = p.status === "verified"
          ? `<span class="pill-verified">✓ Verified</span>`
          : `<span class="pill-unverified">Unverified</span>`;
        const extraTag = p.silent ? `<span class="meta-text">[silent]</span>` : "";
        return `
          <button class="piece-row-compact" data-open-piece="${esc(p.piece_id)}">
            <span class="piece-compact-left">
              <strong class="tabular">${esc(shortId)}</strong>
              <span class="meta-text tabular">${esc(timeRange)} · ${p.duration_s.toFixed(1)} s</span>
              ${renderLangTag(p.main_language)}
              ${extraTag}
              <span class="piece-preview">${preview ? `“${esc(preview)}”` : "(empty)"}</span>
            </span>
            ${badge}
          </button>`;
      }

      const isVerified = p.status === "verified";
      const badge = isVerified
        ? `<span id="piece-status-badge" class="pill-verified">✓ Verified</span>`
        : `<span id="piece-status-badge" class="pill-unverified">Unverified</span>`;
      const prevVerified = idx > 0 && state.pieces[idx - 1].status === "verified";
      const nextVerified = idx + 1 < state.pieces.length && state.pieces[idx + 1].status === "verified";
      const canEdge = !isVerified;
      const canMoveStart = canEdge && !prevVerified;
      const canMoveEnd = canEdge && !nextVerified;
      const canMerge = canEdge && idx + 1 < state.pieces.length && !nextVerified;

      const ch = getCallPlayheadSec();
      const splitValid = ch > p.start_s + 0.45 && ch < p.end_s - 0.45;
      const splitT = splitValid ? ch : (p.start_s + p.end_s) / 2;

      const needsNat = p.needs_update_script === "text";
      const needsRom = p.needs_update_script === "text_roman";
      const langCodes = Object.keys(state.langMeta).filter((k) => k !== "none");
      const allMainCodes = [...langCodes, "none"];

      return `
        <article class="piece-card-open" data-piece-id="${esc(p.piece_id)}">
          <div class="piece-open-head">
            <div class="piece-open-meta">
              <strong class="tabular">${esc(shortId)}</strong>
              <span class="meta-text tabular">${esc(timeRange)} · ${p.duration_s.toFixed(1)} s</span>
              ${renderLangTag(p.main_language)}
              ${(p.also_languages || []).map((l) => renderLangTag(l, "also")).join("")}
              ${p.edges_edited ? `<span class="meta-text">[edges edited]</span>` : ""}
              ${p.flags ? `<span class="pill-flag">${esc(p.flags)}</span>` : ""}
            </div>
            <div class="piece-open-meta">
              <span id="save-note" class="save-note">${p.updated_at ? "Saved" : ""}</span>
              ${badge}
            </div>
          </div>

          <div class="scripts-grid">
            <div class="script-pane">
              <div class="script-label-row">
                <label class="script-label" for="edit-native">Native script</label>
                <button type="button" id="chip-needs-native" class="needs-update-chip ${needsNat ? "" : "hidden"}" data-confirm-script="text" title="Roman was edited — click to mark Native ok">Needs update · Confirm ok</button>
              </div>
              <textarea id="edit-native" class="edit-box ${needsNat ? "needs-update" : ""}" rows="3" placeholder="Native script transcription…">${esc(p.current_text)}</textarea>
            </div>

            <div class="script-pane">
              <div class="script-label-row">
                <label class="script-label" for="edit-roman">Romanized</label>
                <button type="button" id="chip-needs-roman" class="needs-update-chip ${needsRom ? "" : "hidden"}" data-confirm-script="text_roman" title="Native was edited — click to mark Roman ok">Needs update · Confirm ok</button>
              </div>
              <textarea id="edit-roman" class="edit-box ${needsRom ? "needs-update" : ""}" rows="3" placeholder="Romanized transcription…">${esc(p.current_text_roman)}</textarea>
            </div>
          </div>

          <div class="langs-row">
            <div class="lang-group">
              <span class="meta-text">Main language:</span>
              ${allMainCodes.map((code) => `
                <button type="button" class="chip-btn ${p.main_language === code ? "selected" : ""}"
                        style="--tag-color:${esc(langColor(code))}"
                        data-set-main="${esc(code)}">${esc(code === "none" ? "No speech" : code)}</button>
              `).join("")}
            </div>
            <div class="lang-group">
              <span class="meta-text">Also contains:</span>
              ${langCodes.filter((c) => c !== p.main_language).map((code) => `
                <button type="button" class="chip-btn ${(p.also_languages || []).includes(code) ? "selected" : ""}"
                        style="--tag-color:${esc(langColor(code))}"
                        data-toggle-also="${esc(code)}">+ ${esc(code)}</button>
              `).join("")}
            </div>
          </div>

          <div class="fold-section">
            <button type="button" id="btn-toggle-edge" class="fold-toggle">
              <span>${state.edgeOpen ? "▾" : "▸"}</span> Edge fix (move start/end, split, merge)
            </button>
            <div id="edge-panel" class="edge-controls ${state.edgeOpen ? "" : "hidden"}">
              <div class="edge-group">
                <span class="meta-text">Start (${p.start_s.toFixed(1)}s):</span>
                ${[-0.5, -0.1, 0.1, 0.5].map((d) => `
                  <button type="button" class="btn btn-quiet" ${canMoveStart ? "" : "disabled"}
                          data-move-side="start" data-move-delta="${d}">${d > 0 ? `+${d}` : d}s</button>
                `).join("")}
              </div>
              <div class="edge-group">
                <span class="meta-text">End (${p.end_s.toFixed(1)}s):</span>
                ${[-0.5, -0.1, 0.1, 0.5].map((d) => `
                  <button type="button" class="btn btn-quiet" ${canMoveEnd ? "" : "disabled"}
                          data-move-side="end" data-move-delta="${d}">${d > 0 ? `+${d}` : d}s</button>
                `).join("")}
              </div>
              <button type="button" id="btn-edge-split" class="btn btn-quiet" ${canEdge && p.duration_s >= 0.85 ? "" : "disabled"}
                      data-split-time="${splitT.toFixed(3)}">Split at ${fmtTimeDec(splitT)}${splitValid ? " (playhead)" : " (mid)"}</button>
              <button type="button" id="btn-edge-merge" class="btn btn-quiet" ${canMerge ? "" : "disabled"}>Merge with next</button>
              ${!canEdge ? `<span class="meta-text">Unverify piece to adjust edges.</span>` : ""}
            </div>
          </div>

          <div class="fold-section">
            <button type="button" id="btn-toggle-hist" class="fold-toggle">
              <span>${state.histOpen ? "▾" : "▸"}</span> ${esc(p.draft_by || "Model draft")} &amp; history
            </button>
            <div id="hist-panel" class="history-panel ${state.histOpen ? "" : "hidden"}">
              ${renderHistoryContent()}
            </div>
          </div>

          <div class="piece-actions">
            <div class="piece-actions-left">
              <button type="button" id="btn-piece-undo" class="btn btn-quiet" title="Undo last saved change (Ctrl+Z)">Undo</button>
              ${isVerified ? `<button type="button" id="btn-piece-unverify" class="btn btn-danger-outline" title="Mark piece as unverified">Unverify</button>` : ""}
            </div>
            <div class="piece-actions-right">
              <button type="button" id="btn-piece-skip" class="btn btn-quiet" title="Skip and open next unverified piece (S)">Skip</button>
              <button type="button" id="btn-piece-verify" class="btn btn-verify" title="Verify & open next (Enter) · Verify & stay (Ctrl+Enter)">✓ Verify</button>
            </div>
          </div>
        </article>`;
    }).join("");

    // Wire events inside rendered pieces list
    cont.querySelectorAll("[data-open-piece]").forEach((btn) => {
      btn.addEventListener("click", () => openPiece(btn.dataset.openPiece, { autoplay: true }));
    });

    const taNat = $("edit-native");
    const taRom = $("edit-roman");
    autoSizeTextarea(taNat);
    autoSizeTextarea(taRom);

    const onTextEdit = (scriptKey) => {
      const piece = currentPiece();
      if (!piece) return;
      piece.current_text = taNat.value;
      piece.current_text_roman = taRom.value;
      // Highlight the other script box until it is edited or confirmed
      if (state.lastEditedBox && state.lastEditedBox !== scriptKey) {
        piece.needs_update_script = "";
      } else if (piece.needs_update_script === scriptKey) {
        piece.needs_update_script = "";
      } else {
        piece.needs_update_script = scriptKey === "text" ? "text_roman" : "text";
      }
      state.lastEditedBox = scriptKey;
      updateNeedsUpdateBorders();
      scheduleAutosave(piece.piece_id);
    };

    if (taNat) taNat.addEventListener("input", () => { autoSizeTextarea(taNat); onTextEdit("text"); });
    if (taRom) taRom.addEventListener("input", () => { autoSizeTextarea(taRom); onTextEdit("text_roman"); });

    cont.querySelectorAll("[data-confirm-script]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        piece.needs_update_script = "";
        state.lastEditedBox = null;
        await savePieceNow(piece.piece_id, { needs_update_script: "" });
        renderPieces();
      });
    });

    cont.querySelectorAll("[data-set-main]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        const ml = btn.dataset.setMain;
        piece.main_language = ml;
        piece.also_languages = (piece.also_languages || []).filter((x) => x !== ml);
        await savePieceNow(piece.piece_id, { main_language: ml, also_languages: piece.also_languages });
        renderPieces();
      });
    });

    cont.querySelectorAll("[data-toggle-also]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        const code = btn.dataset.toggleAlso;
        const cur = new Set(piece.also_languages || []);
        if (cur.has(code)) cur.delete(code);
        else cur.add(code);
        piece.also_languages = Array.from(cur);
        await savePieceNow(piece.piece_id, { also_languages: piece.also_languages });
        renderPieces();
      });
    });

    const toggleEdge = $("btn-toggle-edge");
    if (toggleEdge) {
      toggleEdge.addEventListener("click", () => {
        state.edgeOpen = !state.edgeOpen;
        renderPieces();
      });
    }

    const toggleHist = $("btn-toggle-hist");
    if (toggleHist) {
      toggleHist.addEventListener("click", async () => {
        state.histOpen = !state.histOpen;
        if (state.histOpen && state.openPieceId) {
          state.histData = await api(`/api/pieces/${encodeURIComponent(state.openPieceId)}/history`);
        }
        renderPieces();
      });
    }

    cont.querySelectorAll("[data-move-side]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        const side = btn.dataset.moveSide;
        const delta = parseFloat(btn.dataset.moveDelta);
        const newT = (side === "start" ? piece.start_s : piece.end_s) + delta;
        await runEdgeFix(`/api/pieces/${encodeURIComponent(piece.piece_id)}/move-edge`, { side, time_s: newT });
      });
    });

    const btnSplit = $("btn-edge-split");
    if (btnSplit) {
      btnSplit.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        const splitTime = parseFloat(btnSplit.dataset.splitTime);
        await runEdgeFix(`/api/pieces/${encodeURIComponent(piece.piece_id)}/split`, { time_s: splitTime });
      });
    }

    const btnMerge = $("btn-edge-merge");
    if (btnMerge) {
      btnMerge.addEventListener("click", async () => {
        const piece = currentPiece();
        if (!piece) return;
        await runEdgeFix(`/api/pieces/${encodeURIComponent(piece.piece_id)}/merge`, {});
      });
    }

    const btnUndo = $("btn-piece-undo");
    if (btnUndo) btnUndo.addEventListener("click", () => undoPiece());

    const btnUnv = $("btn-piece-unverify");
    if (btnUnv) {
      btnUnv.addEventListener("click", () => {
        askUnverifyConfirm(async () => {
          const piece = currentPiece();
          if (!piece) return;
          const res = await api(`/api/pieces/${encodeURIComponent(piece.piece_id)}/unverify`, { method: "POST" });
          applyPieceAndCallUpdate(res.piece, res.call);
          loadCalls({ append: false, autoOpenFirst: false }).catch(() => {});
        });
      });
    }

    const btnSkip = $("btn-piece-skip");
    if (btnSkip) btnSkip.addEventListener("click", () => skipPiece());

    const btnVer = $("btn-piece-verify");
    if (btnVer) btnVer.addEventListener("click", () => verifyCurrentPiece(true));
  }

  function renderHistoryContent() {
    const p = currentPiece();
    const hd = state.histData;
    if (!state.histOpen || !hd || hd.piece_id !== p?.piece_id) {
      return `<div class="muted">Loading history…</div>`;
    }
    const d = hd.draft || {};
    const revs = hd.revisions || [];
    const revHtml = revs.length
      ? revs.slice(0, 12).map((r) => `
          <div class="hist-row ${r.undone ? "muted" : ""}">
            <div class="call-item-row">
              <strong>${esc(r.action)}</strong> by ${esc(r.saved_by)} · ${esc(r.saved_at)} ${r.undone ? "(undone)" : ""}
            </div>
            <div>Native: ${renderDiffOpsList(r.word_diff_native)}</div>
            <div>Roman: ${renderDiffOpsList(r.word_diff_roman)}</div>
          </div>`).join("")
      : `<div class="muted">No edits recorded yet.</div>`;

    return `
      <div>
        <div class="meta-text">Previously transcribed draft (${esc(d.draft_by || "none")}${d.draft_language ? ` · ${esc(d.draft_language)}` : ""}):</div>
        <div class="word-diff-line"><strong>Native:</strong> ${esc(d.draft_text || "(empty)")}</div>
        <div class="word-diff-line"><strong>Roman:</strong> ${esc(d.draft_text_roman || "(empty)")}</div>
      </div>
      <div>
        <div class="meta-text">${p.status === "verified" ? "Verified" : "Current"} text vs draft (changes highlighted):</div>
        <div class="word-diff-line"><strong>Native:</strong> ${renderWordDiffHtml(d.diff_from_draft_native, p.current_text)}</div>
        <div class="word-diff-line"><strong>Roman:</strong> ${renderWordDiffHtml(d.diff_from_draft_roman, p.current_text_roman)}</div>
      </div>
      <div class="history-revs">${revHtml}</div>`;
  }

  function updateNeedsUpdateBorders() {
    const p = currentPiece();
    const taNat = $("edit-native");
    const taRom = $("edit-roman");
    if (!p || !taNat || !taRom) return;
    const natNeed = p.needs_update_script === "text";
    const romNeed = p.needs_update_script === "text_roman";
    taNat.classList.toggle("needs-update", natNeed);
    taRom.classList.toggle("needs-update", romNeed);
    const chipNat = $("chip-needs-native");
    const chipRom = $("chip-needs-roman");
    if (chipNat) chipNat.classList.toggle("hidden", !natNeed);
    if (chipRom) chipRom.classList.toggle("hidden", !romNeed);
    const note = $("save-note");
    if (note) note.textContent = "Unsaved";
  }

  // -- Saving, Verifying, Skipping, Undo, Edge Fix -------------------------------------------------------------------

  function scheduleAutosave(pieceId) {
    state.dirtyPieceId = pieceId;
    if (state.autosaveTimer) clearTimeout(state.autosaveTimer);
    state.autosaveTimer = setTimeout(() => {
      state.autosaveTimer = null;
      savePieceNow(pieceId).catch((err) => {
        const note = $("save-note");
        if (note) note.textContent = `Save error: ${err.message}`;
      });
    }, 2000);
  }

  async function flushAutosave() {
    if (state.autosaveTimer) {
      clearTimeout(state.autosaveTimer);
      state.autosaveTimer = null;
      if (state.dirtyPieceId) {
        await savePieceNow(state.dirtyPieceId);
      }
    }
  }

  async function savePieceNow(pieceId, overrides = {}) {
    const piece = state.pieces.find((x) => x.piece_id === pieceId);
    if (!piece) return null;
    if (state.openPieceId === pieceId) {
      const taNat = $("edit-native");
      const taRom = $("edit-roman");
      if (taNat) piece.current_text = taNat.value;
      if (taRom) piece.current_text_roman = taRom.value;
    }
    const payload = {
      current_text: piece.current_text,
      current_text_roman: piece.current_text_roman,
      main_language: piece.main_language,
      also_languages: piece.also_languages,
      needs_update_script: piece.needs_update_script,
      ...overrides,
    };
    const note = $("save-note");
    if (note) note.textContent = "Saving…";
    const res = await api(`/api/pieces/${encodeURIComponent(pieceId)}/save`, { method: "POST", json: payload });
    if (res.piece) {
      const idx = state.pieces.findIndex((x) => x.piece_id === pieceId);
      if (idx >= 0) state.pieces[idx] = res.piece;
    }
    if (res.call) {
      state.call = { ...state.call, ...res.call };
      renderCallHeader();
      updateCallItemInList(state.call);
    }
    state.dirtyPieceId = null;
    if (note) note.textContent = "Saved";
    if (state.histOpen && state.openPieceId === pieceId) {
      state.histData = await api(`/api/pieces/${encodeURIComponent(pieceId)}/history`);
      const hp = $("hist-panel");
      if (hp) hp.innerHTML = renderHistoryContent();
    }
    if (res.piece && res.piece.status === "verified") {
      refreshFilteredProgress();
    }
    return res;
  }

  function applyPieceAndCallUpdate(updatedPiece, updatedCall) {
    if (updatedPiece) {
      const idx = state.pieces.findIndex((x) => x.piece_id === updatedPiece.piece_id);
      if (idx >= 0) state.pieces[idx] = updatedPiece;
    }
    if (updatedCall) {
      state.call = { ...state.call, ...updatedCall };
      renderCallHeader();
      updateCallItemInList(state.call);
    }
    drawWaveform();
    renderPieces();
  }

  async function verifyCurrentPiece(openNext = true) {
    const piece = currentPiece();
    if (!piece) return;
    if (state.autosaveTimer) {
      clearTimeout(state.autosaveTimer);
      state.autosaveTimer = null;
    }
    const taNat = $("edit-native");
    const taRom = $("edit-roman");
    if (taNat) piece.current_text = taNat.value;
    if (taRom) piece.current_text_roman = taRom.value;

    const note = $("save-note");
    try {
      const res = await api(`/api/pieces/${encodeURIComponent(piece.piece_id)}/verify`, {
        method: "POST",
        json: {
          current_text: piece.current_text,
          current_text_roman: piece.current_text_roman,
          main_language: piece.main_language,
          also_languages: piece.also_languages,
        },
      });
      state.dirtyPieceId = null;
      state.lastEditedBox = null;
      applyPieceAndCallUpdate(res.piece, res.call);
      // Refresh top bar progress counter smoothly
      refreshFilteredProgress();
      if (openNext && res.next_unverified_piece_id && res.next_unverified_piece_id !== piece.piece_id) {
        await openPiece(res.next_unverified_piece_id, { autoplay: true });
      } else if (state.histOpen && state.openPieceId === piece.piece_id) {
        state.histData = await api(`/api/pieces/${encodeURIComponent(piece.piece_id)}/history`);
        const hp = $("hist-panel");
        if (hp) hp.innerHTML = renderHistoryContent();
      }
    } catch (err) {
      if (note) note.textContent = err.message;
    }
  }

  async function refreshFilteredProgress() {
    try {
      const f = state.filters;
      const qs = new URLSearchParams({ state: f.state, lang: f.lang, source: f.source, status: f.status, q: f.q, limit: "1", offset: "0" });
      const res = await api(`/api/calls?${qs.toString()}`);
      const vh = res.filtered_verified_hours || 0;
      const th = res.filtered_total_hours || 0;
      const pct = th > 0 ? Math.min(100, (vh / th) * 100) : 0;
      $("top-progress-fill").style.width = `${pct.toFixed(1)}%`;
      $("top-progress-text").textContent = `${vh.toFixed(1)} / ${th.toFixed(1)} h verified`;
      if (res.verified_by_lang) renderLangHoursBar(res.verified_by_lang);
    } catch (_) {}
  }

  function scrollOpenPieceIntoView() {
    const openEl = document.querySelector(".piece-card-open");
    const callArea = $("call-area");
    if (!openEl || !callArea) return;
    const stickyEl = $("call-sticky-top");
    const areaRect = callArea.getBoundingClientRect();
    const stickyH = stickyEl ? stickyEl.offsetHeight : 0;
    const safeTop = areaRect.top + stickyH + 10;
    const elRect = openEl.getBoundingClientRect();
    if (elRect.top < safeTop) {
      callArea.scrollTop -= (safeTop - elRect.top);
    } else if (elRect.bottom > areaRect.bottom - 12) {
      const maxDown = Math.max(0, elRect.top - safeTop);
      const neededDown = elRect.bottom - (areaRect.bottom - 12);
      callArea.scrollTop += Math.min(maxDown, neededDown);
    }
    if (stickyEl) stickyEl.classList.toggle("is-stuck", callArea.scrollTop > 4);
  }

  async function openPiece(pieceId, { autoplay = true } = {}) {
    if (state.openPieceId === pieceId) {
      loadAudioSource({ mode: "piece", autoplay, seekSec: 0 });
      return;
    }
    await flushAutosave();
    state.openPieceId = pieceId;
    state.lastEditedBox = null;
    state.histData = null;
    renderPieces();
    drawWaveform();
    scrollOpenPieceIntoView();
    loadAudioSource({ mode: "piece", autoplay, seekSec: 0 });
    if (state.histOpen) {
      state.histData = await api(`/api/pieces/${encodeURIComponent(pieceId)}/history`);
      const hp = $("hist-panel");
      if (hp) hp.innerHTML = renderHistoryContent();
    }
  }

  async function skipPiece() {
    await flushAutosave();
    if (!state.pieces.length) return;
    const curIdx = state.pieces.findIndex((x) => x.piece_id === state.openPieceId);
    for (let step = 1; step <= state.pieces.length; step++) {
      const cand = state.pieces[(curIdx + step) % state.pieces.length];
      if (cand.piece_id !== state.openPieceId && cand.status === "unverified") {
        await openPiece(cand.piece_id, { autoplay: true });
        return;
      }
    }
    // Fallback: move to next adjacent piece even if verified
    if (state.pieces.length > 1) {
      await openPiece(state.pieces[(curIdx + 1) % state.pieces.length].piece_id, { autoplay: true });
    }
  }

  function stepPiece(dir) {
    if (!state.pieces.length) return;
    const idx = Math.max(0, state.pieces.findIndex((x) => x.piece_id === state.openPieceId));
    const targetIdx = clampNum(idx + dir, 0, state.pieces.length - 1);
    if (targetIdx !== idx) {
      openPiece(state.pieces[targetIdx].piece_id, { autoplay: true });
    }
  }

  async function undoPiece(historyId = null, confirmUnverify = false) {
    const piece = currentPiece();
    if (!piece) return;
    await flushAutosave();
    const res = await api(`/api/pieces/${encodeURIComponent(piece.piece_id)}/undo`, {
      method: "POST",
      json: { history_id: historyId, confirm_unverify: confirmUnverify },
    });
    if (res.needs_confirm) {
      askUnverifyConfirm(() => undoPiece(res.history_id, true));
      return;
    }
    if (!res.undone) {
      const note = $("save-note");
      if (note) note.textContent = res.message || "Nothing to undo";
      return;
    }
    if (res.edge_restored) {
      state.pieces = res.pieces || [];
      if (!state.pieces.some((x) => x.piece_id === state.openPieceId)) {
        state.openPieceId = (state.pieces[0] || {}).piece_id || null;
      }
      state.call = { ...state.call, ...res.call };
      renderCallHeader();
      updateCallItemInList(state.call);
      renderPieces();
      drawWaveform();
      if (state.openPieceId) loadAudioSource({ mode: "piece", autoplay: false, seekSec: 0 });
    } else {
      applyPieceAndCallUpdate(res.piece, res.call);
    }
    refreshFilteredProgress();
  }

  async function runEdgeFix(url, body) {
    await flushAutosave();
    try {
      const res = await api(url, { method: "POST", json: body });
      state.pieces = res.pieces || [];
      if (!state.pieces.some((x) => x.piece_id === state.openPieceId)) {
        state.openPieceId = (state.pieces[0] || {}).piece_id || null;
      }
      if (res.call) {
        state.call = { ...state.call, ...res.call };
        renderCallHeader();
        updateCallItemInList(state.call);
      }
      renderPieces();
      drawWaveform();
      if (state.openPieceId) loadAudioSource({ mode: "piece", autoplay: true, seekSec: 0 });
    } catch (err) {
      const note = $("save-note");
      if (note) note.textContent = err.message;
    }
  }

  function askUnverifyConfirm(onOk) {
    state.confirmCallback = onOk;
    $("confirm-overlay").classList.remove("hidden");
    $("btn-confirm-ok").focus();
  }

  $("btn-confirm-cancel").addEventListener("click", () => {
    state.confirmCallback = null;
    $("confirm-overlay").classList.add("hidden");
  });
  $("btn-confirm-ok").addEventListener("click", async () => {
    const cb = state.confirmCallback;
    state.confirmCallback = null;
    $("confirm-overlay").classList.add("hidden");
    if (cb) await cb();
  });

  // -- Spelling Guide Drawer (docked beside call area; never covers text boxes) --------------------------------------

  function renderMarkdownSafe(md) {
    const lines = String(md || "").replace(/\r/g, "").split("\n");
    const out = [];
    let inUl = false;
    let inOl = false;
    const closeLists = () => {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
    };
    const inlineFmt = (s) => esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    for (const line of lines) {
      const t = line.trim();
      if (!t) { closeLists(); continue; }
      if (t.startsWith("### ")) { closeLists(); out.push(`<h3>${inlineFmt(t.slice(4))}</h3>`); continue; }
      if (t.startsWith("## ")) { closeLists(); out.push(`<h2>${inlineFmt(t.slice(3))}</h2>`); continue; }
      if (t.startsWith("# ")) { closeLists(); out.push(`<h1>${inlineFmt(t.slice(2))}</h1>`); continue; }
      const olMatch = t.match(/^(\d+)\.\s+(.*)$/);
      if (olMatch) {
        if (inUl) { out.push("</ul>"); inUl = false; }
        if (!inOl) { out.push("<ol>"); inOl = true; }
        out.push(`<li>${inlineFmt(olMatch[2])}</li>`);
        continue;
      }
      if (t.startsWith("- ") || t.startsWith("* ")) {
        if (inOl) { out.push("</ol>"); inOl = false; }
        if (!inUl) { out.push("<ul>"); inUl = true; }
        out.push(`<li>${inlineFmt(t.slice(2))}</li>`);
        continue;
      }
      closeLists();
      out.push(`<p>${inlineFmt(t)}</p>`);
    }
    closeLists();
    return out.join("");
  }

  async function toggleGuide(forceState = null) {
    state.guideOpen = forceState === null ? !state.guideOpen : forceState;
    $("guide-drawer").classList.toggle("hidden", !state.guideOpen);
    $("btn-guide").classList.toggle("btn-active", state.guideOpen);
    if (state.guideOpen && !state.guideMarkdown) {
      const res = await api("/api/guide");
      state.guideMarkdown = res.content || "";
      $("guide-view").innerHTML = renderMarkdownSafe(state.guideMarkdown);
    }
    setTimeout(() => drawWaveform(), 60);
  }

  $("btn-guide").addEventListener("click", () => toggleGuide());
  $("btn-guide-close").addEventListener("click", () => toggleGuide(false));
  $("btn-guide-edit").addEventListener("click", () => {
    state.guideEditing = true;
    $("guide-editor").value = state.guideMarkdown;
    $("guide-view").classList.add("hidden");
    $("guide-editor").classList.remove("hidden");
    $("btn-guide-edit").classList.add("hidden");
    $("btn-guide-save").classList.remove("hidden");
    $("btn-guide-cancel").classList.remove("hidden");
  });
  $("btn-guide-cancel").addEventListener("click", () => {
    state.guideEditing = false;
    $("guide-view").classList.remove("hidden");
    $("guide-editor").classList.add("hidden");
    $("btn-guide-edit").classList.remove("hidden");
    $("btn-guide-save").classList.add("hidden");
    $("btn-guide-cancel").classList.add("hidden");
  });
  $("btn-guide-save").addEventListener("click", async () => {
    const content = $("guide-editor").value;
    const res = await api("/api/guide", { method: "PUT", json: { content } });
    state.guideMarkdown = res.content;
    $("guide-view").innerHTML = renderMarkdownSafe(state.guideMarkdown);
    $("btn-guide-cancel").click();
  });

  // -- Progress & WER Modal ------------------------------------------------------------------------------------------

  async function openProgressModal() {
    $("progress-overlay").classList.remove("hidden");
    $("progress-content").innerHTML = `<p class="muted">Calculating progress &amp; model error rates…</p>`;
    const d = await api("/api/progress");
    const s = d.summary;
    const langRows = (d.by_lang || []).map((r) => `
      <tr>
        <td>${renderLangTag(r.language)} ${esc(r.label)}</td>
        <td class="num">${r.verified_hours.toFixed(2)} / ${r.total_hours.toFixed(2)} h</td>
        <td class="num">${r.verified_pieces.toLocaleString()} / ${r.total_pieces.toLocaleString()}</td>
        <td class="num">${r.v3_wer_native === null ? "—" : `${r.v3_wer_native.toFixed(1)}%`}</td>
        <td class="num">${r.v3_wer_roman === null ? "—" : `${r.v3_wer_roman.toFixed(1)}%`}</td>
        <td class="num">${r.v3_eval_pieces.toLocaleString()}</td>
      </tr>`).join("");

    const stateRows = (d.by_state || []).map((r) => `
      <tr>
        <td>${esc(r.state)}</td>
        <td class="num">${r.verified_hours.toFixed(2)} / ${r.total_hours.toFixed(2)} h</td>
        <td class="num">${r.verified_pieces.toLocaleString()} / ${r.total_pieces.toLocaleString()}</td>
      </tr>`).join("");

    $("progress-content").innerHTML = `
      <div class="status-box">
        <strong>Overall:</strong> ${s.verified_hours.toFixed(2)} / ${s.total_hours.toFixed(2)} h verified
        (${s.verified_pieces.toLocaleString()} / ${s.total_pieces.toLocaleString()} pieces ·
        ${s.complete_calls.toLocaleString()} complete calls / ${s.verified_calls.toLocaleString()} touched)
      </div>
      <div>
        <div class="field-label">By Language (v3 draft WER measured against your verified labels)</div>
        <table class="data-table">
          <thead>
            <tr>
              <th>Language</th>
              <th class="num">Verified / Total h</th>
              <th class="num">Pieces</th>
              <th class="num">v3 WER (Native)</th>
              <th class="num">v3 WER (Roman)</th>
              <th class="num">v3 Checked</th>
            </tr>
          </thead>
          <tbody>${langRows || `<tr><td colspan="6" class="muted">No pieces yet</td></tr>`}</tbody>
        </table>
      </div>
      <div>
        <div class="field-label">By State / Region</div>
        <table class="data-table">
          <thead><tr><th>State</th><th class="num">Verified / Total h</th><th class="num">Pieces</th></tr></thead>
          <tbody>${stateRows}</tbody>
        </table>
      </div>`;
  }

  $("btn-progress").addEventListener("click", openProgressModal);
  $("btn-backup-now").addEventListener("click", async () => {
    const res = await api("/api/backup", { method: "POST" });
    $("btn-backup-now").textContent = res.backup_path ? "Backed up ✓" : "No DB yet";
    setTimeout(() => { $("btn-backup-now").textContent = "Backup DB"; }, 2500);
  });

  // -- Import Kaggle Zips Modal --------------------------------------------------------------------------------------

  $("btn-import").addEventListener("click", () => {
    $("import-result").classList.add("hidden");
    $("import-overlay").classList.remove("hidden");
  });

  function showImportResult(res) {
    const box = $("import-result");
    box.classList.remove("hidden");
    box.innerHTML = `
      <strong>Synced:</strong> ${esc((res.staged_files || []).join(", ") || "updated")}<br>
      Rows read: <strong>${res.rows_in_source.toLocaleString()}</strong> ·
      Newly drafted pieces: <strong>${res.new_pieces_with_draft.toLocaleString()}</strong> ·
      Verified preserved untouched: <strong>${res.verified_preserved.toLocaleString()}</strong>`;
    loadFilters().catch(() => {});
    loadCalls({ append: false, autoOpenFirst: false }).catch(() => {});
  }

  $("btn-import-upload").addEventListener("click", async () => {
    const fi = $("import-file").files[0];
    if (!fi) return;
    const box = $("import-result");
    box.classList.remove("hidden");
    box.textContent = "Uploading and syncing labelled data…";
    const fd = new FormData();
    fd.append("file", fi);
    try {
      const res = await api("/api/import/upload", { method: "POST", body: fd });
      showImportResult(res);
    } catch (err) {
      box.innerHTML = `<span class="form-error">${esc(err.message)}</span>`;
    }
  });

  $("btn-import-path").addEventListener("click", async () => {
    const p = $("import-path").value.trim();
    if (!p) return;
    const box = $("import-result");
    box.classList.remove("hidden");
    box.textContent = "Importing path and syncing…";
    try {
      const res = await api("/api/import/path", { method: "POST", json: { path: p } });
      showImportResult(res);
    } catch (err) {
      box.innerHTML = `<span class="form-error">${esc(err.message)}</span>`;
    }
  });

  // -- Overlap Check & Train/Test Split Modal ------------------------------------------------------------------------

  async function refreshSplitStatus() {
    const st = await api("/api/split/status");
    const sc = st.scan || {};
    const scanLine = sc.running
      ? `Scanning (${sc.scope})… ${sc.done}/${sc.total} · matches found: ${sc.matches_found}`
      : (sc.error ? `Error: ${sc.error}` : `Cached audio fingerprints: ${st.fingerprints_cached.toLocaleString()} · Confirmed overlaps (old ↔ crm): ${st.overlaps_count}`);
    const ovRows = (st.overlaps || []).slice(0, 20).map((o) => `
      <tr>
        <td>${esc(o.old_call_id)}</td>
        <td>${esc(o.crm_call_id)}</td>
        <td class="num">${(o.score * 100).toFixed(1)}%</td>
        <td class="num">${o.dur_diff_s.toFixed(2)}s</td>
      </tr>`).join("");
    const spLine = Object.entries(st.current_splits || {}).map(([k, v]) => `${k}: ${v} calls`).join(" · ") || "Not applied yet";
    $("split-status-box").innerHTML = `
      <div class="status-box">${esc(scanLine)}<br>Current splits: <strong>${esc(spLine)}</strong></div>
      ${ovRows ? `<table class="data-table"><thead><tr><th>Old Call</th><th>CRM Call</th><th class="num">Similarity</th><th class="num">Δ dur</th></tr></thead><tbody>${ovRows}</tbody></table>` : ""}`;
    if (sc.running) {
      if (state.splitPollTimer) clearTimeout(state.splitPollTimer);
      state.splitPollTimer = setTimeout(refreshSplitStatus, 1000);
    }
  }

  $("btn-split").addEventListener("click", () => {
    $("split-overlay").classList.remove("hidden");
    refreshSplitStatus().catch(() => {});
  });

  async function triggerOverlapScan(scope) {
    await api("/api/split/check-overlaps", { method: "POST", json: { scope, background: true } });
    refreshSplitStatus();
  }
  $("btn-check-overlaps-verified").addEventListener("click", () => triggerOverlapScan("verified"));
  $("btn-check-overlaps-all").addEventListener("click", () => triggerOverlapScan("all"));

  async function runSplitPlan(apply) {
    const ratio = clampNum(parseFloat($("split-ratio").value || "0.15"), 0.01, 0.5);
    const endpoint = apply ? "/api/split/apply" : "/api/split/preview";
    const res = await api(endpoint, { method: "POST", json: { test_ratio: ratio, seed: 42, respect_old_train: true } });
    const rows = Object.values(res.by_lang || {}).map((l) => `
      <tr>
        <td>${renderLangTag(l.language)}</td>
        <td class="num">${l.train_hours.toFixed(2)}h (${l.train_pieces})</td>
        <td class="num">${l.test_hours.toFixed(2)}h (${l.test_pieces})</td>
        <td class="num">${(l.test_ratio_actual * 100).toFixed(1)}%</td>
      </tr>`).join("");
    $("split-status-box").innerHTML = `
      <div class="status-box">
        <strong>${res.applied ? "Applied & exported to dataset.jsonl!" : "Preview"}</strong>
        — Verified: ${res.total_verified_pieces} pieces (${res.total_verified_hours.toFixed(2)} h) ·
        Lead/overlap leakage: <strong>${res.leakage_check.shared_leads_across_splits + res.leakage_check.shared_overlaps_across_splits}</strong>
      </div>
      <table class="data-table">
        <thead><tr><th>Lang</th><th class="num">Train</th><th class="num">Test</th><th class="num">Test %</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="4" class="muted">No verified pieces yet</td></tr>`}</tbody>
      </table>`;
    if (res.applied && state.currentCallId) {
      selectCall(state.currentCallId, { autoplay: false, openPieceId: state.openPieceId }).catch(() => {});
    }
  }
  $("btn-preview-split").addEventListener("click", () => runSplitPlan(false));
  $("btn-apply-split").addEventListener("click", () => runSplitPlan(true));

  // -- User & Admin modal --------------------------------------------------------------------------------------------

  async function openUserModal() {
    $("user-overlay").classList.remove("hidden");
    const u = state.user;
    if (!u) return;
    let usersHtml = "";
    if (u.role === "admin") {
      try {
        const res = await api("/api/users");
        const rows = (res.users || []).map((item) => `
          <tr>
            <td>${esc(item.username)}</td>
            <td>${esc(item.display_name)}</td>
            <td>${esc(item.role)}</td>
            <td>${item.active ? "active" : "disabled"}</td>
            <td>
              <button class="btn btn-quiet" data-toggle-user="${esc(item.username)}" data-next-active="${!item.active}">
                ${item.active ? "Disable" : "Enable"}
              </button>
            </td>
          </tr>`).join("");
        usersHtml = `
          <table class="data-table">
            <thead><tr><th>Username</th><th>Name</th><th>Role</th><th>Status</th><th></th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <form id="add-user-form" class="import-box">
            <div class="field-label">Add / Update Reviewer (required for ngrok remote logins)</div>
            <div class="row-gap">
              <input id="new-u-name" type="text" placeholder="username" required />
              <input id="new-u-display" type="text" placeholder="Display name" />
              <input id="new-u-pass" type="password" placeholder="password (min 4)" required />
              <select id="new-u-role"><option value="reviewer">reviewer</option><option value="admin">admin</option></select>
              <button type="submit" class="btn btn-accent">Save User</button>
            </div>
            <div id="user-form-msg" class="meta-text"></div>
          </form>`;
      } catch (_) {}
    }

    $("user-modal-body").innerHTML = `
      <div class="status-box">
        Signed in as <strong>${esc(u.display_name)}</strong> (<code>${esc(u.username)}</code>, ${esc(u.role)}) ·
        Connection: <strong>${state.isLocal ? "Localhost (lossless)" : "Remote (login + 24 kbps Opus)"}</strong>
        ${!state.isLocal ? `<button id="btn-logout-modal" class="btn btn-danger-outline" style="margin-left:12px">Sign out</button>` : ""}
      </div>
      ${usersHtml}`;

    const btnLogout = $("btn-logout-modal");
    if (btnLogout) {
      btnLogout.addEventListener("click", async () => {
        await api("/api/logout", { method: "POST" });
        $("user-overlay").classList.add("hidden");
        showLogin();
      });
    }
    const form = $("add-user-form");
    if (form) {
      form.addEventListener("submit", async (e) => {
        e.preventDefault();
        try {
          await api("/api/users", {
            method: "POST",
            json: {
              username: $("new-u-name").value,
              display_name: $("new-u-display").value,
              password: $("new-u-pass").value,
              role: $("new-u-role").value,
            },
          });
          openUserModal();
        } catch (err) {
          $("user-form-msg").textContent = err.message;
        }
      });
    }
    $("user-modal-body").querySelectorAll("[data-toggle-user]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        await api("/api/users", {
          method: "POST",
          json: { username: btn.dataset.toggleUser, active: btn.dataset.nextActive === "true" },
        });
        openUserModal();
      });
    });
  }

  $("btn-user").addEventListener("click", openUserModal);

  // Modal close buttons
  document.querySelectorAll("[data-close]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $(btn.dataset.close).classList.add("hidden");
    });
  });

  // Mobile sidebar drawer
  function closeMobileDrawer() {
    $("call-sidebar").classList.remove("drawer-open");
    $("drawer-backdrop").classList.add("hidden");
  }
  $("btn-drawer-toggle").addEventListener("click", () => {
    const opening = !$("call-sidebar").classList.contains("drawer-open");
    $("call-sidebar").classList.toggle("drawer-open", opening);
    $("drawer-backdrop").classList.toggle("hidden", !opening);
  });
  $("drawer-backdrop").addEventListener("click", closeMobileDrawer);

  // -- Filter, Player Controls & Global Keyboard Shortcuts -----------------------------------------------------------

  ["filter-state", "filter-lang", "filter-source", "filter-status"].forEach((id) => {
    $(id).addEventListener("change", () => {
      state.filters.state = $("filter-state").value;
      state.filters.lang = $("filter-lang").value;
      state.filters.source = $("filter-source").value;
      state.filters.status = $("filter-status").value;
      loadCalls({ append: false, autoOpenFirst: true });
    });
  });

  $("sort-calls").addEventListener("change", () => {
    state.filters.sort = $("sort-calls").value;
    loadCalls({ append: false, autoOpenFirst: false });
  });

  let searchTimer = null;
  $("filter-search").addEventListener("input", () => {
    if (searchTimer) clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.filters.q = $("filter-search").value;
      loadCalls({ append: false, autoOpenFirst: false });
    }, 220);
  });

  $("btn-clear-filters").addEventListener("click", () => {
    $("filter-state").value = "all";
    $("filter-lang").value = "all";
    $("filter-source").value = "all";
    $("filter-status").value = "ready";
    $("filter-search").value = "";
    Object.assign(state.filters, { state: "all", lang: "all", source: "all", status: "ready", q: "" });
    loadCalls({ append: false, autoOpenFirst: true });
  });

  $("btn-load-more").addEventListener("click", () => {
    state.filters.offset += state.filters.limit;
    loadCalls({ append: true, autoOpenFirst: false });
  });

  $("btn-play").addEventListener("click", togglePlay);
  $("btn-back2").addEventListener("click", () => seekDelta(-2));
  $("btn-mode-piece").addEventListener("click", () => loadAudioSource({ mode: "piece", autoplay: true, seekSec: 0 }));
  $("btn-mode-call").addEventListener("click", () => loadAudioSource({ mode: "call", autoplay: true, seekSec: getCallPlayheadSec() }));
  $("speed-slider").addEventListener("input", () => {
    setSpeed(parseFloat($("speed-slider").value));
  });
  $("btn-speed-down").addEventListener("click", () => setSpeed(state.speed - 0.1));
  $("btn-speed-up").addEventListener("click", () => setSpeed(state.speed + 0.1));
  document.querySelectorAll(".speed-preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => setSpeed(parseFloat(btn.dataset.speed)));
  });

  $("call-area").addEventListener("scroll", () => {
    const stickyTop = $("call-sticky-top");
    if (stickyTop) stickyTop.classList.toggle("is-stuck", $("call-area").scrollTop > 4);
  }, { passive: true });
  $("btn-boost").addEventListener("click", () => {
    state.boost = !state.boost;
    if (state.boost) initBoostGraph();
    applyBoostUi();
  });
  $("btn-loop").addEventListener("click", () => {
    state.loop = !state.loop;
    applyLoopUi();
  });
  $("btn-opus").addEventListener("click", () => {
    state.useOpus = !state.useOpus;
    localStorage.setItem(LS.opus, state.useOpus ? "1" : "0");
    applyOpusUi();
    loadAudioSource({ mode: state.playMode, autoplay: state.playing, seekSec: audioEl.currentTime || 0 });
  });

  $("btn-toggle-ref").addEventListener("click", () => {
    state.refOpen = !state.refOpen;
    renderReferenceCard();
  });

  $("btn-theme").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    applyTheme(cur === "dark" ? "light" : "dark");
  });

  // Keyboard shortcuts following PLAN.md strictly
  document.addEventListener("keydown", (e) => {
    const inLogin = !$("login-screen").classList.contains("hidden");
    if (inLogin) return;

    // Escape closes active confirmation or modal or guide
    if (e.key === "Escape") {
      if (!$("confirm-overlay").classList.contains("hidden")) { $("btn-confirm-cancel").click(); return; }
      for (const id of ["progress-overlay", "import-overlay", "split-overlay", "user-overlay"]) {
        if (!$(id).classList.contains("hidden")) { $(id).classList.add("hidden"); return; }
      }
      if (state.guideOpen) toggleGuide(false);
      return;
    }

    const activeEl = document.activeElement;
    const typing = isTypingInInput(activeEl);
    const inButton = activeEl && (activeEl.tagName === "BUTTON" || activeEl.tagName === "A");
    const inPieceBox = activeEl && (activeEl.id === "edit-native" || activeEl.id === "edit-roman");
    const anyModalOpen = !["confirm-overlay", "progress-overlay", "import-overlay", "split-overlay", "user-overlay"]
      .every((id) => $(id).classList.contains("hidden"));
    if (anyModalOpen) return;

    // Shortcuts safe even inside text boxes (Alt or Ctrl+Enter / Enter inside piece boxes)
    if (e.altKey && e.key === "ArrowUp") { e.preventDefault(); stepPiece(-1); return; }
    if (e.altKey && e.key === "ArrowDown") { e.preventDefault(); stepPiece(1); return; }
    if (e.altKey && e.key === "ArrowLeft") { e.preventDefault(); seekDelta(-2); return; }
    if (e.altKey && e.key === "ArrowRight") { e.preventDefault(); seekDelta(2); return; }
    if (e.altKey && e.code === "Space") { e.preventDefault(); togglePlay(); return; }

    if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && state.openPieceId) {
      e.preventDefault();
      verifyCurrentPiece(false);
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey && state.openPieceId) {
      if (!inButton && (!typing || inPieceBox)) {
        e.preventDefault();
        verifyCurrentPiece(true);
        return;
      }
    }

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && state.openPieceId) {
      // Outside dirty unsaved typing buffer -> trigger server undo
      if (!inPieceBox || state.dirtyPieceId === null) {
        e.preventDefault();
        undoPiece();
        return;
      }
    }

    // Never interfere with typing when any text box is focused
    if (typing) return;

    if (e.code === "Space") {
      if (!inButton) { e.preventDefault(); togglePlay(); }
      return;
    }
    if (e.key === "ArrowLeft") { e.preventDefault(); seekDelta(-2); return; }
    if (e.key === "ArrowRight") { e.preventDefault(); seekDelta(2); return; }
    if (e.key === "[") { e.preventDefault(); setSpeed(state.speed - 0.1); return; }
    if (e.key === "]") { e.preventDefault(); setSpeed(state.speed + 0.1); return; }
    if (e.key.toLowerCase() === "l") { e.preventDefault(); state.loop = !state.loop; applyLoopUi(); return; }
    if (e.key.toLowerCase() === "s") { e.preventDefault(); skipPiece(); return; }
    if (e.key === "?") { e.preventDefault(); toggleGuide(); return; }
  });

  // -- Boot & Auth ---------------------------------------------------------------------------------------------------

  function showLogin() {
    $("app-shell").classList.add("hidden");
    $("login-screen").classList.remove("hidden");
    $("login-username").focus();
  }

  async function enterApp(me) {
    state.user = me.user;
    state.isLocal = !!me.is_local;
    const savedOpus = localStorage.getItem(LS.opus);
    state.useOpus = savedOpus !== null ? savedOpus === "1" : (me.default_audio_fmt === "opus");

    $("login-screen").classList.add("hidden");
    $("app-shell").classList.remove("hidden");
    $("btn-user").textContent = me.user.display_name || me.user.username;

    applyTheme(localStorage.getItem(LS.theme) || "light");
    applySpeedUi();
    applyBoostUi();
    applyLoopUi();
    applyOpusUi();
    applyPlayModeUi();

    await loadFilters();
    await loadCalls({ append: false, autoOpenFirst: true });
  }

  $("login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    $("login-error").classList.add("hidden");
    try {
      const me = await api("/api/login", {
        method: "POST",
        json: { username: $("login-username").value, password: $("login-password").value },
      });
      await enterApp(me);
    } catch (err) {
      $("login-error").textContent = err.message;
      $("login-error").classList.remove("hidden");
    }
  });

  (async function boot() {
    applyTheme(localStorage.getItem(LS.theme) || "light");
    try {
      const me = await api("/api/me");
      if (!me.authenticated) {
        showLogin();
        return;
      }
      await enterApp(me);
    } catch (_) {
      showLogin();
    }
  })();
})();

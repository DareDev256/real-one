/* Real One watch page. Two <video>s of one render (the finished look with the sound, the receipt look muted on top),
   held within a frame of each other; a counter read from the render's own receipt json; the song's sections from its
   shot plan; the director's treatment beside the lyric. Everything comes from data/, written by
   scripts/build_watch_page.py. Nothing here invents a number: a missing file hides its panel. */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const A = $("v-finished");   // master: carries the audio and the clock
  const B = $("v-receipt");    // follower: muted, cross-faded on top
  const stage = $("stage");
  const S = { fps: 30, frames: 0, dur: 0, start: 0, receipt: null, shots: null, treatment: null, lyrics: null,
              sections: [],
              manifest: null, on: false, playing: false, lastFrame: -1, lastSec: null, seekToken: 0 };
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");

  async function getJSON(name) {
    const r = await fetch("data/" + name, { cache: "no-cache" });
    if (!r.ok) throw new Error(name + ": HTTP " + r.status);
    return r.json();
  }
  const opt = (name) => (name ? getJSON(name).catch(() => null) : Promise.resolve(null));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const pad = (n, w) => String(n).padStart(w, "0");
  const fmtTime = (t) => { t = Math.max(0, t); const m = Math.floor(t / 60); const s = t - m * 60;
    return pad(m, 2) + ":" + (s < 10 ? "0" : "") + s.toFixed(2); };
  const pct = (x) => (x == null ? "NO READ" : (Math.round(x * 10) / 10).toFixed(1) + "%");
  const split = (x) => { if (x == null || Number.isNaN(x)) return [null, null];
    const f = Math.round(x * 10) / 10; return [f, Math.round((100 - f) * 10) / 10]; };

  // ------------------------------------------------------------------ sync
  function frameOf(t) { return clamp(Math.floor(t * S.fps + 1e-3), 0, S.frames - 1); }

  function seekOne(v, t) {
    return new Promise((res) => {
      if (Math.abs(v.currentTime - t) < 1e-4 && !v.seeking) return res();
      const done = () => { v.removeEventListener("seeked", done); res(); };
      v.addEventListener("seeked", done);
      v.currentTime = t;
    });
  }

  function ready(v) {   // enough decoded to start without one of the pair stalling at once
    return new Promise((res) => {
      if (v.readyState >= 3) return res();
      const done = () => { v.removeEventListener("canplay", done); res(); };
      v.addEventListener("canplay", done);
      setTimeout(done, 1500);
    });
  }

  async function seek(t) {
    const token = ++S.seekToken;
    const was = S.playing;
    t = clamp(t, 0, Math.max(0, S.dur - 0.5 / S.fps));
    A.pause(); B.pause();
    await Promise.all([seekOne(A, t), seekOne(B, t)]);
    await Promise.all([ready(A), ready(B)]);
    if (token !== S.seekToken) return;   // a newer seek took over
    render(true);
    if (was) start();
  }

  function start() {
    // A first: the one with the sound starts late (audio priming). B waits for A's clock to move, then joins it
    // (hold() jumps B onto A, plus the learned seek lead, and plays it).
    S.playing = true; stage.classList.add("playing"); setPlayLabel();
    B.pause(); B.playbackRate = 1;
    SYNC.lastA = A.currentTime; SYNC.aMovedAt = -1e9; SYNC.held = true;
    A.play().catch(() => { stop(); });
  }

  function stop() {
    S.playing = false; stage.classList.remove("playing"); setPlayLabel();
    A.pause(); B.pause(); B.playbackRate = 1;
    if (Math.abs(B.currentTime - A.currentTime) > 1e-3) B.currentTime = A.currentTime;
  }

  function toggle() { S.playing ? stop() : start(); }

  // Hold B on A's clock.
  // - A stalls (its clock stops while B's runs on, common for a moment after a seek): B waits, then re-aligns.
  // - Large drift: jump B, aimed ahead by the seek time this browser has shown it needs (learned from the
  //   residual 300 ms after each jump).
  // - Small drift: nudge B's rate, rarely (WebKit drops about a frame on every rate change, so a rate that changes
  //   each animation frame makes drift worse): change it only past 1 frame, put it back under a quarter frame.
  const SYNC = { rateAt: 0, jumpAt: -1e9, check: 0, lead: 2 / 30, lastA: -1, aMovedAt: 0, held: false };
  function jumpB(now) {
    B.playbackRate = 1; B.currentTime = A.currentTime + SYNC.lead;
    SYNC.jumpAt = now; SYNC.check = now + 300;
  }
  function hold(now) {
    if (!S.playing || A.paused || A.seeking || B.seeking || A.readyState < 2) return;
    const fr = 1 / S.fps;
    if (A.currentTime !== SYNC.lastA) { SYNC.lastA = A.currentTime; SYNC.aMovedAt = now; }
    if (now - SYNC.aMovedAt > 70) {                       // A's clock is stuck or not started: hold B
      if (!B.paused) { B.pause(); SYNC.held = true; }
      return;
    }
    if (SYNC.held) {                                      // A moves again: bring B back on it
      SYNC.held = false; jumpB(now); B.play().catch(() => {}); return;
    }
    if (B.paused) { B.play().catch(() => {}); return; }
    const d = B.currentTime - A.currentTime;   // > 0: B is ahead
    if (SYNC.check && now >= SYNC.check) { SYNC.lead = clamp(SYNC.lead - d, 0, 0.4); SYNC.check = 0; }
    if (Math.abs(d) > 2.5 * fr && now - SYNC.jumpAt > 400) { jumpB(now); return; }
    if (now - SYNC.rateAt < 200) return;
    let want = B.playbackRate;
    if (Math.abs(d) > 1 * fr) want = d > 0 ? 0.92 : 1.08;
    else if (Math.abs(d) < 0.25 * fr) want = 1;
    if (want !== B.playbackRate) { B.playbackRate = want; SYNC.rateAt = now; }
  }

  A.addEventListener("waiting", () => { if (S.playing && !B.paused) { B.pause(); SYNC.held = true; } });
  A.addEventListener("ended", () => { stop(); });

  function loop(now) {
    hold(now || performance.now());
    render(false);
    requestAnimationFrame(loop);
  }

  function setReceipt(on) {
    S.on = on;
    stage.classList.toggle("on", on);
    const b = $("btn-receipt");
    b.setAttribute("aria-pressed", on ? "true" : "false");
    $("receipt-state").textContent = on ? "ON" : "OFF";
    if (!S.playing && Math.abs(B.currentTime - A.currentTime) > 1e-3) B.currentTime = A.currentTime;
    render(true);
  }

  function setPlayLabel() {
    $("btn-play").textContent = S.playing ? "PAUSE" : "PLAY";
    $("btn-play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
    $("stage-play").setAttribute("aria-label", S.playing ? "Pause" : "Play");
  }

  // ------------------------------------------------------------------ counter
  function lookName() { return S.on ? S.manifest.looks.receipt : S.manifest.looks.finished; }

  function render(force) {
    if (!S.receipt) return;
    const t = A.currentTime;
    const f = frameOf(t);
    const playFrac = S.dur ? clamp(t / S.dur, 0, 1) : 0;
    $("strip-head").style.left = (playFrac * 100).toFixed(3) + "%";
    $("strip-rail").setAttribute("aria-valuenow", t.toFixed(2));
    $("clock").textContent = fmtTime(t) + " / " + fmtTime(S.dur);
    if (f === S.lastFrame && !force) return;
    S.lastFrame = f;
    const out = S.receipt.outputs[lookName()];
    const [fi, ge] = split(out.frames.filmed_pct[f]);
    $("r-frame").textContent = pad(f + 1, 4) + " / " + pad(S.frames, 4);
    $("r-time").textContent = fmtTime(t);
    $("r-filmed").textContent = pct(fi);
    $("r-gen").textContent = pct(ge);
    $("o-filmed").textContent = pct(fi);
    $("o-gen").textContent = pct(ge);
    $("r-bar").style.width = (fi == null ? 0 : fi) + "%";
    $("w-filmed").textContent = pct(out.video.filmed_pct);
    $("w-gen").textContent = pct(out.video.generated_pct);
    $("slip-look").textContent = lookName().toUpperCase() + " LOOK";
    document.body.dataset.frame = String(f);
    sectionNow(S.start + t);
  }

  // ------------------------------------------------------------------ sections strip
  function buildStrip() {
    const sh = S.shots, start = S.start, end = S.start + S.dur;
    const secs = $("strip-sections"), fz = $("strip-freezes"), cuts = $("strip-cuts"), bars = $("strip-bars");
    const at = (songT) => clamp((songT - start) / S.dur, 0, 1) * 100;
    const frozen = new Set((sh.freezes || []).map((z) => z.section_index));
    S.sections.forEach((s) => {
      const a = Math.max(s.start_s, start), b = Math.min(s.end_s, end);
      if (b <= a) return;
      const el = document.createElement("button");
      el.type = "button"; el.className = "sec"; el.dataset.index = s.index;
      el.style.left = at(a) + "%"; el.style.width = (at(b) - at(a)) + "%";
      const narrow = (b - a) / S.dur < 0.16;
      if (narrow) el.classList.add("narrow");
      el.innerHTML = "<b></b><small></small>";
      el.querySelector("b").textContent = s.role.toUpperCase() + (narrow ? "" : " #" + s.index);
      el.querySelector("small").textContent = narrow ? "#" + s.index : "BARS " + s.start_bar + "-" + s.end_bar
        + (frozen.has(s.index) ? " · FREEZE" : "");
      el.setAttribute("aria-label", "Seek to " + s.role + " " + s.index);
      el.addEventListener("click", (e) => { e.stopPropagation(); seek(a - start + 0.001); });
      secs.appendChild(el);
    });
    (sh.freezes || []).forEach((z) => {
      const a = z.start_frame / S.fps, b = (z.end_frame != null ? z.end_frame : z.start_frame + z.frames) / S.fps;
      const el = document.createElement("div");
      el.className = "fz"; el.style.left = (a / S.dur * 100) + "%"; el.style.width = ((b - a) / S.dur * 100) + "%";
      el.title = "Hook Freeze, bars " + z.start_bar + "-" + z.end_bar;
      el.textContent = "FREEZE";
      fz.appendChild(el);
    });
    (sh.shots || []).forEach((c) => {
      if (!c.start_frame) return;
      const el = document.createElement("div");
      el.className = "cut"; el.style.left = (c.start_frame / S.fps / S.dur * 100) + "%";
      el.title = "Cut on the downbeat of bar " + c.bar + " (" + c.cut_error_ms + " ms off)";
      cuts.appendChild(el);
    });
    // bar numbers under the rail, on the events: the first bar, each freeze, each cut, the last bar
    const spb = 60 / sh.bpm * sh.beats_per_bar;
    const marks = new Map();
    const mark = (bar) => { const songT = sh.start_s + (bar - sh.start_bar) * spb;
      if (songT >= start - 0.01 && songT <= end + 0.01) marks.set(bar, songT); };
    mark(sh.start_bar);
    (sh.freezes || []).forEach((z) => mark(z.start_bar));
    (sh.shots || []).forEach((c) => { if (c.start_frame) mark(c.bar); });
    mark(sh.end_bar);
    [...marks.entries()].sort((x, y) => x[0] - y[0]).forEach(([bar, songT], i, all) => {
      const sp = document.createElement("span");
      sp.style.left = at(songT) + "%";
      sp.textContent = "B" + bar;
      if (i === all.length - 1) sp.style.transform = "translateX(-100%)";
      bars.appendChild(sp);
    });
    $("strip-rail").setAttribute("aria-valuemax", S.dur.toFixed(2));
    const rail = $("strip-rail");
    let dragging = false;
    const fromX = (x) => { const r = rail.getBoundingClientRect(); return clamp((x - r.left) / r.width, 0, 1) * S.dur; };
    rail.addEventListener("pointerdown", (e) => {
      dragging = true; rail.setPointerCapture(e.pointerId); seek(fromX(e.clientX));
    });
    rail.addEventListener("pointermove", (e) => { if (dragging) seek(fromX(e.clientX)); });
    rail.addEventListener("pointerup", () => { dragging = false; });
    rail.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 1 / S.fps : 1;
      if (e.key === "ArrowRight") { e.preventDefault(); seek(A.currentTime + step); }
      if (e.key === "ArrowLeft") { e.preventDefault(); seek(A.currentTime - step); }
    });
  }

  function sectionNow(songT) {
    const secs = S.sections;
    const s = secs.find((x) => songT >= x.start_s && songT < x.end_s) || secs[secs.length - 1];
    const key = s ? s.index : null;
    document.querySelectorAll(".sec").forEach((el) => el.classList.toggle("now", String(el.dataset.index) === String(key)));
    showTreatment(songT, key);
  }

  // ------------------------------------------------------------------ treatment
  function showTreatment(songT, index) {
    const panel = $("director");
    if (!S.treatment) { panel.hidden = true; return; }
    const t = (S.treatment.sections || []).find((x) => x.index === index);
    if (!t) { panel.hidden = true; return; }
    panel.hidden = false;
    if (S.lastSec !== index) {
      S.lastSec = index;
      $("dir-section").textContent = "#" + t.index + " " + String(t.role).toUpperCase();
      const ol = $("dir-lyric"); ol.textContent = "";
      const lines = t.lyric || [];
      const heard = lyricTimes(index);
      lines.forEach((line) => {
        const li = document.createElement("li"); li.textContent = line;
        const h = heard.find((x) => x.text === line);
        if (h) { li.dataset.a = h.start_s; li.dataset.b = h.end_s; }
        ol.appendChild(li);
      });
      ol.parentElement.hidden = !lines.length;
      $("dir-subtext").textContent = t.subtext || "";
      $("dir-subtext").parentElement.hidden = !t.subtext;
      $("dir-world").textContent = t.world_prompt || "";
      $("dir-why").textContent = t.why ? "WHY: " + t.why : "";
      $("dir-world").parentElement.hidden = !t.world_prompt;
      const fp = t.guard_first_pass;
      const rej = $("dir-rejected");
      if (fp && fp.verdict === "reject" && fp.prompt) {
        rej.hidden = false;
        $("dir-rejected-text").textContent = fp.prompt;
        $("dir-rejected-label").textContent = "FIRST PASS, REJECTED AS LITERAL (" + Math.round(fp.score * 100)
          + "% OF ITS WORDS ARE THE LYRIC'S)";
      } else rej.hidden = true;
    }
    document.querySelectorAll("#dir-lyric li").forEach((li) => {
      const a = parseFloat(li.dataset.a), b = parseFloat(li.dataset.b);
      li.classList.toggle("now", !Number.isNaN(a) && songT >= a && songT <= b);
    });
  }

  function lyricTimes(index) {
    if (!S.lyrics) return [];
    const s = (S.lyrics.sections || []).find((x) => x.index === index);
    return (s && s.heard) || [];
  }

  function treatmentNote() {
    const tr = S.treatment, m = S.manifest;
    const fixture = !!(tr.FIXTURE || String(tr.mode || "").toUpperCase() === "DRY RUN");
    $("dir-fixture").hidden = !fixture;
    const notes = [];
    if (fixture) notes.push("THIS TREATMENT IS A FIXTURE: HAND-WRITTEN TEST RESPONSES, NOT MODEL OUTPUT.");
    const plates = (S.shots.world_sources || []).some((w) => String(w.from || "").includes("local plates"));
    if (plates) notes.push("THE WORLDS ON SCREEN ARE PROCEDURAL TEST PLATES, NOT THIS WORLD.");
    if (m.test && notes.length) { $("dir-note").textContent = notes.join(" "); $("dir-note").hidden = false; }
  }

  // ------------------------------------------------------------------ credits
  function money(x) {  // sub-dime costs keep a third decimal, so US$0.005 does not read as US$0.00
    if (x == null) return "n/a";
    const v = Number(x); return "US$" + v.toFixed(v > 0 && v < 0.1 ? 3 : 2);
  }
  function count(x) { return x == null ? "n/a" : String(x); }

  function buildCredits(c) {
    if (!c || !Array.isArray(c.models) || !c.models.length) return;
    const test = c.status !== "REAL";
    $("made-status").innerHTML = "";
    const b = document.createElement("b"); b.textContent = c.status; $("made-status").appendChild(b);
    $("made-status").appendChild(document.createTextNode(String(c.statement || "").toUpperCase()));
    const grid = $("made-models");
    c.models.forEach((m) => {
      const el = document.createElement("article"); el.className = "model";
      el.innerHTML = '<span class="label mono">META</span><h3></h3><p></p><dl></dl>';
      el.querySelector("h3").textContent = m.name;
      el.querySelector("p").textContent = m.did;
      const dl = el.querySelector("dl");
      [["RAN", m.ran ? "YES" : "NO"], ["CALLS", count(m.calls)], ["FRAMES", count(m.frames)], ["COST", money(m.usd)]]
        .forEach(([k, v]) => { const d = document.createElement("div");
          d.innerHTML = "<dt></dt><i></i><dd></dd>"; d.querySelector("dt").textContent = k;
          d.querySelector("dd").textContent = v; dl.appendChild(d); });
      if (!m.ran && m.stand_in) {
        const s = document.createElement("p"); s.className = "stand";
        s.innerHTML = "<b>STAND-IN</b> "; s.appendChild(document.createTextNode(m.stand_in)); el.appendChild(s);
      }
      if (m.ran && m.note) {
        const s = document.createElement("p"); s.className = "stand";
        s.innerHTML = "<b>NOTE</b> "; s.appendChild(document.createTextNode(m.note)); el.appendChild(s);
      }
      grid.appendChild(el);
    });
    const ul = $("made-code");
    (c.code || []).forEach((k) => { const li = document.createElement("li");
      li.innerHTML = "<b></b>"; li.querySelector("b").textContent = k.name;
      li.appendChild(document.createTextNode(k.did)); ul.appendChild(li); });
    $("made-total").textContent = "BILLED FOR THIS RENDER " + money(c.total_usd) + (test ? "  ·  TEST" : "");
    const sam = c.models.find((m) => m.name === "SAM 3.1");
    if (sam && sam.ran) {  // the receipt's colour region is SAM's output: say so where the numbers are
      $("slip-foot").textContent = "FILMED = PIXELS INSIDE SAM 3.1'S CUT-OUT OF THE PERFORMER, NEVER GENERATED. " +
        "GENERATED = EVERYTHING ELSE: WORLD, PARTICLES, PAPER. TOGGLE THE RECEIPT TO SEE THE CUT-OUT IN COLOUR.";
    }
    $("made").hidden = false;
  }

  // ------------------------------------------------------------------ the director's review of its own cut
  function mmss(t) { const s = Math.max(0, Number(t) || 0); return Math.floor(s / 60) + ":" + pad(Math.floor(s % 60), 2); }

  function buildReview(r) {
    if (!r || !Array.isArray(r.cuts) || !r.cuts.length) return;
    const first = r.cuts[0], last = r.cuts[r.cuts.length - 1];
    $("rev-model").textContent = (r.mode === "keyed" ? String(r.model || "MUSE WONDER") : "FIXTURE").toUpperCase();
    $("review-title").textContent = "Cut 1: " + (first.verdict === "revise" ? "revise." : "keep.");
    const ol = $("rev-notes");
    (first.notes || []).forEach((n) => {
      const li = document.createElement("li"); li.className = "sev-" + (n.severity || "low");
      li.innerHTML = '<span class="mono t"></span><span class="mono k"></span><span class="n"></span>';
      li.querySelector(".t").textContent = mmss(n.t_s);
      li.querySelector(".k").textContent = String(n.against || "").toUpperCase() + " · " +
        String(n.severity || "").toUpperCase();
      li.querySelector(".n").textContent = n.note;
      ol.appendChild(li);
    });
    const ch = (r.changes || [])[0];
    if (ch) {
      $("rev-change").hidden = false;
      const field = String(ch.field || "").replace("_", " ").toUpperCase();
      $("rev-old").textContent = field + ": " + ch.old;
      $("rev-new").textContent = field + ": " + ch.new;
      $("rev-why").textContent = ch.because ? "BECAUSE: " + ch.because : "";
      const box = $("rev-worlds"), stills = r.stills || [];
      stills.forEach((w, i) => {
        const f = document.createElement("figure");
        f.innerHTML = '<img alt="" loading="lazy" decoding="async"><figcaption class="mono"></figcaption>';
        f.querySelector("img").src = "data/" + w;
        f.querySelector("img").alt = (i ? "Cut 2, after the change" : "Cut 1, before the change") + " at " +
          mmss(r.still_t_s);
        f.querySelector("figcaption").textContent = (stills.length > 1 ? (i ? "CUT 2" : "CUT 1") : "CUT") +
          " AT " + mmss(r.still_t_s);
        box.appendChild(f);
      });
    }
    if (r.cuts.length > 1) {
      const top = (last.notes || []).slice().sort((a, b) => "highmidlow".indexOf(a.severity) -
                                                         "highmidlow".indexOf(b.severity))[0];
      $("rev-after").hidden = false;
      $("rev-after").textContent = "CUT 2, WATCHED AGAIN: " + String(last.verdict || "").toUpperCase() +
        (top ? " · " + mmss(top.t_s) + " " + top.note : "");
    }
    $("rev-foot").textContent = "EVERY NOTE ABOVE IS VERBATIM MODEL OUTPUT, NOT EDITED. THIS LOOP: " +
      (r.spend && r.spend.calls != null ? r.spend.calls + " PAID CALLS, " : "") +
      (r.spend && r.spend.usd != null ? money(r.spend.usd) : "") + ".";
    $("review").hidden = false;
  }

  // ------------------------------------------------------------------ boot
  async function boot() {
    const m = S.manifest = await getJSON("manifest.json");
    const [receipt, shots, credits, treatment, lyrics, timeline, review] = await Promise.all([
      getJSON(m.files.receipt), getJSON(m.files.shots), opt(m.files.credits), opt(m.files.treatment),
      opt(m.files.lyrics), opt(m.files.timeline), opt(m.files.review)]);
    Object.assign(S, { receipt, shots, treatment, lyrics, fps: m.fps, frames: m.frames, dur: m.duration_s,
                       start: m.start_s });
    // the song's own sections over the render's range (a bars:A-B plan holds one pseudo-section, index -1)
    const end = m.start_s + m.duration_s;
    S.sections = timeline && timeline.sections
      ? timeline.sections.filter((s) => s.end_s > m.start_s + 1e-3 && s.start_s < end - 1e-3)
      : (shots.sections || []).filter((s) => s.index >= 0);
    if (!S.sections.length) S.sections = shots.sections || [];
    if (m.test) { $("stage-tag").hidden = false; document.documentElement.dataset.test = "1"; }
    $("render-name").textContent = "RENDER " + m.render + (m.test ? " · TEST" : "");
    A.src = "data/" + m.videos.finished; B.src = "data/" + m.videos.receipt;
    A.poster = B.poster = "data/" + m.videos.poster;
    buildStrip();
    if (treatment) treatmentNote();
    buildCredits(credits);
    buildReview(review);
    render(true);
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ input
  stage.addEventListener("click", toggle);
  $("btn-play").addEventListener("click", toggle);
  $("btn-receipt").addEventListener("click", () => setReceipt(!S.on));
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.code === "Space" || e.key === " ") { e.preventDefault(); toggle(); }
    else if (e.key === "r" || e.key === "R") { e.preventDefault(); setReceipt(!S.on); }
  });
  reduced.addEventListener && reduced.addEventListener("change", () => render(true));

  // for the checks in scripts/ (and anyone curious): the two clocks, the frame, the figure on screen
  window.realone = {
    state: () => ({ a: A.currentTime, b: B.currentTime, frame: S.lastFrame, on: S.on, playing: S.playing,
                    filmed: $("r-filmed").textContent, section: S.lastSec, rate: B.playbackRate }),
    seek, setReceipt, start, stop,
  };

  boot().catch((err) => {
    const p = document.createElement("p"); p.className = "mono noscript";
    p.textContent = "The render data did not load (" + err.message + ").";
    document.body.insertBefore(p, document.body.firstChild);
  });
})();

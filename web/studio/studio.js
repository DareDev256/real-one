/* Real One Studio: a replay of the Nov 4 run, every value read from studio.json (written from the run's own records). */
(function () {
  "use strict";
  if (location.hash === "#edit") { location.replace("edit/"); return; }   // the editor lives at /studio/edit/
  var N = 7, cur = 1, D = null;
  var AUTO = /[?&]auto=1\b/.test(location.search);
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var $ = function (id) { return document.getElementById(id); };
  var NAMES = ["Takes", "Song", "Director", "Worlds", "Words", "Edit", "Video"];
  var MINUS = "\u2212";

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function pct(x) { return (Math.max(0, Math.min(1, x)) * 100).toFixed(3) + "%"; }
  function secName(s, all) {
    var hooks = all.filter(function (x) { return x.role === "hook"; });
    if (s.role === "hook" && hooks.length > 1 && hooks[hooks.length - 1].index === s.index) return "LAST HOOK";
    return s.role.toUpperCase();
  }
  function signed(x) { return (x < 0 ? MINUS : "+") + Math.abs(x).toFixed(2) + " S"; }
  function fact(label, value, small) {
    var r = el("div", "fact");
    r.appendChild(el("b", null, label));
    r.appendChild(el("i"));
    var v = el("span", "v", value);
    r.appendChild(v);
    if (small) r.appendChild(el("small", null, small));
    return r;
  }

  /* ---------- navigation */
  function buildRail() {
    var ol = $("rail");
    for (var i = 1; i <= N; i++) {
      var li = el("li"), b = el("button");
      b.type = "button"; b.dataset.n = i;
      b.setAttribute("aria-label", "Step " + i + ", " + NAMES[i - 1]);
      b.appendChild(el("b", null, ("0" + i).slice(-2)));
      b.appendChild(el("span", null, NAMES[i - 1]));
      b.addEventListener("click", function () { go(+this.dataset.n, true); });
      li.appendChild(b); ol.appendChild(li);
    }
  }
  function go(n, push) {
    n = Math.max(1, Math.min(N, n | 0));
    leave(cur);
    cur = n;
    for (var i = 1; i <= N; i++) $("step-" + i).classList.toggle("on", i === n);
    var bs = $("rail").querySelectorAll("button");
    bs.forEach(function (b) {
      var k = +b.dataset.n;
      b.classList.toggle("now", k === n);
      b.classList.toggle("done", k < n);
      if (k === n) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    $("back").disabled = n === 1;
    $("next").querySelector(".lbl").textContent = n === N ? "START OVER" : "NEXT";
    if (push !== false && location.hash !== "#step-" + n) history.replaceState(null, "", "#step-" + n);
    document.title = "Real One Studio \u00b7 " + NAMES[n - 1];
    enter(n);
    window.scrollTo(0, 0);
  }
  function fromHash() {
    var m = /step-(\d)/.exec(location.hash);
    return m ? +m[1] : 1;
  }
  function enter(n) {
    if (n === 1) document.querySelectorAll("#takes video").forEach(function (v) { v.play().catch(function () {}); });
    if (n === 2 && AUTO) toggleSong(true);
    if (n === 2) drawWave();
    if (n === 5) layoutWords();
    if (n === 5 && AUTO) setTimeout(function () { toggleLine(true); }, 500);
    if (n === 6 && AUTO) setTimeout(runEdit, 700);
    if (n === 7) ["fin-clean", "fin-receipt"].forEach(function (id) { var v = $(id); v.currentTime = 0; v.play().catch(function () {}); });
  }
  function leave(n) {
    if (n === 1) document.querySelectorAll("#takes video").forEach(function (v) { v.pause(); });
    if (n === 2) toggleSong(false);
    if (n === 5) toggleLine(false);
    if (n === 6) { var v = $("heat"); v.pause(); }
    if (n === 7) ["fin-clean", "fin-receipt"].forEach(function (id) { $(id).pause(); });
  }

  /* ---------- step 1: takes */
  function stepTakes() {
    var t = D.takes, wrap = $("takes"), facts = $("takes-facts");
    $("ro-1").textContent = "STEP 01 \u00b7 " + t.length + " TAKES \u00b7 4K HEVC \u00b7 60 FPS \u00b7 TRIPOD";
    t.forEach(function (k) {
      var f = el("figure", "take");
      var v = document.createElement("video");
      v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true; v.preload = "auto";
      v.setAttribute("muted", ""); v.setAttribute("playsinline", "");
      v.src = "media/take" + k.n + ".mp4";
      v.setAttribute("aria-label", "Take " + k.n);
      f.appendChild(v);
      var c = el("figcaption", "mono");
      c.appendChild(el("b", null, "TAKE " + k.n));
      c.appendChild(el("span", null, signed(k.offset_s)));
      f.appendChild(c);
      wrap.appendChild(f);
      facts.appendChild(fact("TAKE " + k.n, signed(k.offset_s), "LOCK " + k.r_peak.toFixed(2) + " \u00b7 MARGIN " + k.margin.toFixed(2)));
    });
    var note = el("p", "note");
    note.textContent = "WHERE EACH TAKE STARTS ON THE SONG. A TAKE IS ONLY PLACED IF IT CLEARLY BEATS EVERY OTHER POSITION; OTHERWISE IT REFUSES.";
    note.className = "note mono";
    facts.parentNode.appendChild(note);
    note.style.marginTop = "14px";
    // the song clock chart
    var dur = D.song.duration_s, cc = $("clockchart");
    var secs = el("div", "cc-row");
    secs.appendChild(el("span", null, "SONG"));
    var sb = el("div", "cc-secs");
    D.sections.forEach(function (s) {
      var d = el("div", "cc-sec", secName(s, D.sections));
      d.style.left = pct(s.start_s / dur); d.style.width = pct((s.end_s - s.start_s) / dur);
      sb.appendChild(d);
    });
    secs.appendChild(sb); cc.appendChild(secs);
    t.forEach(function (k) {
      var r = el("div", "cc-row");
      r.appendChild(el("span", null, "TAKE " + k.n));
      var tr = el("div", "cc-track"), bar = el("div", "cc-bar");
      var a = Math.max(0, k.song_start_s), b = Math.min(dur, k.song_end_s);
      bar.style.left = pct(a / dur); bar.style.width = pct((b - a) / dur);
      tr.appendChild(bar);
      var mk = el("div", "cc-mark"); mk.style.left = pct(9.5 / dur); tr.appendChild(mk);
      r.appendChild(tr); cc.appendChild(r);
    });
    var ax = el("div", "cc-row"), axs = el("div", "cc-axis");
    ax.appendChild(el("span"));
    for (var s = 0; s <= dur; s += 10) { var sp = el("span", null, s + " S"); sp.style.left = pct(s / dur); axs.appendChild(sp); }
    ax.appendChild(axs); cc.appendChild(ax);
  }

  /* ---------- step 2: song */
  var song = null, songRAF = 0;
  function stepSong() {
    var dur = D.song.duration_s, bars = D.sections.reduce(function (a, s) { return a + s.bars; }, 0);
    $("ro-2").textContent = "STEP 02 \u00b7 " + dur.toFixed(1) + " S \u00b7 " + D.sections.length + " SECTIONS \u00b7 " + bars + " BARS";
    $("song-credit").textContent = D.song.credit;
    var ws = $("wave-secs");
    D.sections.forEach(function (s) {
      var d = el("div", "wsec");
      d.dataset.i = s.index;
      d.style.left = pct(s.start_s / dur); d.style.width = pct((s.end_s - s.start_s) / dur);
      d.appendChild(el("b", null, secName(s, D.sections)));
      d.appendChild(el("small", null, s.start_s.toFixed(1) + " S \u00b7 " + s.bars + " BARS"));
      ws.appendChild(d);
    });
    var ax = $("wave-axis");
    for (var t = 0; t <= dur; t += 10) { var sp = el("span", null, t + " S"); sp.style.left = pct(t / dur); ax.appendChild(sp); }
    var bars = $("wave-bars"), bn = 0;
    D.sections.forEach(function (s) {
      var len = (s.end_s - s.start_s) / s.bars;
      for (var k = 0; k < s.bars; k++) {
        var t = s.start_s + k * len, tk = el("div", "bar-tick" + (k === 0 ? " sec" : (bn % 4 === 0 ? " down" : "")));
        tk.style.left = pct(t / dur);
        if (k === 0) tk.appendChild(el("span", null, secName(s, D.sections) + " " + (bn + 1)));
        else if (bn % 4 === 0) tk.appendChild(el("span", null, String(bn + 1)));
        bars.appendChild(tk); bn++;
      }
    });
    var bh = el("div", "playhead"); bh.id = "bar-head"; bars.appendChild(bh);
    song = new Audio("media/song.m4a"); song.preload = "none";
    song.addEventListener("ended", function () { toggleSong(false); });
    $("song-play").addEventListener("click", function () { toggleSong(song.paused); });
    window.addEventListener("resize", drawWave);
  }
  function drawWave() {
    var cv = $("wave-canvas"); if (!cv || !cv.clientWidth) return;
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    var g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    var p = D.song.peaks, n = p.length, bw = w / n, mid = h / 2;
    g.fillStyle = "#ffffff";
    for (var i = 0; i < n; i++) {
      var a = Math.max(1, p[i] * (h / 2 - 6));
      g.fillRect(i * bw, mid - a, Math.max(1, bw - 1), a * 2);
    }
  }
  function toggleSong(on) {
    if (!song) return;
    var b = $("song-play"), head = $("song-head");
    if (on) {
      song.play().then(function () {
        b.classList.add("playing"); b.querySelector(".lbl").textContent = "PAUSE"; head.classList.add("on"); $("bar-head").classList.add("on");
        tickSong();
      }).catch(function () {});
    } else {
      song.pause(); cancelAnimationFrame(songRAF);
      b.classList.remove("playing"); b.querySelector(".lbl").textContent = "PLAY THE SONG";
    }
  }
  function tickSong() {
    var t = song.currentTime, dur = D.song.duration_s;
    $("song-head").style.left = pct(t / dur); $("bar-head").style.left = pct(t / dur);
    document.querySelectorAll(".wsec").forEach(function (d) {
      var s = D.sections[+d.dataset.i];
      d.classList.toggle("hot", t >= s.start_s && t < s.end_s);
    });
    if (!song.paused) songRAF = requestAnimationFrame(tickSong);
  }

  /* ---------- step 3: director picks */
  function stepPicks() {
    var P = D.picks;
    $("ro-3").textContent = "STEP 03 \u00b7 MUSE WONDER \u00b7 WATCHED " + D.takes.length + " TAKES \u00b7 US$" + P.usd.toFixed(2);
    var wrap = $("picks");
    P.picks.forEach(function (p) {
      var s = D.sections[p.section], over = p.section === 1 && p.take === 1;
      var c = el("article", "pick" + (over ? " over" : ""));
      var h = el("div", "pick-head");
      h.appendChild(el("b", null, secName(s, D.sections)));
      h.appendChild(el("small", null, s.start_s.toFixed(1) + "\u2192" + s.end_s.toFixed(1) + " S"));
      c.appendChild(h);
      var im = el("div", "pick-img"), img = document.createElement("img");
      img.src = "media/pick_s" + p.section + "_t" + p.take + ".jpg"; img.alt = "Take " + p.take + " in the " + s.role;
      img.width = 360; img.height = 640;
      im.appendChild(img);
      im.appendChild(el("span", "pick-take", "TAKE " + p.take));
      if (over) im.appendChild(el("span", "stamp", "OVERRIDDEN"));
      c.appendChild(im);
      var why = el("p", "pick-why"), q = el("q", null, p.why);
      why.appendChild(q); c.appendChild(why);
      if (over) {
        var o = el("div", "over-to"), oi = document.createElement("img");
        oi.src = "media/pick_s1_t2.jpg"; oi.alt = "Take 2 in the first hook"; oi.width = 360; oi.height = 640;
        o.appendChild(oi);
        var t = el("p");
        t.appendChild(el("b", null, "The cut uses take 2. "));
        t.appendChild(document.createTextNode("Wonder picked take 1; a blind critic saw his head out of frame on the first hook, so it uses take 2."));
        o.appendChild(t); c.appendChild(o);
      }
      c.appendChild(el("p", "pick-src", "MUSE WONDER \u00b7 WATCHED ALL " + D.takes.length + " TAKES"));
      wrap.appendChild(c);
    });
  }

  /* ---------- step 4: worlds */
  function stepWorlds() {
    $("ro-4").textContent = "STEP 04 \u00b7 MUSE IMAGE \u00b7 " + D.worlds.length + " WORLDS \u00b7 1152 \u00d7 2048 EACH";
    var wrap = $("worlds");
    D.worlds.forEach(function (w, i) {
      var s = D.sections[w.section];
      var f = el("figure", "world"), im = el("div", "world-img"), img = document.createElement("img");
      img.src = "media/world" + i + ".jpg"; img.alt = "Muse Image world for the " + s.role; img.width = 540; img.height = 960;
      im.appendChild(img);
      var tag = el("div", "world-tag");
      tag.appendChild(el("b", null, secName(s, D.sections)));
      var mv = w.freeze ? "FREEZE + RECEIPT" : (w.move === "dolly" ? "DOLLY ZOOM" : "NO MOVE");
      tag.appendChild(el("span", w.move === "none" && !w.freeze ? "none" : "", mv));
      im.appendChild(tag);
      f.appendChild(im);
      var cap = el("figcaption");
      cap.appendChild(el("p", "world-why", w.why));
      cap.appendChild(el("p", "world-prompt", w.prompt.replace(/^Vertical 9:16,\s*/i, "")));
      f.appendChild(cap);
      wrap.appendChild(f);
    });
  }

  /* ---------- step 5: words */
  var line = null, lineRAF = 0, WIN = 3.5;
  function stepWords() {
    var W = D.words, ck = W.check;
    $("ro-5").textContent = "STEP 05 \u00b7 DEMUCS + MMS \u00b7 META OPEN WEIGHTS \u00b7 RUN ON THE MAC";
    $("words-num").textContent = ck.matched + " OF " + ck.of;
    $("words-sub").textContent = "WORDS LAND WITHIN A MEDIAN " + Math.round(ck.median_ms) + " MS OF AN INDEPENDENT TRANSCRIBER (WHISPER, USED ONLY AS A CHECK).";
    var ly = $("lyric"), strip = $("wordstrip");
    W.lines.forEach(function (L) {
      var ln = el("span", "ln");
      L.words.forEach(function (w, i) {
        var s = el("span", "w", w.w); s.dataset.a = w.start; s.dataset.b = w.end;
        ln.appendChild(s); if (i < L.words.length - 1) ln.appendChild(document.createTextNode(" "));
        var bx = el("div", "wbox"); bx.dataset.a = w.start; bx.dataset.b = w.end;
        bx.appendChild(el("span", null, w.w.replace(/[.,]$/, "")));
        strip.appendChild(bx);
      });
      ly.appendChild(ln);
    });
    var ax = $("wordaxis");
    for (var t = Math.ceil(W.clip_start_s); t <= W.clip_start_s + WIN; t += 1) {
      var sp = el("span", null, t.toFixed(0) + " S"); sp.style.left = pct((t - W.clip_start_s) / WIN); ax.appendChild(sp);
    }
    line = new Audio("media/line_hook.m4a"); line.preload = "none";
    line.addEventListener("ended", function () { toggleLine(false); markWords(W.clip_start_s + WIN); });
    $("words-play").addEventListener("click", function () { toggleLine(line.paused); });
    layoutWords();
  }
  function layoutWords() {
    var c0 = D.words.clip_start_s;
    document.querySelectorAll(".wbox").forEach(function (b) {
      b.style.left = pct((+b.dataset.a - c0) / WIN);
      b.style.width = pct((+b.dataset.b - +b.dataset.a) / WIN);
    });
  }
  function markWords(t) {
    document.querySelectorAll("#lyric .w, .wbox").forEach(function (e) {
      var a = +e.dataset.a, b = +e.dataset.b;
      e.classList.toggle("now", t >= a && t < b + 0.06);
      e.classList.toggle("past", t >= b + 0.06);
    });
    $("word-head").style.left = pct((t - D.words.clip_start_s) / WIN);
  }
  function toggleLine(on) {
    if (!line) return;
    var b = $("words-play"), head = $("word-head");
    if (on) {
      if (line.ended || line.currentTime >= WIN - 0.05) line.currentTime = 0;
      line.play().then(function () {
        b.classList.add("playing"); b.querySelector(".lbl").textContent = "PAUSE"; head.classList.add("on"); tickLine();
      }).catch(function () {});
    } else {
      line.pause(); cancelAnimationFrame(lineRAF);
      b.classList.remove("playing"); b.querySelector(".lbl").textContent = "PLAY THE LINE";
    }
  }
  function tickLine() {
    markWords(D.words.clip_start_s + line.currentTime);
    if (!line.paused) lineRAF = requestAnimationFrame(tickLine);
  }

  /* ---------- step 6: edit by sentence */
  var editTimers = [], heatTimer = 0;
  function norm(s) { return String(s).toLowerCase().replace(/[^a-z' ]/g, " ").replace(/\s+/g, " ").trim(); }
  function stepEdit() {
    var E = D.edit;
    $("ro-6").textContent = "STEP 06 \u00b7 MUSE WONDER + SAM 3.1 \u00b7 ONE SENTENCE \u00b7 US$" + E.usd.toFixed(3);
    $("say-input").value = E.said;
    $("say").addEventListener("submit", function (e) { e.preventDefault(); runEdit(); });
    var card = $("editcard");
    card.appendChild(el("p", "ec-empty", "PRESS ADD IT. THE ROWS BELOW ARE THE REAL EDIT RECORD FROM THE NOV 4 RUN."));
    var v = $("heat"), snd = $("heat-sound");
    snd.addEventListener("click", function () {
      v.muted = !v.muted; snd.setAttribute("aria-pressed", String(!v.muted)); snd.textContent = v.muted ? "SOUND OFF" : "SOUND ON";
    });
    v.addEventListener("timeupdate", heatCaption);
    v.addEventListener("ended", function () { $("heat-cap").textContent = "THE CUT AT \u201cHEAT\u201d \u00b7 PRESS ADD IT TO REPLAY"; });
  }
  function heatCaption() {
    var v = $("heat"), at = D.cut.heat_in_clip_s, cap = $("heat-cap"), fig = v.parentNode;
    var left = at - v.currentTime;
    if (left > 0.05) { cap.textContent = "\u201cHEAT\u201d IN " + left.toFixed(1) + " S"; fig.classList.remove("heat-flash"); }
    else { cap.textContent = ""; cap.appendChild(el("b", null, "\u201cHEAT\u201d")); cap.appendChild(document.createTextNode(" \u00b7 FIRE OFF HIS HANDS, TRACKED BY SAM 3.1")); fig.classList.add("heat-flash"); }
  }
  function row(dl, label, parts) {
    var r = el("div", "ec-row"), dt = el("dt", null, label), dd = el("dd");
    parts.forEach(function (p) { if (typeof p === "string") dd.appendChild(document.createTextNode(p)); else dd.appendChild(el("b", null, p.b)); });
    r.appendChild(dt); r.appendChild(dd); dl.appendChild(r); return r;
  }
  function model(card, name, what, mag) {
    var m = el("div", "ec-model");
    m.appendChild(el("span", "dot"));
    m.appendChild(el("span", "chip" + (mag ? " m" : ""), name));
    m.appendChild(el("span", null, what));
    card.appendChild(m); return m;
  }
  function stagger(nodes, gap) {
    nodes.forEach(function (n, i) {
      if (reduce) { n.classList.add("in"); return; }
      editTimers.push(setTimeout(function () { n.classList.add("in"); }, 120 + i * gap));
    });
    return reduce ? 0 : 120 + nodes.length * gap;
  }
  function runEdit() {
    editTimers.forEach(clearTimeout); editTimers = [];
    var E = D.edit, fx = E.fx[0], said = $("say-input").value.trim(), card = $("editcard"), v = $("heat");
    v.pause(); v.currentTime = 0; v.parentNode.classList.remove("heat-flash");
    card.textContent = "";
    var dl = el("dl");
    if (norm(said) === norm(E.said)) {
      var nodes = [];
      nodes.push(row(dl, "YOU SAID", ["\u201c" + E.said + "\u201d"]));
      card.appendChild(dl);
      nodes.push(model(card, "MUSE WONDER", "READS THE SENTENCE AND THE SONG'S SECTIONS", true));
      var dl2 = el("dl");
      nodes.push(row(dl2, "TRACK", [{ b: "\u201c" + fx.track + "\u201d" }]));
      nodes.push(row(dl2, "WORD", [{ b: "\u201c" + fx.word + "\u201d" }, " \u00b7 " + secName(D.sections[fx.section], D.sections)]));
      nodes.push(row(dl2, "LINE", ["\u201c" + E.line + "\u201d"]));
      nodes.push(row(dl2, "LANDS AT", [{ b: E.heat_word_start_s.toFixed(3) + " S" }, " on the song, from your word timings (", fx.at_s.toFixed(3) + " s with a 50 ms lead)"]));
      nodes.push(row(dl2, "EFFECT", [{ b: fx.effect.toUpperCase() }, " \u00b7 " + fx.seconds.toFixed(1) + " s"]));
      nodes.push(row(dl2, "WHY", ["\u201c" + E.why + "\u201d"]));
      nodes.push(row(dl2, "COST", ["US$" + E.usd.toFixed(3) + " \u00b7 1 call"]));
      card.appendChild(dl2);
      nodes.push(model(card, "SAM 3.1", "SEGMENTS \u201c" + fx.track.toUpperCase() + "\u201d ON EVERY FRAME", false));
      var done = stagger(nodes, 260);
      editTimers.push(setTimeout(function () {
        v.currentTime = 0; v.play().catch(function () {});
      }, done + 200));
    } else {
      var n2 = [];
      n2.push(row(dl, "YOU SAID", ["\u201c" + said + "\u201d"]));
      n2.push(row(dl, "TODAY", ["New sentences go through Muse Wonder from the command line. This page replays the real run and never makes up a model reply."]));
      card.appendChild(dl);
      var cli = el("div", "ec-row ec-cli"), dt = el("dt", null, "RUN"), dd = el("dd"), pre = el("pre", "mono");
      pre.textContent = "python -m realone.director edit \\\n  --treatment out/director_real_moments_treatment.json \\\n  --word-times footage/song_b/words_mms.json \\\n  --say \"" + said.replace(/["\\]/g, "\\$&") + "\" --out edits.json\n\npython -m realone render ... --edits edits.json";
      dd.appendChild(pre); cli.appendChild(dt); cli.appendChild(dd); card.appendChild(cli); n2.push(cli);
      var back = el("button", "btn", null); back.type = "button"; back.appendChild(el("span", "lbl", "REPLAY THE REAL ONE"));
      back.style.marginTop = "16px";
      back.addEventListener("click", function () { $("say-input").value = E.said; runEdit(); });
      card.appendChild(back);
      stagger(n2, 200);
    }
  }

  /* ---------- step 7 */
  function stepVideo() {
    $("ro-7").textContent = "STEP 07 \u00b7 49.7 S \u00b7 1080 \u00d7 1920 \u00b7 MASTER AUDIO IN SYNC";
    $("run-cmd").textContent = "# with MODEL_API_KEY set\npython -m realone.director plan --song song.wav \\\n  --lyrics lyrics.txt --tag mine\n\npython -m realone render --clip take.mov --song song.wav \\\n  --section all --stage tracked \\\n  --from-treatment out/director_mine_treatment.json \\\n  --look clean,receipt --receipt-counter --final --tag mine";
  }

  /* ---------- boot */
  function keys(e) {
    var tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "ArrowRight") { e.preventDefault(); go(cur === N ? 1 : cur + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(cur - 1); }
  }
  buildRail();
  $("back").addEventListener("click", function () { go(cur - 1); });
  $("next").addEventListener("click", function () { go(cur === N ? 1 : cur + 1); });
  document.addEventListener("keydown", keys);
  window.addEventListener("hashchange", function () { if (fromHash() !== cur) go(fromHash(), false); });
  fetch("studio.json", { cache: "no-cache" }).then(function (r) { return r.json(); }).then(function (d) {
    D = d;
    stepTakes(); stepSong(); stepPicks(); stepWorlds(); stepWords(); stepEdit(); stepVideo();
    go(fromHash(), false);
    document.documentElement.dataset.ready = "1";
  }).catch(function (e) {
    document.getElementById("steps").textContent = "The studio data did not load: " + e;
  });
})();

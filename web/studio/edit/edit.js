/* Real One Editor: a reel-app editor whose every control writes something the renderer reads.
   - Text, where it sits: edits.json "cards" [{line, place: world|top|bottom}] (fx.load_cards; render.py applies it
     over lyrictext.card_places: "world" is a sign on the floor's 3D track, a top or bottom card sits in front).
   - Effects: fire on a tapped sung word, tracking a part (edits.json "fx", realone/fx.py; at_s = start - 0.05,
     as realone/director/edit.py places it); particles: --particles.
   - Colour: the section's palette in the treatment (lyrictext.accent_of picks the accent).
   - World: --section-world '#N=PROMPT'.
   The preview is a real frame of the final cut with the card drawn in CSS; the render makes the real thing. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var D, T, sec = 2, line = 0, word = null, tool = null, sel = false;
  var colour = {}, prompts = {}, parts = "dust", fx = [], place = {}, track = "his hands", intensity = 1;
  var CAPTURE = /[?&]capture=1/.test(location.search);
  // where his hanging hand is in the heat clip (normalized x, y by clip time), measured from five frames of the
  // verse without the fire (draft 1, take 3, cut 26.8 to 29.6 s); the render follows SAM 3.1's mask of his hands
  var HAND = [[0.0, 0.26, 0.52], [0.7, 0.39, 0.54], [1.4, 0.35, 0.58], [2.1, 0.35, 0.56], [2.7, 0.44, 0.58]];
  var NAMES = {intro: "Intro", hook: "Hook", verse: "Verse"};
  var SEC_COL = ["#8A9BA8", "#F2A541", "#2EC4B6", "#D9D9DE"];
  var PLACES = [["auto", "Director's pick"], ["world", "In the world"], ["top", "Top card"], ["bottom", "Bottom card"]];
  var offs = [];

  function hsv(h) {
    h = h.replace("#", "");
    var r = parseInt(h.slice(0, 2), 16) / 255, g = parseInt(h.slice(2, 4), 16) / 255, b = parseInt(h.slice(4, 6), 16) / 255;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    return {s: mx ? (mx - mn) / mx : 0, v: mx};
  }
  function qualifies(h) { var c = hsv(h); return c.v >= 0.55 && c.s >= 0.35; }
  function role(s) { return s.index === 3 ? "Last hook" : (NAMES[s.role] || s.role); }
  function paletteFor(s, pick) {
    var keep = s.palette.filter(function (h) { return !qualifies(h); });
    var pal = (pick ? [pick] : []).concat(keep);
    ["#000000", "#FFFFFF"].forEach(function (h) { if (pal.length < 3 && pal.indexOf(h) < 0) pal.push(h); });
    return pal.slice(0, 5);
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function bare(w) { return w.replace(/[.,!?;:"]/g, ""); }
  function g() { return offs[sec] + line; }                      // the treatment's global line index
  function placeOf(k) { return place[k] || "auto"; }
  function mmss(s) { var m = Math.floor(s / 60), r = s - 60 * m; return m + ":" + (r < 10 ? "0" : "") + r.toFixed(1); }
  function curLine() { var ls = D.sections[sec].lines; return ls.length ? ls[Math.max(0, Math.min(ls.length - 1, line))] : null; }

  function setTool(t) {
    tool = (tool === t) ? null : t;
    document.querySelectorAll(".tools button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.tool === tool)); });
    $("sheet").hidden = !tool;
    document.querySelector(".phone").classList.toggle("open", !!tool);
    document.querySelector(".phone").classList.toggle("fx", tool === "fx" && sec === 2);
    ["text", "fx", "col", "world", "out"].forEach(function (k) { $("pane-" + k).hidden = (tool !== k); });
    if (tool === "text" || tool === "fx") sel = true;
    render();
  }

  function drawTimeline() {
    var box = $("secs"); box.textContent = "";
    D.sections.forEach(function (s) {
      var b = el("button", null, role(s)); b.type = "button"; b.style.background = SEC_COL[s.index];
      b.style.width = Math.max(64, (s.end_s - s.start_s) * 5.2) + "px";
      b.setAttribute("aria-pressed", String(s.index === sec));
      b.addEventListener("click", function () { sec = s.index; line = defaultLine(); word = null; render(); });
      box.appendChild(b);
    });
    var lb = $("lines"); lb.textContent = "";
    var ls = D.sections[sec].lines;
    if (!ls.length) lb.appendChild(el("span", "note", "No sung words here."));
    ls.forEach(function (l, i) {
      var b = el("button"); b.type = "button";
      var p = placeOf(offs[sec] + i);
      if (p !== "auto") b.appendChild(el("span", "pl", p === "world" ? "3D" : p === "top" ? "TOP" : "BOT"));
      b.appendChild(document.createTextNode(l.text));
      b.setAttribute("aria-pressed", String(i === line));
      b.addEventListener("click", function () { line = i; word = null; sel = true; if (!tool) setTool("text"); else render(); });
      lb.appendChild(b);
    });
    var on = lb.children[line];
    if (on && on.scrollIntoView) { lb.scrollLeft = Math.max(0, on.offsetLeft - lb.clientWidth * 0.4); }
    var fr = $("fxrow"); fr.textContent = "";
    var here = fx.filter(function (e) { return e.section === sec; });
    if (!here.length) fr.appendChild(el("span", "empty", "Effects track: empty"));
    here.forEach(function (e) { fr.appendChild(el("span", null, "🔥 fire on “" + e.word + "”")); });
  }
  function defaultLine() {
    var ls = D.sections[sec].lines;
    for (var i = 0; i < ls.length; i++) if (/heat/i.test(ls[i].text)) return i;
    return 0;
  }

  function clipOn() { return tool === "fx" && sec === 2; }
  function drawPreview() {
    var s = D.sections[sec];
    var v = $("clip"), on = clipOn();
    document.querySelector(".phone").classList.toggle("fx", on);
    if (on && !v.getAttribute("src")) v.src = "media/heat_nofire.mp4";
    v.hidden = !on; $("frame").style.visibility = on ? "hidden" : "visible";
    if (on && !CAPTURE && v.paused) v.play().catch(function () {});
    if (!on && !v.paused) v.pause();
    if ($("frame").getAttribute("src") !== s.frame) $("frame").src = s.frame;
    var c = colour[sec] !== undefined ? colour[sec] : s.accent;
    var card = $("card"); card.textContent = "";
    var l = curLine();
    var p = l ? placeOf(g()) : "auto";
    card.dataset.place = p === "auto" ? "bottom" : p;
    card.classList.toggle("sel", sel && !!l);
    $("trk").hidden = (p !== "world");
    if (l) {
      var ws = l.words, k = word ? ws.indexOf(word) : ws.length - 1;
      ws.slice(0, k + 1).forEach(function (w, i) { card.appendChild(el("span", i === k ? "now" : null, bare(w.w) + (i < k ? " " : ""))); });
      $("t-now").textContent = mmss((word || ws[ws.length - 1]).start);
    }
    card.style.setProperty("--c", c && qualifies(c) ? c : "#ffffff");
    $("t-sec").textContent = role(s);
    var here = fx.filter(function (e) { return e.section === sec; });
    $("fxtag").textContent = here.length ? "Fire on “" + here[here.length - 1].word + "”, tracking " + here[here.length - 1].track : "";
    $("hint").textContent = tool ? "" : "Tap a lyric to edit it";
  }

  function drawText() {
    var l = curLine(); $("line-big").textContent = l ? l.text : "No sung words in this section";
    var box = $("places"); box.textContent = "";
    PLACES.forEach(function (pp) {
      var b = el("button", null, pp[1]); b.type = "button";
      b.setAttribute("aria-pressed", String(placeOf(g()) === pp[0]));
      b.disabled = !l;
      b.addEventListener("click", function () { if (pp[0] === "auto") delete place[g()]; else place[g()] = pp[0]; render(); });
      box.appendChild(b);
    });
    var p = placeOf(g());
    $("place-note").textContent = p === "world" ? "A sign standing on the floor's 3D track: it moves with the camera and you can walk in front of it."
      : p === "top" ? "A card across the top, in front of everything." : p === "bottom" ? "A card in the lower third, in front of everything."
      : "The director decides: a card on close shots, a sign in the world on wide ones.";
  }
  function drawFx() {
    var l = curLine(), box = $("words"); box.textContent = "";
    if (l) l.words.forEach(function (w) {
      var b = el("button", null, bare(w.w)); b.type = "button";
      b.setAttribute("aria-pressed", String(word === w));
      b.addEventListener("click", function () { word = (word === w) ? null : w; render(); });
      box.appendChild(b);
    });
    var tb = $("tracks"); tb.textContent = "";
    ["his hands", "his shoes", "his hood"].forEach(function (t) {
      var b = el("button", null, "Track " + t); b.type = "button"; b.setAttribute("aria-pressed", String(track === t));
      b.addEventListener("click", function () { track = t; render(); });
      tb.appendChild(b);
    });
    $("add").disabled = !word;
    $("add").textContent = word ? "Add fire on “" + bare(word.w).toLowerCase() + "”" : "Tap a word to add fire";
    var ul = $("fxlist"); ul.textContent = "";
    fx.forEach(function (e, i) {
      var li = el("li"); li.appendChild(el("span", null, "🔥 Fire on “" + e.word + "”, " + e.track + ", " + e.intensity.toFixed(2) + "x"));
      var x = el("button", null, "Remove"); x.type = "button"; x.addEventListener("click", function () { fx.splice(i, 1); render(); });
      li.appendChild(x); ul.appendChild(li);
    });
    $("int-out").textContent = intensity.toFixed(1) + "x";
    var pb = $("parts"); pb.textContent = "";
    D.particles.forEach(function (p) {
      var b = el("button", null, p[0].toUpperCase() + p.slice(1)); b.type = "button"; b.setAttribute("aria-pressed", String(p === parts));
      b.addEventListener("click", function () { parts = p; startParticles(); render(); });
      pb.appendChild(b);
    });
  }
  function drawCol(keepPicker) {
    var s = D.sections[sec], box = $("swatches");
    var cur = colour[sec] !== undefined ? colour[sec] : s.accent;
    if (!keepPicker) {
      box.textContent = "";
      s.choices.concat([null]).forEach(function (h) {
        var b = el("button", "sw"); b.type = "button"; b.style.background = h || "#ffffff";
        b.setAttribute("aria-label", h ? "Lyric colour " + h : "White letters, no accent");
        b.setAttribute("aria-pressed", String((cur || null) === h));
        b.addEventListener("click", function () { colour[sec] = h; render(); });
        box.appendChild(b);
      });
      var any = el("label", "sw sw-any"); var inp = el("input"); inp.type = "color";
      inp.value = (cur && /^#[0-9a-f]{6}$/i.test(cur)) ? cur.toLowerCase() : "#f2a541";
      inp.setAttribute("aria-label", "Any colour");
      inp.addEventListener("input", function () { colour[sec] = inp.value.toUpperCase(); render(true); });
      any.appendChild(inp); any.appendChild(document.createTextNode("Any colour")); box.appendChild(any);
    }
    $("col-note").textContent = !cur ? "White letters: this section has no accent."
      : (!qualifies(cur) ? "Too dark or grey to read on the edge: the render keeps it white." : (colour[sec] === undefined ? "The director's pick." : "Your pick."));
  }
  function drawWorld() {
    var s = D.sections[sec], w = $("world");
    if (document.activeElement !== w) w.value = prompts[sec] !== undefined ? prompts[sec] : s.world_prompt.replace(/^Vertical 9:16, /, "");
    var changed = prompts[sec] !== undefined && prompts[sec] !== s.world_prompt.replace(/^Vertical 9:16, /, "");
    w.classList.toggle("changed", changed);
    $("world-note").textContent = changed ? "A new world: one Muse Image call, quoted and capped before it runs." : "Change it and Muse Image builds a new world for this section.";
  }

  function cards() {
    return Object.keys(place).map(Number).sort(function (a, b) { return a - b; }).map(function (k) {
      var s = 0; while (s + 1 < offs.length && offs[s + 1] <= k) s++;
      return {line: k, place: place[k], text: D.sections[s].lines[k - offs[s]].text};
    });
  }
  function edits() {
    var o = {fx: fx.map(function (e) { return {track: e.track, at_s: e.at_s, effect: "fire", seconds: e.seconds,
      intensity: Math.round(e.intensity * 100) / 100, word: e.word, section: e.section}; })};
    var c = cards(); if (c.length) o.cards = c;
    o.said = "made in the Real One editor";
    return o;
  }
  function treatment() {
    var t = JSON.parse(JSON.stringify(T));
    t.sections.forEach(function (s) { if (colour[s.index] !== undefined) s.palette = paletteFor(D.sections[s.index], colour[s.index]); });
    return t;
  }
  function changedColour() { return Object.keys(colour).some(function (k) { return colour[k] !== D.sections[k].accent; }); }
  function worldChanged(k) { return prompts[k] !== undefined && prompts[k] !== D.sections[k].world_prompt.replace(/^Vertical 9:16, /, ""); }
  function q(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }
  function command() {
    var s = D.sections[sec];
    var a = ["python -m realone render", "  --clip footage/own/take_t" + s.take + ".mov --song footage/song_b/real_one.wav",
      "  --sections footage/song_b/sections.txt --section '#" + sec + "'",
      "  --stage tracked --camera-m 0.8 --word-times footage/song_b/words_mms.json",
      "  --from-treatment " + (changedColour() ? "treatment.json" : "out/director_real_moments_treatment.json"),
      "  --hits --moments --look clean --final --particles " + parts];
    if (fx.length || Object.keys(place).length) a.push("  --edits edits.json");
    Object.keys(prompts).forEach(function (k) { if (worldChanged(k)) a.push("  --section-world " + q("#" + k + "=Vertical 9:16, " + prompts[k])); });
    return a.join(" \\\n");
  }
  function status() {
    var bits = [];
    var c = cards(); if (c.length) bits.push(c.length + " lyric line" + (c.length > 1 ? "s" : "") + " placed (" + c.map(function (x) { return x.place; }).join(", ") + ")");
    if (fx.length) bits.push("fire on “" + fx.map(function (e) { return e.word; }).join("”, “") + "”, tracking " + fx[fx.length - 1].track);
    if (parts !== D.particles_default) bits.push(parts + " in the air");
    if (changedColour()) bits.push("new lyric colour");
    if (Object.keys(prompts).some(worldChanged)) bits.push("a new world");
    return bits.length ? bits.join(" · ") : "Nothing changed yet: the director's cut.";
  }

  function render(keepPicker) {
    drawTimeline(); drawPreview(); drawText(); drawFx(); drawCol(keepPicker); drawWorld();
    $("cmd").textContent = command();
    $("mini").textContent = command();
    $("status").textContent = status();
    $("dl-edits").disabled = !(fx.length || Object.keys(place).length);
    $("dl-treat").disabled = !changedColour();
  }
  function download(name, obj) {
    var blob = new Blob([JSON.stringify(obj, null, 1) + "\n"], {type: "application/json"});
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  function flash(b, text) { var t = b.textContent; b.textContent = text; b.classList.add("done"); setTimeout(function () { b.textContent = t; b.classList.remove("done"); }, 1400); }

  function setIntensity(x) {
    intensity = Math.max(0.25, Math.min(2, x)); $("intensity").value = String(intensity);
    var here = fx.filter(function (e) { return e.section === sec; });
    if (here.length) here[here.length - 1].intensity = intensity;   // the slider edits the selected fire
    $("int-out").textContent = intensity.toFixed(1) + "x";
    $("cmd").textContent = command(); $("mini").textContent = command(); $("status").textContent = status();
    var ul = $("fxlist"); if (ul.lastChild && here.length) ul.lastChild.firstChild.textContent = "🔥 Fire on “" + here[here.length - 1].word + "”, " + here[here.length - 1].track + ", " + intensity.toFixed(2) + "x";
  }
  function handAt(t) {
    var a = HAND[0], b = HAND[HAND.length - 1];
    for (var i = 0; i + 1 < HAND.length; i++) if (t >= HAND[i][0] && t <= HAND[i + 1][0]) { a = HAND[i]; b = HAND[i + 1]; break; }
    var k = b[0] > a[0] ? (t - a[0]) / (b[0] - a[0]) : 0;
    return [a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];
  }
  // fx.burst: FIRE_EMBERS 700 at FIRE_SIZE 1.7, both times the intensity; here a preview of it, emitted per frame
  var embers = [], fcv, fcx, rng = 1;
  function rnd() { rng = (rng * 16807) % 2147483647; return (rng - 1) / 2147483646; }
  function fireStep(t, dt) {
    fcv = fcv || $("fire-canvas"); fcx = fcx || fcv.getContext("2d");
    var r = fcv.getBoundingClientRect(), W = Math.max(1, r.width | 0), H = Math.max(1, r.height | 0);
    if (fcv.width !== W || fcv.height !== H) { fcv.width = W; fcv.height = H; }
    fcx.clearRect(0, 0, W, H);
    var on = clipOn() && fx.some(function (e) { return e.section === 2; });
    if (!on) { embers = []; return; }
    // the hand in canvas px through the clip's object-fit: cover (object-position 50% 62%)
    var v = $("clip"), vw = v.videoWidth || 540, vh = v.videoHeight || 960, sc = Math.max(W / vw, H / vh);
    var dw = vw * sc, dh = vh * sc, ox = (W - dw) * 0.5, oy = (H - dh) * 0.62;
    var g = intensity, hp = handAt(t), hx = ox + hp[0] * dw, hy = oy + hp[1] * dh;
    W = dw; H = dh;   // ember spread and speed scale with the clip as shown
    var rate = 700 * g * 0.9;                     // embers a second at this intensity
    var n = Math.round(rate * dt + rnd());
    for (var i = 0; i < n; i++) embers.push({x: hx + (rnd() - 0.5) * W * 0.07 * (0.6 + g * 0.4), y: hy + (rnd() - 0.5) * H * 0.03,
      vx: (rnd() - 0.5) * W * 0.25 * g, vy: -(0.25 + rnd() * 0.6) * H * (0.5 + 0.5 * g), life: 0.35 + rnd() * 0.55, age: 0, s: (0.8 + rnd() * 1.6) * g * 1.7});
    // the glow the fire throws (fx.part_glow), wider and brighter with the intensity
    var gr = fcx.createRadialGradient(hx, hy, 0, hx, hy, W * 0.16 * (0.6 + 0.5 * g));
    gr.addColorStop(0, "rgba(255,190,90," + Math.min(0.75, 0.32 * g + 0.12) + ")"); gr.addColorStop(1, "rgba(255,120,30,0)");
    fcx.fillStyle = gr; fcx.fillRect(0, 0, fcv.width, fcv.height);
    fcx.globalCompositeOperation = "lighter";
    embers = embers.filter(function (e) { e.age += dt; return e.age < e.life; });
    embers.forEach(function (e) {
      e.x += e.vx * dt; e.y += e.vy * dt; e.vy *= 0.97; e.vx *= 0.97;
      var u = e.age / e.life, gg = Math.round(240 - 170 * u), bb = Math.round(150 * (1 - u) * (1 - u));
      fcx.fillStyle = "rgba(255," + gg + "," + bb + "," + (1 - u).toFixed(3) + ")";
      fcx.beginPath(); fcx.arc(e.x, e.y, Math.max(0.6, e.s * (1 - 0.5 * u)), 0, 6.283); fcx.fill();
    });
    // the core: a white-hot flame at the hand
    var core = fcx.createRadialGradient(hx, hy, 0, hx, hy, W * 0.045 * (0.5 + 0.6 * g));
    core.addColorStop(0, "rgba(255,255,230,0.95)"); core.addColorStop(1, "rgba(255,170,60,0)");
    fcx.fillStyle = core; fcx.beginPath(); fcx.arc(hx, hy, W * 0.045 * (0.5 + 0.6 * g), 0, 6.283); fcx.fill();
    fcx.globalCompositeOperation = "source-over";
  }
  var last = 0;
  function fireLoop() {
    (function tick(ts) {
      var v = $("clip"), dt = last ? Math.min(0.05, (ts - last) / 1000) : 1 / 60; last = ts;
      fireStep(v && !v.hidden ? v.currentTime : 0, dt);
      requestAnimationFrame(tick);
    })(0);
  }
  // capture mode (?capture=1): one deterministic frame at clip time t and slider x, for a frame-exact recording
  function captureFrame(t, x, dt) {
    var v = $("clip"); setIntensity(x);
    return new Promise(function (res) {
      var done = function () { fireStep(t, dt || 1 / 30); requestAnimationFrame(function () { res(true); }); };
      if (Math.abs(v.currentTime - t) < 0.001) return done();
      v.addEventListener("seeked", done, {once: true}); v.currentTime = t;
    });
  }

  // particles over the preview: the four kinds the renderer draws, as light dots (a preview, not the render)
  var cv, cx, dots = [], raf = 0;
  function startParticles() {
    cv = $("parts-canvas"); cx = cv.getContext("2d");
    var r = cv.getBoundingClientRect(); cv.width = Math.max(1, r.width | 0); cv.height = Math.max(1, r.height | 0);
    dots = []; cancelAnimationFrame(raf);
    if (parts === "none") { cx.clearRect(0, 0, cv.width, cv.height); return; }
    var n = parts === "rain" ? 140 : parts === "embers" ? 80 : 70;
    for (var i = 0; i < n; i++) dots.push({x: Math.random() * cv.width, y: Math.random() * cv.height, z: 0.4 + Math.random() * 0.8, p: Math.random() * 6.28});
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    (function tick() {
      cx.clearRect(0, 0, cv.width, cv.height);
      dots.forEach(function (d) {
        if (!reduce) {
          if (parts === "rain") { d.y += 9 * d.z; d.x += 1.2 * d.z; }
          else if (parts === "embers") { d.y -= 1.2 * d.z; d.p += 0.05; d.x += Math.sin(d.p) * 0.6; }
          else { d.y += 0.25 * d.z; d.p += 0.02; d.x += Math.sin(d.p) * 0.3; }
          if (d.y > cv.height + 10) d.y = -10; if (d.y < -10) d.y = cv.height + 10; if (d.x > cv.width + 10) d.x = -10;
        }
        if (parts === "rain") { cx.strokeStyle = "rgba(220,230,255," + (0.25 + 0.3 * d.z) + ")"; cx.lineWidth = 1;
          cx.beginPath(); cx.moveTo(d.x, d.y); cx.lineTo(d.x - 2.5 * d.z, d.y - 14 * d.z); cx.stroke(); }
        else if (parts === "embers") { cx.fillStyle = "rgba(255," + (120 + 60 * d.z | 0) + ",40," + (0.55 + 0.4 * d.z) + ")";
          cx.beginPath(); cx.arc(d.x, d.y, 1.6 + 1.8 * d.z, 0, 6.28); cx.fill(); }
        else { cx.fillStyle = "rgba(255,255,255," + (0.35 + 0.4 * d.z) + ")"; cx.beginPath(); cx.arc(d.x, d.y, 1 + 1.4 * d.z, 0, 6.28); cx.fill(); }
      });
      if (!reduce) raf = requestAnimationFrame(tick);
    })();
  }

  Promise.all([fetch("edit.json").then(function (r) { return r.json(); }), fetch("treatment.json").then(function (r) { return r.json(); })])
    .then(function (v) {
      D = v[0]; T = v[1];
      var o = 0; offs = D.sections.map(function (s) { var x = o; o += s.lines.length; return x; });
      line = defaultLine();
      document.querySelectorAll(".tools button").forEach(function (b) { b.addEventListener("click", function () { setTool(b.dataset.tool); }); });
      $("export-top").addEventListener("click", function () { if (tool !== "out") setTool("out"); });
      $("card").addEventListener("click", function () { sel = true; if (tool !== "text") setTool("text"); else render(); });
      $("card").addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("card").click(); } });
      $("seconds").addEventListener("input", function () { $("sec-out").textContent = (+$("seconds").value).toFixed(1) + " s"; });
      $("add").addEventListener("click", function () {
        if (!word) return;
        fx.push({track: track, at_s: Math.round((word.start - D.lead_s) * 1000) / 1000, effect: "fire", intensity: intensity,
                 seconds: Math.min(D.seconds[1], Math.max(D.seconds[0], +$("seconds").value)), word: bare(word.w).toLowerCase(), section: sec});
        word = null; render();
      });
      $("world").addEventListener("input", function () { prompts[sec] = $("world").value; render(); });
      $("copy").addEventListener("click", function () {
        var txt = $("cmd").textContent, b = this;
        (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () { flash(b, "Copied"); }, function () {
          var r = document.createRange(); r.selectNodeContents($("cmd")); var s = getSelection(); s.removeAllRanges(); s.addRange(r); flash(b, "Selected: press copy");
        });
      });
      $("dl-edits").addEventListener("click", function () { download("edits.json", edits()); flash(this, "Downloaded"); });
      $("dl-treat").addEventListener("click", function () { download("treatment.json", treatment()); flash(this, "Downloaded"); });
      $("intensity").addEventListener("input", function () { setIntensity(+$("intensity").value); });
      window.__editor = {edits: edits, treatment: treatment, command: command, setTool: setTool, frame: captureFrame,
                         setIntensity: setIntensity};
      render(); startParticles(); if (!CAPTURE) fireLoop();
      window.addEventListener("resize", startParticles);
    })
    .catch(function (e) { document.querySelector(".phone").textContent = "The editor data did not load: " + e; });
})();

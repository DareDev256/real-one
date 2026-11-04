/* Real One Editor. Every control writes something the renderer reads:
   - lyric colour: the section's palette in the treatment (render.py reads it; lyrictext.accent_of picks the accent),
   - particles: --particles, a world: --section-world '#N=PROMPT',
   - a tracked effect: the --edits file, in realone/fx.py's format, timed from the word times (at_s = start - 0.05,
     as realone/director/edit.py places it).
   The preview is a real frame of the final cut with the lyric card drawn in CSS; the render makes the real thing. */
(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var D, T, sec = 2, line = 0, word = null, colour = {}, prompts = {}, parts = "dust", fx = [];
  var NAMES = {intro: "Intro", hook: "Hook", verse: "Verse"};

  function hsv(h) {
    h = h.replace("#", "");
    var r = parseInt(h.slice(0, 2), 16) / 255, g = parseInt(h.slice(2, 4), 16) / 255, b = parseInt(h.slice(4, 6), 16) / 255;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    return {s: mx ? (mx - mn) / mx : 0, v: mx};
  }
  // lyrictext.accent_of: the most saturated colour bright enough to read on the dark edge (v >= 0.55, s >= 0.35)
  function qualifies(h) { var c = hsv(h); return c.v >= 0.55 && c.s >= 0.35; }
  function accentOf(pal) {
    var best = null, sc = 0;
    pal.forEach(function (h) { var c = hsv(h); if (c.v >= 0.55 && c.s >= 0.35 && c.s * c.v > sc) { best = h; sc = c.s * c.v; } });
    return best;
  }
  function role(s) { return s.index === 3 ? "Last hook" : (NAMES[s.role] || s.role); }
  // the palette that makes accent_of return the pick (or no accent, so the letters stay white): the pick, then the
  // director's colours that cannot be an accent; padded with black and white to the schema's 3 to 5 colours
  function paletteFor(s, pick) {
    var keep = s.palette.filter(function (h) { return !qualifies(h); });
    var pal = (pick ? [pick] : []).concat(keep);
    ["#000000", "#FFFFFF"].forEach(function (h) { if (pal.length < 3 && pal.indexOf(h) < 0) pal.push(h); });
    return pal.slice(0, 5);
  }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function bare(w) { return w.replace(/[.,!?;:"]/g, ""); }

  function drawSecs() {
    var box = $("secs"); box.textContent = "";
    D.sections.forEach(function (s) {
      var b = el("button", null, role(s)); b.type = "button";
      b.setAttribute("aria-pressed", String(s.index === sec));
      b.addEventListener("click", function () { sec = s.index; line = defaultLine(); word = null; render(); });
      box.appendChild(b);
    });
  }
  function drawSwatches() {
    var s = D.sections[sec], box = $("swatches"); box.textContent = "";
    var cur = colour[sec] !== undefined ? colour[sec] : s.accent;
    s.choices.concat([null]).forEach(function (h) {
      var b = el("button", "sw"); b.type = "button";
      b.style.background = h || "#ffffff";
      b.setAttribute("aria-label", h ? "Lyric colour " + h : "White letters, no accent");
      b.setAttribute("aria-pressed", String((cur || null) === h));
      b.addEventListener("click", function () { colour[sec] = h; render(); });
      box.appendChild(b);
    });
    var any = el("label", "sw sw-any"); var inp = el("input"); inp.type = "color";
    inp.value = (cur && /^#[0-9a-f]{6}$/i.test(cur)) ? cur.toLowerCase() : "#f2a541";
    inp.setAttribute("aria-label", "Any colour");
    inp.addEventListener("input", function () { colour[sec] = inp.value.toUpperCase(); render(true); });
    any.appendChild(inp); any.appendChild(document.createTextNode("Any colour"));
    box.appendChild(any);
    var custom = colour[sec] && s.palette.indexOf(colour[sec]) < 0;
    $("col-note").textContent = !cur ? "white: no accent in this section"
      : (custom && !qualifies(colour[sec]) ? "too dark or grey to read on the edge: the render keeps it white"
      : (colour[sec] === undefined ? "the director's pick" : "your pick"));
  }
  function drawParts() {
    var box = $("parts"); box.textContent = "";
    D.particles.forEach(function (p) {
      var b = el("button", null, p); b.type = "button"; b.setAttribute("aria-pressed", String(p === parts));
      b.addEventListener("click", function () { parts = p; startParticles(); render(); });
      box.appendChild(b);
    });
  }
  function defaultLine() {
    var ls = D.sections[sec].lines;
    for (var i = 0; i < ls.length; i++) if (/heat/i.test(ls[i].text)) return i;
    return 0;
  }
  function drawWords() {
    var s = D.sections[sec], ls = s.lines, box = $("words"); box.textContent = "";
    if (!ls.length) { box.appendChild(el("span", "soft", "No sung words in this section.")); return; }
    line = Math.max(0, Math.min(ls.length - 1, line));
    ls[line].words.forEach(function (w) {
      var b = el("button", null, bare(w.w)); b.type = "button";
      b.setAttribute("aria-pressed", String(word === w));
      b.title = "sung at " + w.start.toFixed(2) + " s";
      b.addEventListener("click", function () { word = (word === w) ? null : w; render(); });
      box.appendChild(b);
    });
    $("line-prev").disabled = line === 0; $("line-next").disabled = line >= ls.length - 1;
    $("add").disabled = !word || !$("track").value.trim();
    $("add").textContent = word ? "Add fire on “" + bare(word.w).toLowerCase() + "”" : "Add fire";
  }
  function drawFx() {
    var ul = $("fxlist"); ul.textContent = "";
    fx.forEach(function (e, i) {
      var li = el("li"); var t = el("span");
      t.innerHTML = "";
      t.appendChild(document.createTextNode("Fire on "));
      t.appendChild(el("b", null, "“" + e.word + "”"));
      t.appendChild(document.createTextNode(", tracking " + e.track + ", at " + e.at_s.toFixed(3) + " s for " + e.seconds + " s"));
      var x = el("button", null, "remove"); x.type = "button";
      x.addEventListener("click", function () { fx.splice(i, 1); render(); });
      li.appendChild(t); li.appendChild(x); ul.appendChild(li);
    });
  }
  function drawPreview() {
    var s = D.sections[sec];
    if ($("frame").getAttribute("src") !== s.frame) $("frame").src = s.frame;
    var c = colour[sec] !== undefined ? colour[sec] : s.accent;
    var show = c && qualifies(c) ? c : null;
    var card = $("card"); card.textContent = "";
    var ls = s.lines; if (!ls.length) return;
    var ws = ls[line].words, k = word ? ws.indexOf(word) : ws.length - 1;
    // the card as the render draws it: the line so far, the newest (sung) word in the section's accent
    ws.slice(0, k + 1).forEach(function (w, i) {
      var sp = el("span", i === k ? "now" : null, bare(w.w) + (i < k ? " " : ""));
      card.appendChild(sp);
    });
    card.style.setProperty("--c", show || "#ffffff");
    var here = fx.filter(function (e) { return e.section === sec; });
    $("fxtag").textContent = here.length ? here.map(function (e) { return "fire on “" + e.word + "”, " + e.track; }).join(" · ") : "";
  }
  function edits() {
    return {fx: fx.map(function (e) { return {track: e.track, at_s: e.at_s, effect: "fire", seconds: e.seconds, word: e.word, section: e.section}; }),
            said: "made in the Real One editor"};
  }
  function treatment() {
    var t = JSON.parse(JSON.stringify(T));
    t.sections.forEach(function (s) {
      var d = D.sections[s.index];
      if (colour[s.index] !== undefined) s.palette = paletteFor(d, colour[s.index]);
    });
    return t;
  }
  function changedColour() { return Object.keys(colour).some(function (k) { return colour[k] !== D.sections[k].accent; }); }
  function q(s) { return "'" + String(s).replace(/'/g, "'\\''") + "'"; }
  function command() {
    var s = D.sections[sec];
    var a = ["python -m realone render", "  --clip footage/own/take_t" + s.take + ".mov --song footage/song_b/real_one.wav",
      "  --sections footage/song_b/sections.txt --section '#" + sec + "'",
      "  --stage tracked --camera-m 0.8 --word-times footage/song_b/words_mms.json",
      "  --from-treatment " + (changedColour() ? "treatment.json" : "out/director_real_moments_treatment.json"),
      "  --hits --moments --look clean --final --particles " + parts];
    if (fx.length) a.push("  --edits edits.json");
    Object.keys(prompts).forEach(function (k) {
      if (prompts[k] && prompts[k] !== D.sections[k].world_prompt) a.push("  --section-world " + q("#" + k + "=" + prompts[k]));
    });
    return a.join(" \\\n");
  }
  function render(keepPicker) {
    drawSecs(); if (!keepPicker) drawSwatches(); else drawSwatchesNote(); drawParts(); drawWords(); drawFx(); drawPreview();
    var s = D.sections[sec], w = $("world");
    if (document.activeElement !== w) w.value = prompts[sec] !== undefined ? prompts[sec] : s.world_prompt;
    w.classList.toggle("changed", prompts[sec] !== undefined && prompts[sec] !== s.world_prompt);
    $("world-note").textContent = w.classList.contains("changed") ? "a new world: one Muse Image call, quoted and capped before it runs"
      : "the director's prompt; change it for a new Muse Image world";
    $("cmd").textContent = command();
    $("dl-edits").disabled = !fx.length;
    $("dl-treat").disabled = !changedColour();
  }
  function drawSwatchesNote() {
    var s = D.sections[sec], cur = colour[sec];
    document.querySelectorAll("#swatches .sw:not(.sw-any)").forEach(function (b) { b.setAttribute("aria-pressed", "false"); });
    $("col-note").textContent = cur && !qualifies(cur) ? "too dark or grey to read on the edge: the render keeps it white" : "your pick";
  }
  function download(name, obj) {
    var blob = new Blob([JSON.stringify(obj, null, 1) + "\n"], {type: "application/json"});
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }
  function flash(b, text) { var t = b.textContent; b.textContent = text; b.classList.add("done"); setTimeout(function () { b.textContent = t; b.classList.remove("done"); }, 1400); }

  // particles over the preview: the four kinds the renderer draws, as light dots (a preview, not the render)
  var cv, cx, dots = [], raf = 0;
  function startParticles() {
    cv = $("parts-canvas"); cx = cv.getContext("2d");
    var r = cv.getBoundingClientRect(); cv.width = Math.max(1, r.width | 0); cv.height = Math.max(1, r.height | 0);
    dots = []; if (parts === "none") { cx.clearRect(0, 0, cv.width, cv.height); cancelAnimationFrame(raf); return; }
    var n = parts === "rain" ? 140 : parts === "embers" ? 70 : 90;
    for (var i = 0; i < n; i++) dots.push({x: Math.random() * cv.width, y: Math.random() * cv.height, z: 0.4 + Math.random() * 0.8, p: Math.random() * 6.28});
    cancelAnimationFrame(raf);
    var reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
    (function tick() {
      cx.clearRect(0, 0, cv.width, cv.height);
      dots.forEach(function (d) {
        if (!reduce) {
          if (parts === "rain") { d.y += 9 * d.z; d.x += 1.2 * d.z; }
          else if (parts === "embers") { d.y -= 1.1 * d.z; d.p += 0.05; d.x += Math.sin(d.p) * 0.5; }
          else { d.y += 0.25 * d.z; d.p += 0.02; d.x += Math.sin(d.p) * 0.3; }
          if (d.y > cv.height + 10) d.y = -10; if (d.y < -10) d.y = cv.height + 10; if (d.x > cv.width + 10) d.x = -10;
        }
        if (parts === "rain") { cx.strokeStyle = "rgba(220,230,255," + (0.25 + 0.3 * d.z) + ")"; cx.lineWidth = 1;
          cx.beginPath(); cx.moveTo(d.x, d.y); cx.lineTo(d.x - 2.5 * d.z, d.y - 14 * d.z); cx.stroke(); }
        else if (parts === "embers") { cx.fillStyle = "rgba(255," + (120 + 60 * d.z | 0) + ",40," + (0.5 + 0.4 * d.z) + ")";
          cx.beginPath(); cx.arc(d.x, d.y, 1.4 + 1.6 * d.z, 0, 6.28); cx.fill(); }
        else { cx.fillStyle = "rgba(255,255,255," + (0.35 + 0.4 * d.z) + ")"; cx.beginPath(); cx.arc(d.x, d.y, 1 + 1.4 * d.z, 0, 6.28); cx.fill(); }
      });
      if (!reduce) raf = requestAnimationFrame(tick);
    })();
  }

  Promise.all([fetch("edit.json").then(function (r) { return r.json(); }), fetch("treatment.json").then(function (r) { return r.json(); })])
    .then(function (v) {
      D = v[0]; T = v[1]; line = defaultLine();
      $("line-prev").addEventListener("click", function () { line--; word = null; render(); });
      $("line-next").addEventListener("click", function () { line++; word = null; render(); });
      ["his hands", "his shoes", "his hood"].forEach(function (p) {
        var b = el("button", null, p); b.type = "button";
        b.addEventListener("click", function () { $("track").value = p; render(); });
        $("chips").appendChild(b);
      });
      $("track").addEventListener("input", function () { $("add").disabled = !word || !$("track").value.trim(); });
      $("seconds").addEventListener("input", function () { $("sec-out").textContent = (+$("seconds").value).toFixed(1) + " s"; });
      $("add").addEventListener("click", function () {
        var tr = $("track").value.trim().toLowerCase();
        if (!word || !tr || tr.split(/\s+/).length > 5) return;   // director/edit.py: a short phrase, at most 5 words
        fx.push({track: tr, at_s: Math.round((word.start - D.lead_s) * 1000) / 1000, effect: "fire",
                 seconds: Math.min(D.seconds[1], Math.max(D.seconds[0], +$("seconds").value)), word: bare(word.w).toLowerCase(), section: sec});
        word = null; render();
      });
      $("world").addEventListener("input", function () { prompts[sec] = $("world").value; $("cmd").textContent = command();
        $("world").classList.toggle("changed", prompts[sec] !== D.sections[sec].world_prompt); });
      $("copy").addEventListener("click", function () {
        var txt = $("cmd").textContent, b = this;
        (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () { flash(b, "Copied"); }, function () {
          var r = document.createRange(); r.selectNodeContents($("cmd")); var s = getSelection(); s.removeAllRanges(); s.addRange(r); flash(b, "Selected: press copy");
        });
      });
      $("dl-edits").addEventListener("click", function () { download("edits.json", edits()); flash(this, "Downloaded"); });
      $("dl-treat").addEventListener("click", function () { download("treatment.json", treatment()); flash(this, "Downloaded"); });
      window.__editor = {edits: edits, treatment: treatment, command: command};
      render(); startParticles();
      window.addEventListener("resize", startParticles);
    })
    .catch(function (e) { document.querySelector(".controls").textContent = "The editor data did not load: " + e; });
})();

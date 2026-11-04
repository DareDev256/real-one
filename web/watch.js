/* Real One watch page. Everything comes from data/, written by scripts/build_watch_page.py from the render's own
   records: the two looks (finished and Real Receipt, same frames), the receipt's measured filmed share (whole video
   and per frame), and which Meta models ran. No number is typed into this file. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const root = document.documentElement;
  const get = async (name) => {
    const r = await fetch("data/" + name, { cache: "no-cache" });
    if (!r.ok) throw new Error(name + " " + r.status);
    return r.json();
  };

  async function main() {
    const m = await get("manifest.json");
    const rc = await get("receipt.json");
    let credits = null;
    try { credits = await get("credits.json"); } catch (e) { credits = null; }

    const A = $("#v-finished"), B = $("#v-receipt");
    A.src = "data/" + m.videos.finished;
    B.src = "data/" + m.videos.receipt;
    A.poster = B.poster = "data/" + m.videos.poster;

    const look = (m.looks && m.looks.finished) || "clean";
    const whole = rc.outputs[look].video.filmed_pct;
    const perFrame = rc.outputs.receipt.frames.filmed_pct;
    const fps = m.fps || 30;
    $("#r-num").textContent = whole.toFixed(1) + "%";

    // the demo link appears once it is a real link
    document.querySelectorAll("[data-demo]").forEach((a) => {
      if (a.getAttribute("href") === "#demo") a.hidden = true;
    });

    if (credits && Array.isArray(credits.models)) {
      const ran = credits.models.filter((x) => x.ran).map((x) => x.name);
      if (ran.length) {
        $("#made").textContent = "Made with Meta's " + (ran.length > 1 ? ran.slice(0, -1).join(", ") + " and " + ran[ran.length - 1] : ran[0]) + ".";
      }
    }

    // the two looks play as one: B follows A
    const sync = () => {
      if (Math.abs(B.currentTime - A.currentTime) > 0.06) B.currentTime = A.currentTime;
      if (A.paused && !B.paused) B.pause();
      if (!A.paused && B.paused) B.play().catch(() => {});
    };
    A.addEventListener("play", sync);
    A.addEventListener("pause", sync);
    A.addEventListener("seeked", sync);
    setInterval(sync, 700);

    const sound = $("#sound");
    const setSound = (on) => {
      A.muted = !on;
      sound.setAttribute("aria-pressed", on ? "true" : "false");
      sound.textContent = on ? "Sound off" : "Sound on";
      if (on && A.paused) A.play().catch(() => {});
    };
    const play = $("#play");
    play.addEventListener("click", (e) => {
      e.stopPropagation();
      A.currentTime = 0;
      setSound(true);
      play.hidden = true;
      sound.hidden = false;
    });
    sound.addEventListener("click", (e) => { e.stopPropagation(); setSound(A.muted); });
    $("#frame").addEventListener("click", () => { if (play.hidden) setSound(A.muted); else play.click(); });

    const btn = $("#receipt"), label = $("#r-label"), what = $("#r-what"), num = $("#r-num");
    const frame = $("#frame");
    const setReceipt = (on) => {
      root.dataset.receipt = on ? "1" : "0";
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      label.textContent = on ? "Back to the video" : "Show the receipt";
      frame.style.setProperty("--fw", frame.clientWidth + "px");
      if (!on) { num.textContent = whole.toFixed(1) + "%"; what.textContent = "of this video is filmed. The rest is made."; }
    };
    btn.addEventListener("click", () => setReceipt(root.dataset.receipt !== "1"));
    window.addEventListener("resize", () => frame.style.setProperty("--fw", frame.clientWidth + "px"));
    document.addEventListener("keydown", (e) => {
      if (e.target && /INPUT|TEXTAREA/.test(e.target.tagName)) return;
      if (e.key === "r" || e.key === "R") setReceipt(root.dataset.receipt !== "1");
      if (e.key === " ") { e.preventDefault(); A.paused ? A.play() : A.pause(); }
    });

    // while the receipt is up, the number is this frame's measured filmed share
    const tick = () => {
      if (root.dataset.receipt === "1" && perFrame.length) {
        const i = Math.max(0, Math.min(perFrame.length - 1, Math.floor(A.currentTime * fps)));
        num.textContent = perFrame[i].toFixed(1) + "%";
        what.textContent = "of this frame is filmed. Over the whole video, " + whole.toFixed(1) + "%.";
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    A.play().catch(() => {});
  }

  main().catch((e) => {
    const p = document.createElement("p");
    p.className = "credit";
    p.textContent = "The video did not load (" + e.message + "). Reload the page.";
    document.querySelector(".words").appendChild(p);
  });
})();

/* The step counter in the header. studio.js owns the steps; this only watches which one is on. */
(function () {
  "use strict";
  var count = document.getElementById("count");
  var steps = document.querySelectorAll(".step"), N = steps.length;
  function update() {
    var on = document.querySelector(".step.on");
    if (!on) return;
    var n = Number(on.getAttribute("data-n"));
    count.textContent = n + " / " + N + "  " + on.getAttribute("data-name");
    document.documentElement.setAttribute("data-step", String(n));
  }
  new MutationObserver(update).observe(document.getElementById("steps"), {subtree: true, attributes: true, attributeFilter: ["class"]});
  update();
})();

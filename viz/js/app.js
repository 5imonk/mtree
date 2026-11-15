import { mount as home } from "./views/home.js";
import { mount as queryMatrix } from "./views/query_matrix.js";
import { mount as baseline } from "./views/baseline.js";
import { mount as streamMix } from "./views/stream_mix.js";
import { mount as paramSweep } from "./views/param_sweep.js";
import { mount as rknnMutate } from "./views/rknn_mutate.js";
import { mount as seedHint } from "./views/seed_hint.js";
import { mount as fromEntry } from "./views/from_entry.js";
import { mount as benches } from "./views/benches.js";

const ROUTES = {
  "": home,
  home: home,
  "query-matrix": queryMatrix,
  baseline: baseline,
  "stream-mix": streamMix,
  "param-sweep": paramSweep,
  "rknn-mutate": rknnMutate,
  "seed-hint": seedHint,
  "from-entry": fromEntry,
  benches: benches,
};

const root = document.getElementById("view");
const navLinks = [...document.querySelectorAll("nav a.nav-link")];

async function render() {
  const hash = (location.hash || "#home").replace(/^#/, "");
  const mount = ROUTES[hash] || home;
  root.innerHTML = "";
  navLinks.forEach((a) => {
    const id = (a.getAttribute("href") || "").replace(/^#/, "");
    a.classList.toggle("active", id === hash || (hash === "" && id === "home"));
  });
  try {
    await mount(root);
  } catch (err) {
    root.innerHTML = "";
    const pre = document.createElement("div");
    pre.className = "msg warn";
    pre.textContent = String(err && err.stack ? err.stack : err);
    root.appendChild(pre);
  }
}

window.addEventListener("hashchange", render);
render();

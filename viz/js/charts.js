/** Chart.js helpers shared by views. */

const COLORS = [
  "#5b9fd4",
  "#5cad7a",
  "#d4a05b",
  "#c75c8a",
  "#8b7bd4",
  "#5bc4c4",
  "#c75c5c",
  "#7ab0c7",
  "#a0c75c",
  "#d47a5b",
];

export function colorAt(i) {
  return COLORS[i % COLORS.length];
}

export function destroyChart(chart) {
  if (chart) chart.destroy();
}

export function lineChart(ctx, { labels, datasets, yTitle, xTitle, logY = false }) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: datasets.map((ds, i) => ({
        borderColor: ds.color || colorAt(i),
        backgroundColor: "transparent",
        tension: 0.15,
        pointRadius: 3,
        ...ds,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#c5d0db" } },
      },
      scales: {
        x: {
          title: { display: !!xTitle, text: xTitle, color: "#8b9aab" },
          ticks: { color: "#8b9aab" },
          grid: { color: "#2a3542" },
        },
        y: {
          type: logY ? "logarithmic" : "linear",
          title: { display: !!yTitle, text: yTitle, color: "#8b9aab" },
          ticks: { color: "#8b9aab" },
          grid: { color: "#2a3542" },
        },
      },
    },
  });
}

export function barChart(ctx, { labels, datasets, yTitle }) {
  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: datasets.map((ds, i) => ({
        backgroundColor: ds.color || colorAt(i),
        borderRadius: 4,
        ...ds,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#c5d0db" } },
      },
      scales: {
        x: {
          ticks: { color: "#8b9aab", maxRotation: 45, minRotation: 0 },
          grid: { display: false },
        },
        y: {
          title: { display: !!yTitle, text: yTitle, color: "#8b9aab" },
          ticks: { color: "#8b9aab" },
          grid: { color: "#2a3542" },
        },
      },
    },
  });
}

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

export function selectControl(label, options, value, onChange) {
  const wrap = el("div", { className: "control" });
  wrap.appendChild(el("label", { text: label }));
  const sel = el("select");
  for (const opt of options) {
    const o = el("option", { value: String(opt), text: String(opt) });
    if (String(opt) === String(value)) o.selected = true;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => onChange(coerceOpt(sel.value, options)));
  wrap.appendChild(sel);
  return wrap;
}

function coerceOpt(v, options) {
  const sample = options[0];
  if (typeof sample === "number") return Number(v);
  return v;
}

export function missingCsv(path, hint) {
  const box = el("div", { className: "msg warn" });
  box.appendChild(el("div", { text: `CSV fehlt: ${path}` }));
  box.appendChild(
    el("div", {
      text: hint || "Report zuerst erzeugen, dann diese Seite neu laden.",
    })
  );
  return box;
}

export function fileProtocolHint() {
  if (location.protocol !== "file:") return null;
  const box = el("div", { className: "msg warn" });
  box.appendChild(
    document.createTextNode("file:// blockiert fetch. Aus dem Repo-Root starten: ")
  );
  box.appendChild(el("code", { text: "python3 -m http.server 8765" }));
  box.appendChild(document.createTextNode(" und öffnen: "));
  box.appendChild(el("code", { text: "http://localhost:8765/viz/" }));
  return box;
}

/** Short how-to-read box: title + bullet strings (may contain HTML via {html:true}). */
export function readGuide(title, bullets) {
  const box = el("div", { className: "read-guide" });
  box.appendChild(el("div", { className: "rg-title", text: title }));
  const ul = el("ul");
  for (const b of bullets) {
    const li = el("li");
    if (typeof b === "string") {
      li.textContent = b;
    } else if (b && b.html) {
      li.innerHTML = b.html;
    } else if (b && b.nodes) {
      for (const n of b.nodes) li.appendChild(typeof n === "string" ? document.createTextNode(n) : n);
    }
    ul.appendChild(li);
  }
  box.appendChild(ul);
  return box;
}

import { loadCsv, uniqueSorted } from "../csv.js";
import {
  destroyChart,
  el,
  fileProtocolHint,
  lineChart,
  missingCsv,
  readGuide,
  selectControl,
} from "../charts.js";

const PATH = "../reports/knn_query_matrix.csv";
const HINT =
  "cargo run --release --example knn_query_matrix_report";

export async function mount(root) {
  const hint = fileProtocolHint();
  if (hint) root.appendChild(hint);

  let rows;
  try {
    ({ rows } = await loadCsv(PATH));
  } catch {
    root.appendChild(missingCsv(PATH, HINT));
    return;
  }

  const ks = uniqueSorted(rows, "k", true);
  const filters = uniqueSorted(rows, "filter");
  const modes = uniqueSorted(rows, "mode");
  const metrics = [
    ["avg_ms", "avg_ms"],
    ["avg_routing", "avg_routing"],
    ["avg_leaves", "avg_leaves"],
    ["speedup_vs_needle_none_filter_off", "speedup"],
  ];

  let k = ks[0];
  let filter = filters.includes("off") ? "off" : filters[0];
  let metric = "avg_ms";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Query Matrix" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "<strong>avg_ms</strong> — Median der mittleren Query-Zeit (ms/Query). <span class=\"better\">Niedriger = schneller</span>.",
    },
    {
      html: "<strong>avg_routing / avg_leaves</strong> — besuchte Routing- bzw. Leaf-Knoten pro Query (<code>knn_stats</code>). <span class=\"better\">Weniger Visits = weniger Distanzarbeit</span>.",
    },
    {
      html: "<strong>speedup</strong> — vs. <code>needle_none</code> + filter=off; &gt;1 heißt schneller als Baseline.",
    },
    {
      html: "<strong>Modi</strong>: <code>needle_none</code> (kalt), <code>needle_hint</code> (Seed), <code>from_entry</code> (gespeicherter Key). Filter≈50 % active.",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Latency / visits vs n (by mode)" }));
  const canvas = el("canvas");
  chartWrap.appendChild(canvas);

  function render() {
    destroyChart(chart);
    const ns = uniqueSorted(
      rows.filter((r) => r.k === k && r.filter === filter),
      "n",
      true
    );
    const datasets = modes.map((mode) => ({
      label: mode,
      data: ns.map((n) => {
        const row = rows.find(
          (r) => r.n === n && r.k === k && r.filter === filter && r.mode === mode
        );
        return row ? row[metric] : null;
      }),
    }));
    chart = lineChart(canvas.getContext("2d"), {
      labels: ns,
      datasets,
      xTitle: "n",
      yTitle: metric,
      logY: metric === "avg_ms",
    });
  }

  controls.appendChild(
    selectControl("k", ks, k, (v) => {
      k = v;
      render();
    })
  );
  controls.appendChild(
    selectControl("filter", filters, filter, (v) => {
      filter = v;
      render();
    })
  );
  controls.appendChild(
    selectControl(
      "metric",
      metrics.map((m) => m[0]),
      metric,
      (v) => {
        metric = v;
        render();
      }
    )
  );

  root.appendChild(header);
  root.appendChild(guide);
  root.appendChild(controls);
  root.appendChild(chartWrap);
  render();
}

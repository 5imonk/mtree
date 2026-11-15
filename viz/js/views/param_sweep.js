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

const PATH = "../reports/knn_param_sweep.csv";
const HINT = "cargo run --release --example knn_param_sweep_report";

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

  const ns = uniqueSorted(rows, "n", true);
  let n = ns.includes(1000) ? 1000 : ns[0];
  let metric = "avg_ms";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Param Sweep" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "X-Achse = <strong>max_node_size</strong>; feste k=10, <code>needle_none</code>, filter off.",
    },
    {
      html: "<strong>avg_ms</strong> — Query-Latenz. <span class=\"better\">Niedriger = bessere Knotengröße für dieses n</span>.",
    },
    {
      html: "<strong>rel_to_best_n</strong> — avg_ms / bestes avg_ms bei gleichem n; 1.0 = Optimum. <strong>kind</strong>: sweep / heuristic / fixed.",
    },
    {
      html: "<strong>insert_us / erase_us / update_us</strong> — Mutations-Nebenkosten (µs/op). <span class=\"better\">Niedriger = günstiger Hotpath</span>.",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Metric vs max_node_size" }));
  const canvas = el("canvas");
  chartWrap.appendChild(canvas);

  function render() {
    destroyChart(chart);
    const slice = rows
      .filter((r) => r.n === n)
      .slice()
      .sort((a, b) => a.max - b.max);
    const kinds = uniqueSorted(slice, "kind");
    const maxes = uniqueSorted(slice, "max", true);
    const datasets = kinds.map((kind) => ({
      label: kind,
      data: maxes.map((max) => {
        const row = slice.find((r) => r.max === max && r.kind === kind);
        return row ? row[metric] : null;
      }),
    }));
    chart = lineChart(canvas.getContext("2d"), {
      labels: maxes,
      datasets,
      xTitle: "max_node_size",
      yTitle: metric,
    });
  }

  controls.appendChild(
    selectControl("n", ns, n, (v) => {
      n = v;
      render();
    })
  );
  controls.appendChild(
    selectControl(
      "metric",
      ["avg_ms", "build_ms", "insert_us", "erase_us", "update_us", "rel_to_best_n"],
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

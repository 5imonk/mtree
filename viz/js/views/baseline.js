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

const PATH = "../reports/knn_baseline_latency.csv";
const HINT = "cargo run --release --example knn_baseline_latency_report";

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
  const impls = uniqueSorted(rows, "impl");
  let k = ks.includes(10) ? 10 : ks[0];
  let metric = "avg_ms";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Baseline Latency" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "<strong>avg_ms</strong> — Query-Latenz (ms). <span class=\"better\">Niedriger = besser</span>. Fairer Headline: <code>mtree_needle_none</code> vs. vpsearch / vp_tree / ball_tree.",
    },
    {
      html: "<strong>mtree_needle_hint</strong> ist unfair (Seed-Wissen); kiddo/rstar sind Euclidean-Ceiling, nicht Metrik-Peers.",
    },
    {
      html: "<strong>ratio_vs_kiddo</strong> — kiddo/avg; &gt;1 heißt schneller als kiddo. <strong>build_ms</strong> nur Nebenmetrik (Index-Aufbau).",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Impl comparison vs n" }));
  const canvas = el("canvas");
  chartWrap.appendChild(canvas);

  function render() {
    destroyChart(chart);
    const ns = uniqueSorted(
      rows.filter((r) => r.k === k),
      "n",
      true
    );
    const datasets = impls.map((impl) => ({
      label: impl,
      data: ns.map((n) => {
        const row = rows.find((r) => r.n === n && r.k === k && r.impl === impl);
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
    selectControl("metric", ["avg_ms", "build_ms", "ratio_vs_kiddo"], metric, (v) => {
      metric = v;
      render();
    })
  );

  root.appendChild(header);
  root.appendChild(guide);
  root.appendChild(controls);
  root.appendChild(chartWrap);
  render();
}

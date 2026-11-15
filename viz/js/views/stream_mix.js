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

const PATH = "../reports/knn_stream_mix.csv";
const HINT = "cargo run --release --example knn_stream_mix_report";

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
  const modes = uniqueSorted(rows, "mode");
  const mixes = uniqueSorted(rows, "mix");
  let k = ks[0];
  let mode = modes[0];
  let metric = "avg_ms";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Stream Mix" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "<strong>avg_ms</strong> — Medianzeit pro Zyklus (ms). <span class=\"better\">Niedriger = besser</span>.",
    },
    {
      html: "<strong>ops_per_s</strong> — Durchsatz. <span class=\"better\">Höher = besser</span>.",
    },
    {
      html: "<strong>mix</strong>: <code>query_only</code>, <code>replace_only</code> (<code>update_key</code>), <code>query_replace</code> — Stream-Clustering-Workload.",
    },
    {
      html: "<strong>mode</strong>: <code>needle_none</code> vs. <code>needle_hint</code> (Seed bei Queries).",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Mix comparison vs n" }));
  const canvas = el("canvas");
  chartWrap.appendChild(canvas);

  function render() {
    destroyChart(chart);
    const filtered = rows.filter((r) => r.k === k && r.mode === mode);
    const ns = uniqueSorted(filtered, "n", true);
    const datasets = mixes.map((mix) => ({
      label: mix,
      data: ns.map((n) => {
        const row = filtered.find((r) => r.n === n && r.mix === mix);
        return row ? row[metric] : null;
      }),
    }));
    chart = lineChart(canvas.getContext("2d"), {
      labels: ns,
      datasets,
      xTitle: "n",
      yTitle: metric,
    });
  }

  controls.appendChild(
    selectControl("k", ks, k, (v) => {
      k = v;
      render();
    })
  );
  controls.appendChild(
    selectControl("mode", modes, mode, (v) => {
      mode = v;
      render();
    })
  );
  controls.appendChild(
    selectControl("metric", ["avg_ms", "ops_per_s"], metric, (v) => {
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

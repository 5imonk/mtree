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

const PATH = "../reports/knn_seed_hint.csv";
const HINT = "cargo run --release --example knn_seed_hint_report";

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

  const policies = uniqueSorted(rows, "policy");
  const ks = uniqueSorted(rows, "k", true);
  const modes = uniqueSorted(rows, "mode");
  let policy = policies.includes("scaled") ? "scaled" : policies[0];
  let k = ks[0];
  let metric = "avg_ms";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Seed Hint" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "<strong>avg_ms</strong> — Query-Zeit. <span class=\"better\">Niedriger = besser</span>. <strong>speedup_vs_none</strong> &gt;1 = schneller als ohne Seed.",
    },
    {
      html: "<strong>Modi</strong>: <code>none</code> (Baseline), <code>nearest</code> (1-NN-Seed), <code>kth</code> (k-ter Treffer), <code>random</code> (Mittel über Draws).",
    },
    {
      html: "<strong>avg_routing</strong> / <strong>routing_ratio_vs_none</strong> — Visit-Proxy; Ratio &gt;1 = weniger Routing als <code>none</code>.",
    },
    {
      html: "<strong>policy</strong>: <code>fixed</code> (16/64/16) vs. <code>scaled</code> (<code>node_params_for_capacity</code>).",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Seed modes vs n" }));
  const canvas = el("canvas");
  chartWrap.appendChild(canvas);

  function render() {
    destroyChart(chart);
    const slice = rows.filter((r) => r.policy === policy && r.k === k);
    const ns = uniqueSorted(slice, "n", true);
    const datasets = modes.map((mode) => ({
      label: mode,
      data: ns.map((n) => {
        const row = slice.find((r) => r.n === n && r.mode === mode);
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
    selectControl("policy", policies, policy, (v) => {
      policy = v;
      render();
    })
  );
  controls.appendChild(
    selectControl("k", ks, k, (v) => {
      k = v;
      render();
    })
  );
  controls.appendChild(
    selectControl(
      "metric",
      ["avg_ms", "avg_routing", "speedup_vs_none", "routing_ratio_vs_none"],
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

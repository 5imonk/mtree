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

const PATH = "../reports/knn_from_entry.csv";
const HINT = "cargo run --release --example knn_from_entry_report";

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
    el("h2", { text: "From Entry" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "Queries sind gespeicherte <strong>EntryId</strong>s (nicht Raum-Needles).",
    },
    {
      html: "<strong>avg_ms</strong> — <span class=\"better\">niedriger besser</span>. <strong>speedup_vs_needle_none</strong> &gt;1 = schneller als Needle ohne Seed.",
    },
    {
      html: "<strong>from_entry</strong> schließt Self aus; <code>needle_*</code> inkludieren Self bei Dist 0 — Vergleich mit Bedacht.",
    },
    {
      html: "<strong>avg_routing</strong> / Ratio — Visit-Proxy für Distanzarbeit im Baum.",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const chartWrap = el("div", { className: "chart-wrap" });
  chartWrap.appendChild(el("h3", { text: "Modes vs n" }));
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
      [
        "avg_ms",
        "avg_routing",
        "speedup_vs_needle_none",
        "routing_ratio_vs_needle_none",
      ],
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

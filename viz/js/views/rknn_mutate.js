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

const PATH = "../reports/rknn_mutate_cost.csv";
const HINT = "cargo run --release --example rknn_mutate_cost_report";

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

  const ops = uniqueSorted(rows, "op");
  const ks = uniqueSorted(rows, "k", true);
  let op = ops.includes("insert") ? "insert" : ops[0];
  let k = ks.includes(10) ? 10 : ks[0];
  let chartMain;
  let chartAff;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "RkNN Mutate Cost" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "Alle Zeiten in <strong>µs/op</strong> (Median). <span class=\"better\">Niedriger = günstiger</span>.",
    },
    {
      html: "<strong>mutate_us</strong> — Insert/Erase mit Graph-Maintenance. <strong>plain_mutate_us</strong> — ohne Graph (geometrische Baseline).",
    },
    {
      html: "<strong>discover_us</strong> + <strong>affected_count</strong> — wer muss angepasst werden (RkNN bzw. Reverse-Scan). <strong>refresh_us_total</strong> — Forward-<code>knn_from_entry</code> je Opfer.",
    },
    {
      html: "<strong>selective_total_us</strong> ≈ discover + plain + refresh — SDO-artiger Pfad zum Vergleich mit <code>mutate_us</code>.",
    },
    {
      html: "<strong>avg_routing / avg_leaves</strong> — Visits pro Affected beim Refresh (<code>knn_stats</code>). <span class=\"better\">Weniger = weniger Distanzarbeit</span>.",
    },
  ]);
  const controls = el("div", { className: "controls" });

  const wrapMain = el("div", { className: "chart-wrap" });
  wrapMain.appendChild(
    el("h3", { text: "µs/op: graph mutate vs selective total vs plain" })
  );
  const canvasMain = el("canvas");
  wrapMain.appendChild(canvasMain);

  const wrapAff = el("div", { className: "chart-wrap" });
  wrapAff.appendChild(el("h3", { text: "Affected count & refresh routing" }));
  const canvasAff = el("canvas");
  wrapAff.appendChild(canvasAff);

  function render() {
    destroyChart(chartMain);
    destroyChart(chartAff);
    const slice = rows.filter((r) => r.op === op && r.k === k);
    const ns = uniqueSorted(slice, "n", true);
    const byN = (field) =>
      ns.map((n) => {
        const row = slice.find((r) => r.n === n);
        return row ? row[field] : null;
      });

    chartMain = lineChart(canvasMain.getContext("2d"), {
      labels: ns,
      datasets: [
        { label: "mutate_us (graph)", data: byN("mutate_us") },
        { label: "plain_mutate_us", data: byN("plain_mutate_us") },
        { label: "discover_us", data: byN("discover_us") },
        { label: "refresh_us_total", data: byN("refresh_us_total") },
        { label: "selective_total_us", data: byN("selective_total_us") },
      ],
      xTitle: "n",
      yTitle: "µs / op",
      logY: true,
    });

    chartAff = lineChart(canvasAff.getContext("2d"), {
      labels: ns,
      datasets: [
        { label: "affected_count", data: byN("affected_count") },
        { label: "avg_routing", data: byN("avg_routing") },
        { label: "avg_leaves", data: byN("avg_leaves") },
      ],
      xTitle: "n",
      yTitle: "count / visits",
    });
  }

  controls.appendChild(
    selectControl("op", ops, op, (v) => {
      op = v;
      render();
    })
  );
  controls.appendChild(
    selectControl("k", ks, k, (v) => {
      k = v;
      render();
    })
  );

  root.appendChild(header);
  root.appendChild(guide);
  root.appendChild(controls);
  root.appendChild(wrapMain);
  root.appendChild(wrapAff);
  render();
}

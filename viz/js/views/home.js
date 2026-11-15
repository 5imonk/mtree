import { el, readGuide } from "../charts.js";

const LINKS = [
  {
    id: "query-matrix",
    title: "Query Matrix",
    csv: "reports/knn_query_matrix.csv",
    cmd: "cargo run --release --example knn_query_matrix_report",
  },
  {
    id: "baseline",
    title: "Baseline Latency",
    csv: "reports/knn_baseline_latency.csv",
    cmd: "cargo run --release --example knn_baseline_latency_report",
  },
  {
    id: "stream-mix",
    title: "Stream Mix",
    csv: "reports/knn_stream_mix.csv",
    cmd: "cargo run --release --example knn_stream_mix_report",
  },
  {
    id: "param-sweep",
    title: "Param Sweep",
    csv: "reports/knn_param_sweep.csv",
    cmd: "cargo run --release --example knn_param_sweep_report",
  },
  {
    id: "rknn-mutate",
    title: "RkNN Mutate Cost",
    csv: "reports/rknn_mutate_cost.csv",
    cmd: "cargo run --release --example rknn_mutate_cost_report",
  },
  {
    id: "seed-hint",
    title: "Seed Hint",
    csv: "reports/knn_seed_hint.csv",
    cmd: "cargo run --release --example knn_seed_hint_report",
  },
  {
    id: "from-entry",
    title: "From Entry",
    csv: "reports/knn_from_entry.csv",
    cmd: "cargo run --release --example knn_from_entry_report",
  },
  {
    id: "benches",
    title: "Criterion Benches",
    csv: "reports/criterion_benches.csv",
    cmd: "cargo bench … && cargo run --release --example export_criterion_csv",
  },
];

export async function mount(root) {
  root.appendChild(
    el("div", { className: "view-header" }, [
      el("h2", { text: "mtree-experiment Hub" }),
      el("div", {
        className: "meta",
        text: "Standalone · CSV aus reports/",
      }),
    ])
  );

  root.appendChild(
    el("div", { className: "msg info" }, [
      "Server aus Repo-Root: ",
      el("code", { text: "python3 -m http.server 8765" }),
      " → ",
      el("code", { text: "http://localhost:8765/viz/" }),
    ])
  );

  root.appendChild(
    readGuide("Legende (alle Reports)", [
      {
        html: "Jede View hat oben eine <strong>Lesehilfe</strong>: was gemessen wird und was „besser“ bedeutet.",
      },
      {
        html: "Zeiten: meist <strong>ms/Query</strong> oder <strong>µs/op</strong> (Median). <span class=\"better\">Niedriger = schneller</span>, außer Durchsatz (<code>ops_per_s</code>).",
      },
      {
        html: "<strong>Routing/Leaves</strong> (<code>knn_stats</code>) ≈ Distanz-/Visit-Arbeit im Baum — weniger ist günstiger.",
      },
      {
        html: "Speedup/Ratio &gt;1 heißt besser als die jeweilige Baseline (je View definiert).",
      },
    ])
  );

  const list = el("div", { className: "stats" });
  for (const item of LINKS) {
    const card = el("a", {
      className: "stat",
      href: `#${item.id}`,
    });
    card.appendChild(el("div", { className: "label", text: item.title }));
    card.appendChild(el("div", { className: "value", text: "→" }));
    card.appendChild(
      el("div", {
        className: "label",
        text: item.csv,
      })
    );
    list.appendChild(card);
  }
  root.appendChild(list);

  const cmds = el("div", { className: "table-wrap" });
  const table = el("table", { className: "data" });
  table.appendChild(
    el("thead", {}, [
      el("tr", {}, [
        el("th", { text: "View" }),
        el("th", { text: "Generate" }),
      ]),
    ])
  );
  const tbody = el("tbody");
  for (const item of LINKS) {
    tbody.appendChild(
      el("tr", {}, [
        el("td", { text: item.title }),
        el("td", {}, [el("code", { text: item.cmd })]),
      ])
    );
  }
  table.appendChild(tbody);
  cmds.appendChild(table);
  root.appendChild(cmds);
}

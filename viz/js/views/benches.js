import { loadCsv, uniqueSorted } from "../csv.js";
import {
  barChart,
  colorAt,
  destroyChart,
  el,
  fileProtocolHint,
  lineChart,
  missingCsv,
  readGuide,
  selectControl,
} from "../charts.js";

const PATH = "../reports/criterion_benches.csv";
const HINT =
  "cargo bench … && cargo run --release --example export_criterion_csv";

/** Curated cross-group views (not every pair makes sense). */
const COMPARE_PRESETS = [
  { id: "none", label: "— single group —", groups: null },
  {
    id: "rknn_ins_erase",
    label: "RkNN mutate: insert ↔ erase",
    groups: ["rknn_mutate_insert", "rknn_mutate_erase"],
  },
  {
    id: "rknn_erase_vs_erase",
    label: "RkNN mutate_erase ↔ erase",
    groups: ["rknn_mutate_erase", "erase"],
  },
  {
    id: "rknn_insert_vs_insert",
    label: "RkNN mutate_insert ↔ insert",
    groups: ["rknn_mutate_insert", "insert"],
  },
  {
    id: "rknn_discover",
    label: "RkNN discover: search ↔ reverse",
    groups: ["rknn_discover"],
  },
  {
    id: "knn_search",
    label: "knn_search: mtree ↔ naive",
    groups: ["knn_search"],
  },
];

function enrich(row) {
  const id = String(row.bench_id);
  let n = null;
  let k = null;
  let variant = null;

  let m = id.match(/n=(\d+)/) || id.match(/(?:^|[/_])n(\d+)(?:[/_]|$)/i);
  if (m) n = Number(m[1]);
  if (n == null) {
    m = id.match(/\/(\d+)$/);
    if (m) n = Number(m[1]);
  }

  m = id.match(/k=(\d+)/) || id.match(/(?:^|[/_])k(\d+)(?:[/_]|$)/i);
  if (m) k = Number(m[1]);

  if (id.includes("graph_n") || /^graph[_/]/.test(id)) variant = "graph";
  else if (id.includes("plain_n") || /^plain[_/]/.test(id)) variant = "plain";
  else if (/^mtree\//.test(id)) variant = "mtree";
  else if (/^naive/.test(id)) variant = "naive";
  else if (id.includes("rknn_search_")) variant = "rknn_search";
  else if (id.includes("reverse_scan_")) variant = "reverse_scan";
  else if (id.includes("knn_from_entry_")) variant = "knn_from_entry";
  else if (id.endsWith("needle_none") || id.includes("/needle_none"))
    variant = "needle_none";
  else if (id.endsWith("needle_hint") || id.includes("/needle_hint"))
    variant = "needle_hint";
  else if (id.endsWith("from_entry") || id.includes("/from_entry"))
    variant = "from_entry";
  else variant = id.split("/")[0];

  return { ...row, n, k, variant };
}

function seriesKey(row, multiGroup) {
  if (multiGroup) {
    return row.variant ? `${row.group}:${row.variant}` : row.group;
  }
  return row.variant || row.bench_id;
}

function shortLabel(row) {
  const bits = [];
  if (row.variant) bits.push(row.variant);
  if (row.n != null) bits.push(`n=${row.n}`);
  if (row.k != null) bits.push(`k=${row.k}`);
  return bits.length ? bits.join(" ") : row.bench_id;
}

function avg(vals) {
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

export async function mount(root) {
  const hint = fileProtocolHint();
  if (hint) root.appendChild(hint);

  let raw;
  try {
    ({ rows: raw } = await loadCsv(PATH));
  } catch {
    root.appendChild(missingCsv(PATH, HINT));
    return;
  }

  const rows = raw.map(enrich);
  const groups = uniqueSorted(rows, "group");

  let group = groups.find((g) => String(g).startsWith("rknn_mutate")) || groups[0];
  let compareId = "none";
  let nFilter = "all";
  let kFilter = "all";
  let metric = "mean_ns";
  let chart;

  const header = el("div", { className: "view-header" }, [
    el("h2", { text: "Criterion Benches" }),
    el("div", { className: "meta", text: PATH }),
  ]);
  const guide = readGuide("Wie lesen?", [
    {
      html: "Werte in <strong>ns/Iteration</strong>. <span class=\"better\">Niedriger = schneller</span>.",
    },
    {
      html: "<strong>n / k</strong> aus Bench-IDs geparst (<code>graph_n500_k20</code>, <code>mtree/1000</code>, …). „all“ = kein Filter.",
    },
    {
      html: "<strong>Compare</strong>: kuratierte Gegenüberstellungen (Insert↔Erase, RkNN↔plain, Discover). Nicht jede Gruppen-Kombi ist sinnvoll.",
    },
    {
      html: "Compare-/n-Chart: X = Modellgröße, Serien = Gruppe:Variante. Bei k=all und mehreren k wird je Serie@n gemittelt.",
    },
  ]);
  const controls = el("div", { className: "controls" });
  const groupHost = el("span");
  const compareHost = el("span");
  const nHost = el("span");
  const kHost = el("span");
  const metricHost = el("span");
  controls.append(groupHost, compareHost, nHost, kHost, metricHost);

  const chartWrap = el("div", { className: "chart-wrap" });
  const chartTitle = el("h3", { text: "…" });
  const canvas = el("canvas");
  chartWrap.append(chartTitle, canvas);

  const note = el("div", { className: "msg info" }, [
    "Auch Criterion-HTML: ",
    el("code", { text: "target/criterion/report/index.html" }),
  ]);

  function activeGroups() {
    const preset = COMPARE_PRESETS.find((p) => p.id === compareId);
    if (preset?.groups) {
      return preset.groups.filter((g) => groups.includes(g));
    }
    return [group];
  }

  function isMulti() {
    return compareId !== "none";
  }

  function optsFor(field) {
    const gs = activeGroups();
    const vals = [
      ...new Set(
        rows
          .filter((r) => gs.includes(r.group) && r[field] != null)
          .map((r) => r[field])
      ),
    ].sort((a, b) => a - b);
    return ["all", ...vals];
  }

  function filtered() {
    const gs = new Set(activeGroups());
    return rows.filter((r) => {
      if (!gs.has(r.group)) return false;
      if (nFilter !== "all" && r.n !== nFilter) return false;
      // Rows without k (e.g. plain erase/insert) stay visible under a k filter.
      if (kFilter !== "all" && r.k != null && r.k !== kFilter) return false;
      return true;
    });
  }

  function render() {
    destroyChart(chart);
    const slice = filtered();
    const gs = activeGroups();
    const multi = isMulti();

    if (!slice.length) {
      chartTitle.textContent = "Keine Treffer für Filter";
      chart = barChart(canvas.getContext("2d"), {
        labels: ["—"],
        datasets: [{ label: metric, data: [0] }],
        yTitle: metric,
      });
      return;
    }

    const hasN = slice.some((r) => r.n != null);
    const lineMode = hasN && (multi || nFilter === "all");

    if (lineMode) {
      const ns = uniqueSorted(
        slice.filter((r) => r.n != null),
        "n",
        true
      );
      const series = [
        ...new Set(slice.map((r) => seriesKey(r, multi || gs.length > 1))),
      ].sort();
      const datasets = series.map((name, i) => ({
        label: name,
        color: colorAt(i),
        data: ns.map((n) => {
          const hits = slice.filter(
            (r) =>
              r.n === n && seriesKey(r, multi || gs.length > 1) === name
          );
          if (!hits.length) return null;
          return avg(hits.map((h) => Number(h[metric])));
        }),
      }));
      const kNote =
        kFilter === "all" ? "k=all (Mittel je n)" : `k=${kFilter}`;
      chartTitle.textContent = multi
        ? `Compare vs n · ${metric} · ${kNote}`
        : `${gs[0]} vs n · ${metric} · ${kNote}`;
      chart = lineChart(canvas.getContext("2d"), {
        labels: ns,
        datasets,
        xTitle: "n (model size)",
        yTitle: metric,
        logY: true,
      });
      return;
    }

    const sorted = slice
      .slice()
      .sort((a, b) => Number(a[metric]) - Number(b[metric]));
    chartTitle.textContent = `Bars · ${gs.join(" + ")} (${sorted.length})`;
    chart = barChart(canvas.getContext("2d"), {
      labels: sorted.map(shortLabel),
      datasets: [{ label: metric, data: sorted.map((r) => r[metric]) }],
      yTitle: metric,
    });
  }

  function paintControls() {
    const nOpts = optsFor("n");
    const kOpts = optsFor("k");
    if (!nOpts.includes(nFilter) && nFilter !== "all") nFilter = "all";
    if (!kOpts.includes(kFilter) && kFilter !== "all") kFilter = "all";

    groupHost.replaceChildren();
    compareHost.replaceChildren();
    nHost.replaceChildren();
    kHost.replaceChildren();
    metricHost.replaceChildren();

    const groupSel = selectControl("group", groups, group, (v) => {
      group = v;
      compareId = "none";
      paintControls();
      render();
    });
    if (isMulti()) groupSel.querySelector("select").disabled = true;
    groupHost.appendChild(groupSel);

    compareHost.appendChild(
      selectControl(
        "compare",
        COMPARE_PRESETS.map((p) => p.label),
        COMPARE_PRESETS.find((p) => p.id === compareId).label,
        (label) => {
          compareId = COMPARE_PRESETS.find((p) => p.label === label)?.id || "none";
          paintControls();
          render();
        }
      )
    );

    nHost.appendChild(
      selectControl("n (model)", nOpts, nFilter, (v) => {
        nFilter = v === "all" ? "all" : Number(v);
        render();
      })
    );
    kHost.appendChild(
      selectControl("k", kOpts.length ? kOpts : ["all"], kFilter, (v) => {
        kFilter = v === "all" ? "all" : Number(v);
        render();
      })
    );
    metricHost.appendChild(
      selectControl("metric", ["mean_ns", "median_ns", "stddev_ns"], metric, (v) => {
        metric = v;
        render();
      })
    );
  }

  root.append(header, guide, note, controls, chartWrap);
  paintControls();
  render();
}

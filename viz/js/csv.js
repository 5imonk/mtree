/** Minimal CSV helpers for report dashboards. */

export function parseCsv(text) {
  const lines = text
    .trim()
    .split(/\r?\n/)
    .filter((l) => l.length > 0);
  if (lines.length === 0) return { headers: [], rows: [] };
  const headers = splitCsvLine(lines[0]);
  const rows = lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = coerce(cells[i] ?? "");
    });
    return obj;
  });
  return { headers, rows };
}

function splitCsvLine(line) {
  // Reports do not quote fields; keep simple.
  return line.split(",");
}

function coerce(v) {
  if (v === "") return v;
  const t = v.trim();
  if (/^-?\d+(\.\d+)?([eE][-+]?\d+)?$/.test(t)) {
    const n = Number(t);
    if (Number.isFinite(n)) return n;
  }
  return v;
}

export async function loadCsv(path) {
  const res = await fetch(path);
  if (!res.ok) {
    const err = new Error(`Failed to load ${path}: HTTP ${res.status}`);
    err.status = res.status;
    err.path = path;
    throw err;
  }
  const text = await res.text();
  return parseCsv(text);
}

export function uniqueSorted(rows, key, numeric = false) {
  const set = new Set(rows.map((r) => r[key]));
  const arr = [...set];
  if (numeric) {
    arr.sort((a, b) => Number(a) - Number(b));
  } else {
    arr.sort((a, b) => String(a).localeCompare(String(b)));
  }
  return arr;
}

export function filterRows(rows, preds) {
  return rows.filter((r) => preds.every((p) => p(r)));
}

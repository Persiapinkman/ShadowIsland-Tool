from __future__ import annotations

import csv
import json
import shutil
from importlib import resources
from pathlib import Path

from ..io import gc_fraction
from ..types import FastaRecord, GeneRecord, WindowScore


GC_WINDOW_BP = 1000


def write_result_package(
    out_dir: Path | str,
    *,
    records: list[FastaRecord],
    windows: list[WindowScore],
    intervals: dict[str, list[dict[str, object]]],
    genes: list[GeneRecord],
    provenance: dict[str, object],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_window_probs(out_dir / "window_probs.csv", windows)
    write_intervals(out_dir / "predicted_intervals.csv", intervals)
    write_gene_catalog(out_dir / "gene_catalog.csv", genes)
    write_region_summary(out_dir / "region_summary.csv", intervals)
    write_report(out_dir / "functional_report.md", records, intervals, genes)
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_viewer(out_dir / "viewer", records, intervals, genes)


def write_window_probs(path: Path, rows: list[WindowScore]) -> None:
    fieldnames = ["accession", "start", "end", "gc", "p_seq", "p_comp", "p_gi", "logit_seq", "logit_svm", "y_pred"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_intervals(path: Path, intervals: dict[str, list[dict[str, object]]]) -> None:
    rows = [dict(accession=accession, **row) for accession, values in intervals.items() for row in values]
    write_dict_csv(path, rows)


def write_gene_catalog(path: Path, genes: list[GeneRecord]) -> None:
    write_dict_csv(path, [gene.__dict__ for gene in genes])


def write_region_summary(path: Path, intervals: dict[str, list[dict[str, object]]]) -> None:
    rows = []
    for accession, values in intervals.items():
        for tier in ("high", "medium", "low"):
            tier_rows = [row for row in values if row.get("kind") == tier]
            rows.append(
                {
                    "accession": accession,
                    "tier": tier,
                    "n_regions": len(tier_rows),
                    "total_bp": sum(int(row.get("length", 0)) for row in tier_rows),
                    "n_genes": sum(int(row.get("refseq_n_genes", 0) or 0) for row in tier_rows),
                    "n_mobility": sum(int(row.get("refseq_n_mobility", 0) or 0) for row in tier_rows),
                    "n_virulence": sum(int(row.get("refseq_n_virulence", 0) or 0) for row in tier_rows),
                    "n_resistance": sum(int(row.get("refseq_n_resistance", 0) or 0) for row in tier_rows),
                    "n_phage": sum(int(row.get("refseq_n_phage", 0) or 0) for row in tier_rows),
                    "n_trna": sum(int(row.get("refseq_n_trna", 0) or 0) for row in tier_rows),
                }
            )
    write_dict_csv(path, rows)


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, records: list[FastaRecord], intervals: dict[str, list[dict[str, object]]], genes: list[GeneRecord]) -> None:
    lines = ["# ShadowIsland prediction report", ""]
    lines.append(f"- Records: {len(records)}")
    lines.append(f"- GFF features parsed: {len(genes)}")
    lines.append("")
    for record in records:
        rows = intervals.get(record.accession, [])
        lines.append(f"## {record.accession}")
        lines.append("")
        lines.append(f"- Length: {len(record.sequence):,} bp")
        lines.append(f"- GC: {gc_fraction(record.sequence) * 100:.2f}%")
        lines.append(f"- Predicted intervals: {len(rows)}")
        for tier in ("high", "medium", "low"):
            tier_rows = [row for row in rows if row.get("kind") == tier]
            total = sum(int(row.get("length", 0)) for row in tier_rows)
            lines.append(f"- {tier.title()}: {len(tier_rows)} intervals, {total:,} bp")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_viewer(viewer_dir: Path, records: list[FastaRecord], intervals: dict[str, list[dict[str, object]]], genes: list[GeneRecord]) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    write_js_payloads(viewer_dir, records, intervals, genes)
    write_viewer_index(viewer_dir / "index.html")
    with resources.files("shadowisland_tool.assets.viewer").joinpath("app.js").open("rb") as src:
        (viewer_dir / "app.js").write_bytes(src.read())
    with resources.files("shadowisland_tool.assets.viewer").joinpath("styles.css").open("rb") as src:
        (viewer_dir / "styles.css").write_bytes(src.read())


def write_js_payloads(viewer_dir: Path, records: list[FastaRecord], intervals: dict[str, list[dict[str, object]]], genes: list[GeneRecord]) -> None:
    case_data = {
        "generatedFrom": "ShadowIsland Tool",
        "selectionRule": "User-submitted genomes; prediction tiers are generated from saved-weight inference and available GFF evidence.",
        "selectedCases": [{"dataset": "user", "accession": rec.accession} for rec in records],
        "cases": [],
    }
    genes_payload: dict[str, dict[str, object]] = {}
    gc_payload: dict[str, list[list[float]]] = {}
    genes_by_acc: dict[str, list[GeneRecord]] = {}
    for gene in genes:
        genes_by_acc.setdefault(gene.accession, []).append(gene)

    for rec in records:
        values = intervals.get(rec.accession, [])
        tracks = {
            "truth": [],
            "high": [row for row in values if row.get("kind") == "high"],
            "medium": [row for row in values if row.get("kind") == "medium"],
            "low": [row for row in values if row.get("kind") == "low"],
        }
        recommended = tracks["high"] + tracks["medium"]
        low_bp = sum(int(row.get("length", 0)) for row in tracks["low"])
        rec_bp = sum(int(row.get("length", 0)) for row in recommended)
        pred_bp = rec_bp + low_bp
        genome_len = len(rec.sequence)
        evidence_count = sum(int(row.get("n_evidence_classes_y", 0) or 0) for row in recommended)
        case_data["cases"].append(
            {
                "dataset": "user",
                "accession": rec.accession,
                "organism": rec.description,
                "genomeLength": genome_len,
                "genomeGcPct": gc_fraction(rec.sequence) * 100,
                "metrics": {
                    "truthCount": 0,
                    "highCount": len(tracks["high"]),
                    "mediumCount": len(tracks["medium"]),
                    "lowCount": len(tracks["low"]),
                    "truthBp": 0,
                    "recommendedBp": rec_bp,
                    "lowBp": low_bp,
                    "recommendedOverlapBp": pred_bp,
                    "recommendedRecall": evidence_count / max(len(recommended), 1),
                    "recommendedPrecision": rec_bp / max(pred_bp, 1),
                    "recommendedJaccard": rec_bp / max(genome_len, 1),
                    "lowBpFraction": low_bp / max(pred_bp, 1),
                    "selectionScore": sum(float(row.get("score", 0) or 0) for row in values) / max(len(values), 1),
                },
                "tracks": tracks,
            }
        )
        genes_payload[rec.accession] = {
            "length": genome_len,
            "genes": [
                [gene.start, gene.end, 1 if gene.strand == "+" else 0, category_code(gene.category)]
                for gene in sorted(genes_by_acc.get(rec.accession, []), key=lambda item: item.start)
            ],
        }
        gc_payload[rec.accession] = gc_windows(rec.sequence)

    (viewer_dir / "case_data.js").write_text("window.CASE_DATA = " + json.dumps(case_data, indent=2) + ";\n", encoding="utf-8")
    (viewer_dir / "case_data.json").write_text(json.dumps(case_data, indent=2), encoding="utf-8")
    (viewer_dir / "refseq_genes.js").write_text("window.REFSEQ_GENES = " + json.dumps(genes_payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    (viewer_dir / "refseq_genes.json").write_text(json.dumps(genes_payload, separators=(",", ":")), encoding="utf-8")
    (viewer_dir / "gc_windows.js").write_text("window.GC_WINDOWS = " + json.dumps(gc_payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    (viewer_dir / "gc_windows.json").write_text(json.dumps(gc_payload, separators=(",", ":")), encoding="utf-8")


def category_code(category: str) -> int:
    return {"other": 0, "mobility": 1, "virulence": 2, "resistance": 3, "phage": 4, "trna": 5}.get(category, 0)


def gc_windows(seq: str) -> list[list[float]]:
    rows = []
    for start0 in range(0, len(seq), GC_WINDOW_BP):
        frag = seq[start0 : start0 + GC_WINDOW_BP]
        rows.append([start0 + 1, min(start0 + GC_WINDOW_BP, len(seq)), round(gc_fraction(frag), 4)])
    return rows


def write_viewer_index(path: Path) -> None:
    path.write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ShadowIsland Evidence Viewer</title>
    <link rel="stylesheet" href="styles.css" />
  </head>
  <body>
    <main class="app-shell">
      <header class="topbar">
        <div>
          <p class="kicker">ShadowIsland result</p>
          <h1>Predicted genomic island evidence map</h1>
        </div>
        <div class="topbar-actions">
          <a class="text-link" href="../predicted_intervals.csv">Intervals</a>
        </div>
      </header>
      <section class="case-tabs" id="caseTabs" aria-label="Submitted genomes"></section>
      <section class="workspace">
        <aside class="side-panel">
          <section><h2 id="caseTitle">Case</h2><p id="caseSubtitle" class="muted"></p></section>
          <section class="metric-grid" id="metricGrid" aria-label="Case metrics"></section>
          <section><h3>Tracks</h3><div class="track-controls" id="trackControls"></div></section>
          <section><h3>Selected interval</h3><div id="selectionDetail" class="selection-detail muted">Hover or click an arc to inspect evidence.</div></section>
        </aside>
        <section class="viewer-panel">
          <div class="viewer-header">
            <div>
              <h2>Circular evidence map</h2>
              <p>Tracks show ShadowIsland prediction tiers, optional gene evidence categories, and GC deviation.</p>
            </div>
            <div class="legend" id="legend"></div>
          </div>
          <div class="circle-wrap"><svg id="circleMap" viewBox="0 0 820 820" role="img" aria-label="Circular genomic island map"></svg></div>
          <div class="linear-panel">
            <div class="linear-header">
              <div><h2>Linear focus</h2><p id="focusLabel">Predicted intervals across the selected genome.</p></div>
              <input id="focusSlider" type="range" min="0" max="1000" value="500" />
            </div>
            <svg id="linearMap" viewBox="0 0 980 250" role="img" aria-label="Linear interval focus"></svg>
          </div>
          <div class="table-panel">
            <div class="table-header"><h2>Interval table</h2><input id="searchBox" type="search" placeholder="Filter tier, evidence or coordinate" /></div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Track</th><th>Start</th><th>End</th><th>Length</th><th>Evidence score</th><th>Evidence label</th><th>Genes</th><th>Tags</th></tr></thead>
                <tbody id="intervalTable"></tbody>
              </table>
            </div>
          </div>
        </section>
      </section>
    </main>
    <script src="case_data.js"></script>
    <script src="refseq_genes.js"></script>
    <script src="gc_windows.js"></script>
    <script src="app.js"></script>
  </body>
</html>
""",
        encoding="utf-8",
    )

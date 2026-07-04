# Outputs

Each prediction run writes the following files.

| Path | Description |
| --- | --- |
| `window_probs.csv` | Per-window model scores. |
| `predicted_intervals.csv` | Predicted genomic island intervals. |
| `gene_catalog.csv` | Parsed GFF3 features, when a GFF3 file is supplied. |
| `region_summary.csv` | Evidence summary by confidence tier. |
| `functional_report.md` | Human-readable summary. |
| `provenance.json` | Tool/model version and input metadata. |
| `viewer/index.html` | Circular evidence viewer. |


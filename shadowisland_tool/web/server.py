from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..io import sanitize_filename
from ..pipeline import DEFAULT_MODEL_DIR, ShadowIslandPredictor
from ..scripts_support import check_model_bundle


TOOL_ROOT = Path(__file__).resolve().parents[2]
JOBS = TOOL_ROOT / "jobs"
WEB_STATIC = TOOL_ROOT / "web_static"

app = FastAPI(title="ShadowIsland Tool", version="0.1.0")
JOBS.mkdir(parents=True, exist_ok=True)
app.mount("/jobs", StaticFiles(directory=JOBS), name="jobs")


def job_paths(job_id: str) -> dict[str, Path]:
    root = JOBS / job_id
    return {
        "root": root,
        "input": root / "input",
        "results": root / "results",
        "status": root / "status.json",
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ShadowIsland Tool</title>
    <style>
      body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f4f5f7;color:#1c2530}
      main{width:min(980px,calc(100% - 32px));margin:0 auto;padding:32px 0}
      h1{font-size:clamp(2rem,4vw,3.5rem);line-height:1;margin:0 0 12px}
      .panel{background:#fff;border:1px solid #d7dde6;border-radius:8px;padding:18px;margin-top:18px}
      label{display:block;margin:14px 0;font-weight:700} input{display:block;margin-top:8px}
      button,a.button{display:inline-flex;align-items:center;min-height:38px;padding:0 14px;border-radius:8px;border:1px solid #1f5f9f;background:#2d6cb3;color:#fff;text-decoration:none;font-weight:700}
      code{background:#eef1f5;padding:2px 5px;border-radius:4px}
      #status{margin-top:14px;color:#5d6b7a}.links{display:flex;gap:10px;margin-top:14px}
    </style>
  </head>
  <body>
    <main>
      <p><strong>ShadowIsland standalone tool</strong></p>
      <h1>Upload a bacterial genome and generate evidence-viewer outputs.</h1>
      <section class="panel">
        <form id="form">
          <label>Genome FASTA<input name="fasta" type="file" required /></label>
          <label>Annotation GFF3 optional<input name="gff" type="file" /></label>
          <button type="submit">Run prediction</button>
        </form>
        <div id="status">No job submitted.</div>
        <div class="links"><a id="viewer" class="button" href="#" aria-disabled="true">Open viewer</a><a id="download" class="button" href="#" aria-disabled="true">Download</a></div>
      </section>
    </main>
    <script>
      const form=document.querySelector('#form'),status=document.querySelector('#status'),viewer=document.querySelector('#viewer'),download=document.querySelector('#download');
      let timer=null;
      form.addEventListener('submit',async e=>{e.preventDefault();clearInterval(timer);status.textContent='Uploading.';const r=await fetch('/api/jobs',{method:'POST',body:new FormData(form)});const j=await r.json();timer=setInterval(()=>poll(j.status_url),1200);poll(j.status_url);});
      async function poll(url){const r=await fetch(url);const j=await r.json();status.textContent=`${j.status}: ${j.message}`;if(j.viewer_url)viewer.href=j.viewer_url;if(j.download_url)download.href=j.download_url;if(j.status==='complete'||j.status==='failed')clearInterval(timer);}
    </script>
  </body>
</html>"""


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "model": check_model_bundle(DEFAULT_MODEL_DIR)})


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    fasta: UploadFile = File(...),
    gff: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    paths = job_paths(job_id)
    paths["input"].mkdir(parents=True, exist_ok=True)
    paths["results"].mkdir(parents=True, exist_ok=True)

    fasta_path = paths["input"] / sanitize_filename(fasta.filename or "genome.fasta")
    with fasta_path.open("wb") as handle:
        shutil.copyfileobj(fasta.file, handle)
    gff_path = None
    if gff and gff.filename:
        gff_path = paths["input"] / sanitize_filename(gff.filename)
        with gff_path.open("wb") as handle:
            shutil.copyfileobj(gff.file, handle)

    write_status(job_id, "queued", "Job queued.", viewer_url=f"/jobs/{job_id}/results/viewer/index.html")
    background_tasks.add_task(run_job, job_id, fasta_path, gff_path)
    return JSONResponse({"job_id": job_id, "status_url": f"/api/jobs/{job_id}"})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    status = job_paths(job_id)["status"]
    if not status.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    return JSONResponse(json.loads(status.read_text(encoding="utf-8")))


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    paths = job_paths(job_id)
    if not paths["root"].exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    archive = shutil.make_archive(str(paths["root"]), "zip", root_dir=paths["root"])
    return FileResponse(archive, filename=f"shadowisland_{job_id}.zip")


def run_job(job_id: str, fasta_path: Path, gff_path: Path | None) -> None:
    try:
        write_status(job_id, "running", "Running saved-weight prediction.")
        paths = job_paths(job_id)
        result = ShadowIslandPredictor.from_pretrained().predict_fasta(fasta_path, gff=gff_path, out_dir=paths["results"])
        write_status(
            job_id,
            "complete",
            "Complete.",
            viewer_url=f"/jobs/{job_id}/results/viewer/index.html",
            download_url=f"/api/jobs/{job_id}/download",
            n_records=len(result.records),
            n_intervals=result.n_intervals,
            n_windows=result.n_windows,
        )
    except Exception as exc:  # noqa: BLE001
        write_status(job_id, "failed", str(exc))


def write_status(job_id: str, status: str, message: str, **extra: object) -> None:
    paths = job_paths(job_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    payload = {"job_id": job_id, "status": status, "message": message, **extra}
    paths["status"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


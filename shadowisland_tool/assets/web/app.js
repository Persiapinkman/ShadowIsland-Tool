const form = document.querySelector("#jobForm");
const runButton = document.querySelector("#runButton");
const statusText = document.querySelector("#statusText");
const statusFill = document.querySelector("#statusFill");
const jobIdEl = document.querySelector("#jobId");
const recordCountEl = document.querySelector("#recordCount");
const intervalCountEl = document.querySelector("#intervalCount");
const viewerLink = document.querySelector("#viewerLink");
const downloadLink = document.querySelector("#downloadLink");

let pollTimer = null;

function setProgress(status) {
  const width = {
    queued: "18%",
    running: "62%",
    complete: "100%",
    failed: "100%",
  }[status] || "0%";
  statusFill.style.width = width;
  statusFill.dataset.status = status;
}

function enableLink(anchor, href) {
  anchor.href = href;
  anchor.classList.remove("disabled");
  anchor.setAttribute("aria-disabled", "false");
}

function resetLinks() {
  for (const anchor of [viewerLink, downloadLink]) {
    anchor.href = "#";
    anchor.classList.add("disabled");
    anchor.setAttribute("aria-disabled", "true");
  }
}

async function pollJob(statusUrl) {
  const response = await fetch(statusUrl);
  if (!response.ok) throw new Error(`Status request failed: ${response.status}`);
  const job = await response.json();
  jobIdEl.textContent = job.job_id || "-";
  recordCountEl.textContent = job.n_records ?? "-";
  intervalCountEl.textContent = job.n_intervals ?? "-";
  statusText.textContent = job.message || job.status;
  setProgress(job.status);

  if (job.viewer_url) enableLink(viewerLink, job.viewer_url);
  if (job.download_url) enableLink(downloadLink, job.download_url);

  if (job.status === "complete" || job.status === "failed") {
    clearInterval(pollTimer);
    pollTimer = null;
    runButton.disabled = false;
    runButton.textContent = "Run ShadowIsland prediction";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearInterval(pollTimer);
  resetLinks();
  setProgress("queued");
  statusText.textContent = "Uploading input files.";
  runButton.disabled = true;
  runButton.textContent = "Running...";

  const payload = new FormData(form);
  try {
    const response = await fetch("/api/jobs", { method: "POST", body: payload });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Upload failed: ${response.status}`);
    }
    const created = await response.json();
    jobIdEl.textContent = created.job_id;
    pollJob(created.status_url);
    pollTimer = setInterval(() => pollJob(created.status_url).catch(showError), 1200);
  } catch (error) {
    showError(error);
  }
});

function showError(error) {
  clearInterval(pollTimer);
  pollTimer = null;
  setProgress("failed");
  statusText.textContent = error.message || String(error);
  runButton.disabled = false;
  runButton.textContent = "Run ShadowIsland prediction";
}

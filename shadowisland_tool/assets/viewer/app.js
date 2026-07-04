const DATA = window.CASE_DATA;

const TRACKS = [
  { key: "truth", label: "Ground truth", group: "Reference", color: "var(--truth)", radius: 314, width: 18 },
  { key: "high", label: "High", group: "Recommended set", color: "var(--high)", radius: 288, width: 24 },
  { key: "medium", label: "Medium", group: "Recommended set", color: "var(--medium)", radius: 260, width: 24 },
  { key: "low", label: "Exploratory set", group: "Exploratory set", color: "var(--low)", radius: 234, width: 18 },
];

// Whole-genome RefSeq gene tracks (IslandViewer style): an outer "all CDS"
// density ring, then one dedicated ring per functional category.
const GENE_CATEGORIES = [
  { code: 1, key: "mobility", label: "Mobility", color: "var(--mobility)", radius: 200 },
  { code: 2, key: "virulence", label: "Virulence", color: "var(--virulence)", radius: 187 },
  { code: 3, key: "resistance", label: "Resistance", color: "var(--resistance)", radius: 174 },
  { code: 4, key: "phage", label: "Phage", color: "var(--phage)", radius: 161 },
  { code: 5, key: "trna", label: "tRNA", color: "var(--trna)", radius: 148 },
];
const CAT_BY_CODE = Object.fromEntries(GENE_CATEGORIES.map((c) => [c.code, c]));
const CDS_ALL_RADIUS = 214;

// GC deviation ring: a wide background band with a dashed baseline at the genome mean.
// Pulled toward the centre to fill the hub gap, with a taller step range so the
// deviation reads clearly across the whole plot.
const GC_BASE_RADIUS = 96;
const GC_BAND_WIDTH = 96;
const GC_SCALE = 4.6;
const GC_MAX_OFFSET = 50;
// Whole-genome GC content track: pixels of radial deflection per unit GC fraction
// deviation from the genome mean (IslandViewer-style continuous spikes).
const GC_WIN_PX_PER_FRAC = 230;

let activeCaseIndex = 0;
let selectedIntervalId = null;
let visibleTracks = new Set(TRACKS.map((track) => track.key));

const tabsEl = document.querySelector("#caseTabs");
const titleEl = document.querySelector("#caseTitle");
const subtitleEl = document.querySelector("#caseSubtitle");
const metricGridEl = document.querySelector("#metricGrid");
const trackControlsEl = document.querySelector("#trackControls");
const selectionDetailEl = document.querySelector("#selectionDetail");
const circleEl = document.querySelector("#circleMap");
const linearEl = document.querySelector("#linearMap");
const tableEl = document.querySelector("#intervalTable");
const legendEl = document.querySelector("#legend");
const focusSliderEl = document.querySelector("#focusSlider");
const focusLabelEl = document.querySelector("#focusLabel");
const searchBoxEl = document.querySelector("#searchBox");

function formatBp(value) {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} Mb`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)} kb`;
  return `${value} bp`;
}

function formatPct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function escapeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function polarToCartesian(cx, cy, radius, angleDeg) {
  const angleRad = ((angleDeg - 90) * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad),
  };
}

function describeArc(cx, cy, radius, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, radius, endAngle);
  const end = polarToCartesian(cx, cy, radius, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return [
    "M",
    start.x.toFixed(3),
    start.y.toFixed(3),
    "A",
    radius,
    radius,
    0,
    largeArcFlag,
    0,
    end.x.toFixed(3),
    end.y.toFixed(3),
  ].join(" ");
}

function angleForBp(bp, genomeLength) {
  return (bp / genomeLength) * 360;
}

function intervalAngle(interval, genomeLength) {
  const start = angleForBp(interval.start - 1, genomeLength);
  const end = Math.max(start + 0.18, angleForBp(interval.end, genomeLength));
  return { start, end };
}

function createSvgEl(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function getActiveCase() {
  return DATA.cases[activeCaseIndex];
}

function allIntervals(caseData) {
  return TRACKS.flatMap((track) =>
    (caseData.tracks[track.key] || []).map((interval) => ({
      ...interval,
      trackKey: track.key,
      trackLabel: track.label,
      trackColor: track.color,
    })),
  );
}

function getIntervalById(id) {
  if (!id) return null;
  return allIntervals(getActiveCase()).find((interval) => interval.id === id) || null;
}

function renderTabs() {
  tabsEl.innerHTML = DATA.cases
    .map((caseData, index) => {
      const active = index === activeCaseIndex ? " active" : "";
      const m = caseData.metrics;
      return `
        <button class="case-tab${active}" type="button" data-index="${index}">
          <strong>${escapeText(caseData.accession)} <span>${escapeText(caseData.dataset)}</span></strong>
          <span>${escapeText(caseData.organism)}</span>
          <span>Recall ${formatPct(m.recommendedRecall)} / Precision ${formatPct(m.recommendedPrecision)} / Exploratory ${m.lowCount}</span>
        </button>
      `;
    })
    .join("");

  tabsEl.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      activeCaseIndex = Number(button.dataset.index);
      selectedIntervalId = null;
      searchBoxEl.value = "";
      renderAll();
    });
  });
}

function renderHeader() {
  const caseData = getActiveCase();
  titleEl.textContent = `${caseData.dataset} / ${caseData.accession}`;
  const geneEntry = (window.REFSEQ_GENES || {})[caseData.accession];
  const geneNote = geneEntry && geneEntry.genes ? ` ${geneEntry.genes.length} RefSeq CDS.` : "";
  subtitleEl.textContent = `${caseData.organism}. Genome ${formatBp(caseData.genomeLength)}. GC ${caseData.genomeGcPct?.toFixed(2) ?? "NA"}%.${geneNote}`;
}

function renderMetrics() {
  const m = getActiveCase().metrics;
  const hasTruth = (getActiveCase().tracks.truth || []).length > 0;
  const metrics = [
    [hasTruth ? "Truth covered" : "Evidence support", formatPct(m.recommendedRecall)],
    [hasTruth ? "Recommended precision" : "Recommended share", formatPct(m.recommendedPrecision)],
    [hasTruth ? "Truth overlap" : "Predicted bases", formatBp(m.recommendedOverlapBp)],
    ["Exploratory burden", `${m.lowCount} / ${formatPct(m.lowBpFraction)}`],
    ["Recommended set", `${m.highCount + m.mediumCount}`],
    ["Selection score", m.selectionScore.toFixed(3)],
  ];
  metricGridEl.innerHTML = metrics
    .map(
      ([label, value]) => `
        <dl class="metric">
          <dt>${escapeText(label)}</dt>
          <dd>${escapeText(value)}</dd>
        </dl>
      `,
    )
    .join("");
}

function renderControls() {
  trackControlsEl.innerHTML = TRACKS.map(
    (track) => `
      <button class="track-toggle" type="button" data-track="${track.key}" aria-pressed="${visibleTracks.has(track.key)}">
        <span><span class="swatch" style="--track-color:${track.color}"></span> ${escapeText(track.label)}</span>
        <strong>${(getActiveCase().tracks[track.key] || []).length}</strong>
      </button>
    `,
  ).join("");

  trackControlsEl.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.track;
      if (visibleTracks.has(key)) visibleTracks.delete(key);
      else visibleTracks.add(key);
      renderAll(false);
    });
  });

  legendEl.innerHTML = [
    `
      <span class="legend-item">
        <span class="swatch recommend"></span>
        Recommended set (High + Medium)
      </span>
    `,
  ]
    .concat(
      TRACKS.map(
        (track) => `
          <span class="legend-item">
            <span class="swatch" style="--track-color:${track.color}"></span>
            ${escapeText(track.label)}
          </span>
        `,
      ),
      [
        `
          <span class="legend-item">
            <span class="swatch mini cds"></span>
            All RefSeq CDS (density)
          </span>
        `,
      ],
      GENE_CATEGORIES.map(
        (cat) => `
          <span class="legend-item">
            <span class="swatch mini" style="--track-color:${cat.color}"></span>
            ${escapeText(cat.label)}
          </span>
        `,
      ),
      [
        `
          <span class="legend-item">
            <span class="swatch split"></span>
            GC deviation
          </span>
        `,
      ],
    )
    .join("");
}

function renderCircle() {
  const caseData = getActiveCase();
  const cx = 410;
  const cy = 410;
  circleEl.innerHTML = "";

  renderScaffold(caseData, cx, cy);

  TRACKS.forEach((track) => {
    if (!visibleTracks.has(track.key)) return;
    for (const interval of caseData.tracks[track.key] || []) {
      const { start, end } = intervalAngle(interval, caseData.genomeLength);
      const path = createSvgEl("path", {
        class: `track-arc${interval.id === selectedIntervalId ? " selected" : ""}`,
        d: describeArc(cx, cy, track.radius, start, end),
        "stroke-width": track.width,
        "data-id": interval.id,
        tabindex: "0",
      });
      path.style.setProperty("--track-color", track.color);
      path.style.setProperty("--track-width", `${track.width}px`);
      path.addEventListener("mouseenter", () => showInterval(interval.id));
      path.addEventListener("focus", () => showInterval(interval.id));
      path.addEventListener("click", () => selectInterval(interval.id));
      circleEl.appendChild(path);
    }
  });

  renderRefseqGenes(caseData, cx, cy);
  // Prefer the whole-genome GC content track when sliding-window data is available
  // (IslandViewer-style continuous deviation); otherwise fall back to per-island wedges.
  if ((window.GC_WINDOWS || {})[caseData.accession]) {
    renderGcWindows(caseData, cx, cy);
  } else {
    renderGcCurve(caseData, cx, cy);
  }
}

function renderScaffold(caseData, cx, cy) {
  circleEl.appendChild(createSvgEl("circle", { class: "outer-band", cx, cy, r: 338 }));
  circleEl.appendChild(createSvgEl("circle", { class: "genome-ring", cx, cy, r: 330 }));
  circleEl.appendChild(createSvgEl("circle", { class: "inner-field", cx, cy, r: 220 }));
  circleEl.appendChild(
    createSvgEl("circle", { class: "gc-track", cx, cy, r: GC_BASE_RADIUS, "stroke-width": GC_BAND_WIDTH }),
  );
  circleEl.appendChild(createSvgEl("circle", { class: "gc-baseline", cx, cy, r: GC_BASE_RADIUS }));
  circleEl.appendChild(createSvgEl("circle", { class: "hub", cx, cy, r: 18 }));

  TRACKS.forEach((track) => {
    circleEl.appendChild(
      createSvgEl("circle", {
        class: "track-bed",
        cx,
        cy,
        r: track.radius,
        "stroke-width": track.width + 8,
      }),
    );
  });

  // Recommended set highlight band (High + Medium) framed between truth and exploratory.
  const recOuter = TRACKS[0].radius - TRACKS[0].width / 2 - 3;
  const recInner = TRACKS[3].radius + TRACKS[3].width / 2 + 3;
  const bandMid = (recOuter + recInner) / 2;
  const bandWidth = recOuter - recInner;
  circleEl.appendChild(
    createSvgEl("circle", {
      class: "recommend-band",
      cx,
      cy,
      r: bandMid,
      "stroke-width": bandWidth,
    }),
  );
  [recOuter, recInner].forEach((r) => {
    circleEl.appendChild(createSvgEl("circle", { class: "recommend-edge", cx, cy, r }));
  });

  // One baseline ring per gene track (all-CDS density + each functional category).
  const geneTracks = [
    { radius: CDS_ALL_RADIUS, label: "All CDS", color: "var(--muted)" },
    ...GENE_CATEGORIES.map((c) => ({ radius: c.radius, label: c.label, color: c.color })),
  ];
  geneTracks.forEach((t) => {
    circleEl.appendChild(createSvgEl("circle", { class: "cds-bed", cx, cy, r: t.radius }));
  });

  // IslandViewer-style radial gridlines: evenly spaced spokes from the hub out to
  // the genome ring so the eye can line features up across every track.
  const radialDivisions = 24;
  const majorFractions = [0, 0.25, 0.5, 0.75];
  for (let i = 0; i < radialDivisions; i += 1) {
    const fraction = i / radialDivisions;
    if (majorFractions.includes(fraction)) continue;
    const angle = fraction * 360;
    const p1 = polarToCartesian(cx, cy, 20, angle);
    const p2 = polarToCartesian(cx, cy, 330, angle);
    circleEl.appendChild(
      createSvgEl("line", {
        class: "radial-grid",
        x1: p1.x.toFixed(2),
        y1: p1.y.toFixed(2),
        x2: p2.x.toFixed(2),
        y2: p2.y.toFixed(2),
      }),
    );
  }

  majorFractions.forEach((fraction) => {
    const angle = fraction * 360;
    const p1 = polarToCartesian(cx, cy, 20, angle);
    const p2 = polarToCartesian(cx, cy, 338, angle);
    circleEl.appendChild(
      createSvgEl("line", {
        class: "spoke",
        x1: p1.x.toFixed(2),
        y1: p1.y.toFixed(2),
        x2: p2.x.toFixed(2),
        y2: p2.y.toFixed(2),
      }),
    );
  });

  const minorTicks = 96;
  for (let i = 0; i < minorTicks; i += 1) {
    const angle = (i / minorTicks) * 360;
    const major = i % 12 === 0;
    const p1 = polarToCartesian(cx, cy, 338, angle);
    const p2 = polarToCartesian(cx, cy, major ? 350 : 346, angle);
    circleEl.appendChild(
      createSvgEl("line", {
        class: major ? "ruler-tick major" : "ruler-tick",
        x1: p1.x.toFixed(2),
        y1: p1.y.toFixed(2),
        x2: p2.x.toFixed(2),
        y2: p2.y.toFixed(2),
      }),
    );
  }

  [0, 0.25, 0.5, 0.75].forEach((fraction) => {
    const angle = fraction * 360;
    const p = polarToCartesian(cx, cy, 368, angle);
    const text = createSvgEl("text", {
      class: "coord-label",
      x: p.x.toFixed(2),
      y: p.y.toFixed(2),
      "text-anchor": "middle",
      "dominant-baseline": "middle",
    });
    text.textContent = fraction === 0 ? "0" : formatBp(Math.round(caseData.genomeLength * fraction));
    circleEl.appendChild(text);
  });
}

function renderRefseqGenes(caseData, cx, cy) {
  const entry = (window.REFSEQ_GENES || {})[caseData.accession];
  if (!entry || !entry.genes) return;
  const genomeLength = caseData.genomeLength;

  // Outer ring: every RefSeq CDS as a faint dot (whole-genome gene density).
  const baseFrag = document.createDocumentFragment();
  for (const [start, end] of entry.genes) {
    const angle = angleForBp((start + end) / 2, genomeLength);
    const p = polarToCartesian(cx, cy, CDS_ALL_RADIUS, angle);
    baseFrag.appendChild(
      createSvgEl("circle", {
        class: "cds-dot",
        cx: p.x.toFixed(2),
        cy: p.y.toFixed(2),
        r: 1.1,
      }),
    );
  }
  circleEl.appendChild(baseFrag);

  // One dedicated track per functional category, coloured dots on its own ring.
  for (const [start, end, strand, cat] of entry.genes) {
    if (cat === 0) continue;
    const meta = CAT_BY_CODE[cat];
    if (!meta) continue;
    const angle = angleForBp((start + end) / 2, genomeLength);
    const p = polarToCartesian(cx, cy, meta.radius, angle);
    const dot = createSvgEl("circle", {
      class: "func-dot",
      cx: p.x.toFixed(2),
      cy: p.y.toFixed(2),
      r: 2.6,
    });
    dot.style.setProperty("--feature-color", meta.color);
    const title = createSvgEl("title");
    title.textContent = `${meta.label} | ${start.toLocaleString()}-${end.toLocaleString()} (${strand === 1 ? "+" : "-"})`;
    dot.appendChild(title);
    circleEl.appendChild(dot);
  }
}

function renderGcWindows(caseData, cx, cy) {
  const windows = (window.GC_WINDOWS || {})[caseData.accession];
  if (!windows || !windows.length) return;
  const genomeLength = caseData.genomeLength;
  const mean =
    caseData.genomeGcPct != null
      ? caseData.genomeGcPct / 100
      : windows.reduce((s, w) => s + w[2], 0) / windows.length;

  // Build one radial spike per window: outward when GC is above the genome mean,
  // inward when below. Two paths (pos / neg) keep it to two coloured DOM nodes.
  const posSeg = [];
  const negSeg = [];
  for (const [start, end, gc] of windows) {
    const angle = angleForBp((start + end) / 2, genomeLength);
    const dev = gc - mean;
    let offset = dev * GC_WIN_PX_PER_FRAC;
    offset = Math.max(-GC_MAX_OFFSET, Math.min(GC_MAX_OFFSET, offset));
    if (Math.abs(offset) < 0.05) continue;
    const base = polarToCartesian(cx, cy, GC_BASE_RADIUS, angle);
    const tip = polarToCartesian(cx, cy, GC_BASE_RADIUS + offset, angle);
    const seg = `M${base.x.toFixed(2)} ${base.y.toFixed(2)}L${tip.x.toFixed(2)} ${tip.y.toFixed(2)}`;
    (dev >= 0 ? posSeg : negSeg).push(seg);
  }

  if (negSeg.length) {
    const negPath = createSvgEl("path", { class: "gc-spike gc-spike-neg", d: negSeg.join("") });
    circleEl.appendChild(negPath);
  }
  if (posSeg.length) {
    const posPath = createSvgEl("path", { class: "gc-spike gc-spike-pos", d: posSeg.join("") });
    circleEl.appendChild(posPath);
  }
}

function renderGcCurve(caseData, cx, cy) {
  const predictionIntervals = ["high", "medium", "low"].flatMap((key) => caseData.tracks[key] || []);
  for (const interval of predictionIntervals) {
    if (interval.island_gc_delta_pct_vs_genome == null) continue;
    const angles = intervalAngle(interval, caseData.genomeLength);
    const delta = Number(interval.island_gc_delta_pct_vs_genome);
    const offset = Math.max(-GC_MAX_OFFSET, Math.min(GC_MAX_OFFSET, delta * GC_SCALE));
    const r = GC_BASE_RADIUS + offset;
    const color = delta >= 0 ? "var(--gc-pos)" : "var(--gc-neg)";

    // IslandViewer-style GC deviation: a solid wedge filled from the genome-mean
    // baseline out (GC above mean) or in (GC below mean).
    const baseStart = polarToCartesian(cx, cy, GC_BASE_RADIUS, angles.start);
    const topStart = polarToCartesian(cx, cy, r, angles.start);
    const topEnd = polarToCartesian(cx, cy, r, angles.end);
    const baseEnd = polarToCartesian(cx, cy, GC_BASE_RADIUS, angles.end);
    const largeArc = angles.end - angles.start <= 180 ? "0" : "1";
    const d = [
      "M", baseStart.x.toFixed(3), baseStart.y.toFixed(3),
      "L", topStart.x.toFixed(3), topStart.y.toFixed(3),
      "A", r, r, 0, largeArc, 1, topEnd.x.toFixed(3), topEnd.y.toFixed(3),
      "L", baseEnd.x.toFixed(3), baseEnd.y.toFixed(3),
      "A", GC_BASE_RADIUS, GC_BASE_RADIUS, 0, largeArc, 0, baseStart.x.toFixed(3), baseStart.y.toFixed(3),
      "Z",
    ].join(" ");

    const step = createSvgEl("path", {
      class: "gc-step",
      d,
      "data-id": interval.id,
    });
    step.style.setProperty("--gc-color", color);
    step.addEventListener("mouseenter", () => showInterval(interval.id));
    step.addEventListener("click", () => selectInterval(interval.id));
    circleEl.appendChild(step);
  }
}

function renderLinear() {
  const caseData = getActiveCase();
  const width = 980;
  const left = 60;
  const right = 940;
  const centerBp = (Number(focusSliderEl.value) / 1000) * caseData.genomeLength;
  const windowBp = Math.min(caseData.genomeLength, Math.max(450000, caseData.genomeLength * 0.28));
  const startBp = Math.max(1, Math.round(centerBp - windowBp / 2));
  const endBp = Math.min(caseData.genomeLength, Math.round(startBp + windowBp));
  const scale = (bp) => left + ((bp - startBp) / (endBp - startBp)) * (right - left);

  focusLabelEl.textContent = `${caseData.accession}: ${formatBp(startBp)} to ${formatBp(endBp)}`;
  linearEl.innerHTML = "";
  linearEl.appendChild(createSvgEl("line", { class: "linear-base", x1: left, x2: right, y1: 50, y2: 50 }));

  const axisStart = createSvgEl("text", { class: "axis-label", x: left, y: 30, "text-anchor": "start" });
  axisStart.textContent = formatBp(startBp);
  linearEl.appendChild(axisStart);

  const axisEnd = createSvgEl("text", { class: "axis-label", x: right, y: 30, "text-anchor": "end" });
  axisEnd.textContent = formatBp(endBp);
  linearEl.appendChild(axisEnd);

  TRACKS.forEach((track, index) => {
    const y = 82 + index * 38;
    const label = createSvgEl("text", { class: "axis-label", x: 18, y: y + 15 });
    label.textContent = track.label;
    linearEl.appendChild(label);

    if (!visibleTracks.has(track.key)) return;
    for (const interval of caseData.tracks[track.key] || []) {
      if (interval.end < startBp || interval.start > endBp) continue;
      const x = Math.max(left, scale(interval.start));
      const x2 = Math.min(right, scale(interval.end));
      const rect = createSvgEl("rect", {
        class: "linear-interval",
        x: x.toFixed(2),
        y,
        width: Math.max(3, x2 - x).toFixed(2),
        height: track.key === "truth" ? 18 : 22,
        fill: track.color,
        opacity: interval.id === selectedIntervalId ? 1 : 0.82,
        "data-id": interval.id,
      });
      rect.addEventListener("mouseenter", () => showInterval(interval.id));
      rect.addEventListener("click", () => selectInterval(interval.id));
      linearEl.appendChild(rect);
    }
  });
}

function renderTable() {
  const query = searchBoxEl.value.trim().toLowerCase();
  const rows = allIntervals(getActiveCase()).filter((interval) => {
    if (!visibleTracks.has(interval.trackKey)) return false;
    if (!query) return true;
    const haystack = [
      interval.trackLabel,
      interval.start,
      interval.end,
      interval.conf_tags,
      interval.evidence_label,
      interval.conf_evidence_label,
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });

  tableEl.innerHTML = rows
    .map((interval) => {
      const refseq = interval.refseq_n_genes == null ? "NA" : `${interval.refseq_n_genes} genes`;
      const overlap =
        interval.trackKey === "truth"
          ? "reference"
          : interval.score != null
            ? Number(interval.score).toFixed(3)
            : `${formatBp(interval.truth_overlap_bp || 0)} (${formatPct(interval.truth_overlap_frac || 0)})`;
      return `
        <tr class="${interval.id === selectedIntervalId ? "selected-row" : ""}" data-id="${escapeText(interval.id)}">
          <td><span class="tier-pill" style="--tier-color:${interval.trackColor}">${escapeText(interval.trackLabel)}</span></td>
          <td>${interval.start.toLocaleString()}</td>
          <td>${interval.end.toLocaleString()}</td>
          <td>${formatBp(interval.length)}</td>
          <td>${escapeText(overlap)}</td>
          <td>${escapeText(interval.conf_evidence_label || interval.evidence_label || "NA")}</td>
          <td>${escapeText(refseq)}</td>
          <td>${escapeText(interval.conf_tags || "")}</td>
        </tr>
      `;
    })
    .join("");

  tableEl.querySelectorAll("tr").forEach((row) => {
    row.addEventListener("mouseenter", () => showInterval(row.dataset.id));
    row.addEventListener("click", () => selectInterval(row.dataset.id));
  });
}

function intervalDetailHtml(interval) {
  if (!interval) {
    return "Hover or click an arc to inspect evidence.";
  }

  const overlap =
    interval.trackKey === "truth"
      ? "Curated positive GI"
      : interval.score != null
        ? `Prediction score ${Number(interval.score).toFixed(3)}`
        : `${formatBp(interval.truth_overlap_bp || 0)} (${formatPct(interval.truth_overlap_frac || 0)})`;
  const refseq =
    interval.refseq_n_genes == null
      ? "NA"
      : `${interval.refseq_n_genes} genes, mobility ${interval.refseq_n_mobility ?? 0}, virulence ${
          interval.refseq_n_virulence ?? 0
        }, resistance ${interval.refseq_n_resistance ?? 0}, phage ${interval.refseq_n_phage ?? 0}`;
  const gc =
    interval.island_gc_delta_pct_vs_genome == null
      ? "NA"
      : `${Number(interval.island_gc_delta_pct_vs_genome).toFixed(2)} pct point`;

  return `
    <strong>${escapeText(interval.trackLabel)} interval</strong>
    <dl>
      <dt>Coordinates</dt><dd>${interval.start.toLocaleString()}-${interval.end.toLocaleString()}</dd>
      <dt>Length</dt><dd>${formatBp(interval.length)}</dd>
      <dt>Truth overlap</dt><dd>${escapeText(overlap)}</dd>
      <dt>Evidence label</dt><dd>${escapeText(interval.conf_evidence_label || interval.evidence_label || "NA")}</dd>
      <dt>RefSeq</dt><dd>${escapeText(refseq)}</dd>
      <dt>GC delta</dt><dd>${escapeText(gc)}</dd>
      <dt>Tags</dt><dd>${escapeText(interval.conf_tags || "NA")}</dd>
    </dl>
  `;
}

function showInterval(id) {
  const interval = getIntervalById(id);
  selectionDetailEl.innerHTML = intervalDetailHtml(interval);
}

function selectInterval(id) {
  selectedIntervalId = id;
  renderCircle();
  renderLinear();
  renderTable();
  showInterval(id);
}

function renderAll(resetControls = true) {
  renderTabs();
  renderHeader();
  renderMetrics();
  if (resetControls) renderControls();
  else renderControls();
  renderCircle();
  renderLinear();
  renderTable();
  showInterval(selectedIntervalId);
}

focusSliderEl.addEventListener("input", renderLinear);
searchBoxEl.addEventListener("input", renderTable);

renderAll();

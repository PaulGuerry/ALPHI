// ALPHI web interface: reads the patient file, starts the worker, shows the results.
const $ = (id) => document.getElementById(id);
const COLORS = ["#d4842e", "#2e8b6f", "#1c5cab", "#8e5bd1", "#c4495a", "#5f8f2a", "#b0892a", "#4a90b8", "#7a6a5c", "#d16aa3"];
let patients = null, zipBlob = null, lastResult = null;

const worker = new Worker("worker.js");
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") $("status").textContent = m.text;
  else if (m.type === "ready") { $("status").textContent = "Ready"; updateRun(); }
  else if (m.type === "log") { const l = $("log"); l.textContent += m.text + "\n"; l.scrollTop = l.scrollHeight; }
  else if (m.type === "error") { $("status").innerHTML = `<span class="warn">${escapeHtml(m.text)}</span>`; running = false; updateRun(); }
  else if (m.type === "done") { running = false; updateRun(); show(m.result, m.zip); }
};
let workerReady = false, running = false;
worker.addEventListener("message", (e) => { if (e.data.type === "ready") workerReady = true; });

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

// ---------- reading the patient file ----------
function parseTable(text) {
  const sep = text.split("\n")[0].includes("\t") ? "\t" : ",";
  const rows = [];
  let row = [], cell = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') q = false; else cell += c; }
    else if (c === '"') q = true;
    else if (c === sep) { row.push(cell); cell = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cell); rows.push(row); row = []; cell = ""; }
    else cell += c;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const head = rows.shift().map((h) => h.trim().toLowerCase());
  const col = (names) => head.findIndex((h) => names.includes(h));
  const iId = col(["id", "patient", "patient_id"]), iH = col(["hpo", "hpo_terms", "symptoms", "terms"]);
  const iG = col(["gene", "genes"]), iS = col(["status", "alivedead", "alive_dead"]), iA = col(["age", "alivedeadage"]);
  if (iId < 0 || iH < 0) throw new Error("the table needs columns 'id' and 'hpo'");
  return rows.filter((r) => r.length > 1 && r[iId]).map((r) => {
    const codes = r[iH].split(/[;,|\s]+/).map((s) => s.trim().replace(/^HP:/i, "")).filter((s) => /^\d{7}$/.test(s));
    const p = { id: r[iId].trim(), gene: iG >= 0 && r[iG].trim() ? r[iG].split(/[;,]/).map((g) => g.trim()) : ["COHORT"], symptoms: codes };
    if (iS >= 0 && r[iS]) p.alivedead = r[iS].trim();
    if (iA >= 0 && r[iA]) p.alivedeadage = parseFloat(r[iA]);
    return p;
  });
}

$("file").addEventListener("change", async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const text = await f.text();
  try {
    if (/\.json$/i.test(f.name)) {
      const d = JSON.parse(text);
      patients = Array.isArray(d) ? d : d.patients;
      if (!Array.isArray(patients)) throw new Error("expected a list of patients");
    } else patients = parseTable(text);
  } catch (err) {
    patients = null;
    $("fileinfo").innerHTML = `<p class="warn">Could not read the file: ${escapeHtml(err.message)}</p>`;
    updateRun(); return;
  }
  const genes = {};
  for (const p of patients) for (const g of [].concat(p.gene || [])) genes[g] = (genes[g] || 0) + 1;
  const sorted = Object.entries(genes).sort((a, b) => b[1] - a[1]);
  $("fileinfo").innerHTML = `<p>${patients.length} patients read. Genes to include:</p><div class="genes">` +
    sorted.map(([g, n], i) => `<label><input type="checkbox" class="gene" value="${escapeHtml(g)}" ${i === 0 ? "checked" : ""}> ${escapeHtml(g)} <span class="muted">(${n})</span></label>`).join("") + "</div>";
  document.querySelectorAll(".gene").forEach((c) => c.addEventListener("change", updateRun));
  updateRun();
});

function selectedGenes() { return [...document.querySelectorAll(".gene:checked")].map((c) => c.value); }
function updateRun() { $("run").disabled = !(workerReady && patients && selectedGenes().length && !running); }

$("run").addEventListener("click", () => {
  running = true; updateRun();
  $("log").textContent = ""; $("results").hidden = true;
  worker.postMessage({
    type: "run", patients: JSON.stringify(patients),
    options: {
      genes: selectedGenes(), min_symptoms: +$("minsym").value, stability: +$("stab").value,
      lca: $("lca").checked, exclude_uncertain: $("uncertain").checked, excluded_doi: $("doi").value.trim(),
    },
  });
});

// ---------- results ----------
function show(r, zip) {
  lastResult = r;
  zipBlob = new Blob([zip], { type: "application/zip" });
  const s = r.summary;
  const lca = s.lca && s.lca.decision ? ` Latent class check: ${escapeHtml(s.lca.decision)}.` : "";
  $("summary").innerHTML = `${s.n_patients} patients, <b>${s.k} group${s.k === 1 ? "" : "s"}</b> (${Math.round(r.seconds)} s).${lca}`;
  $("status").textContent = "Done";
  drawPlot(r.points);
  $("cards").innerHTML = s.clusters.map((c) => `
    <div class="card">
      <h3><span class="dot" style="background:${color(c.cluster)}"></span>Group ${c.cluster + 1} <span class="muted">· ${c.n} patients</span></h3>
      ${c.stability != null ? `<div class="muted" style="font-size:13px">Stability (mean Jaccard): ${c.stability.toFixed(2)}</div>` : ""}
      ${c.few_feature_flag ? `<div class="warn" style="font-size:13px">Separated by one or two symptoms only</div>` : ""}
      <table><tr><th>Most distinguishing terms</th><th class="n">group</th><th class="n">others</th></tr>
      ${c.terms.map((t) => `<tr><td title="${t.code}">${escapeHtml(t.label || t.code)}</td><td class="n">${t.in}%</td><td class="n">${t.rest}%</td></tr>`).join("")}
      </table>
    </div>`).join("");
  $("results").hidden = false;
}
function color(c) { return c < 0 ? "#9a9a95" : COLORS[c % COLORS.length]; }

function drawPlot(points) {
  const W = 640, H = 420, m = 16;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const sx = (x) => m + (x - x0) / (x1 - x0 || 1) * (W - 2 * m), sy = (y) => H - m - (y - y0) / (y1 - y0 || 1) * (H - 2 * m);
  $("plot").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Patients coloured by group">` +
    points.map((p, i) => `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="5" fill="${color(p.cluster)}" stroke="var(--panel)" stroke-width="1" data-i="${i}"/>`).join("") + "</svg>";
  const tip = $("tip");
  $("plot").querySelectorAll("circle").forEach((c) => {
    c.addEventListener("mousemove", (ev) => {
      const p = points[+c.dataset.i];
      tip.textContent = `${p.id} · ${p.cluster < 0 ? "unassigned" : "group " + (p.cluster + 1)}`;
      tip.style.display = "block"; tip.style.left = ev.clientX + 12 + "px"; tip.style.top = ev.clientY + 12 + "px";
    });
    c.addEventListener("mouseleave", () => (tip.style.display = "none"));
  });
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
$("dlzip").addEventListener("click", () => zipBlob && download(zipBlob, "alphi_results.zip"));
$("dlcsv").addEventListener("click", () => {
  if (!lastResult) return;
  const rows = ["id,group"].concat(lastResult.points.map((p) => `"${p.id.replace(/"/g, '""')}",${p.cluster < 0 ? "" : p.cluster + 1}`));
  download(new Blob([rows.join("\n")], { type: "text/csv" }), "alphi_groups.csv");
});

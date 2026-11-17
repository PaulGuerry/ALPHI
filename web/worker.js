// ALPHI web worker: loads Pyodide and the ALPHI Python files, runs the pipeline off the main thread.
// Nothing is sent anywhere: the patient data stay in this browser tab.
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE_URL + "pyodide.js");
importScripts("js/umap_sgd.js");
self.umapOptimizeLayout = optimizeLayout;      // called by py/umap_lite.py

let py = null;
const status = (text) => postMessage({ type: "status", text });

async function init() {
  status("Loading Python (Pyodide)…");
  py = await loadPyodide({ indexURL: PYODIDE_URL });
  py.setStdout({ batched: (s) => postMessage({ type: "log", text: s }) });
  py.setStderr({ batched: (s) => postMessage({ type: "log", text: s }) });
  status("Loading numpy, scipy, scikit-learn, pandas…");
  await py.loadPackage(["numpy", "scipy", "scikit-learn", "pandas", "micropip"]);
  try { await py.pyimport("micropip").install("tabulate"); } catch (e) { /* fallback in py/shims */ }
  status("Loading ALPHI and the HPO tree…");
  const files = await (await fetch("manifest.json")).json();
  for (const f of files) {
    const buf = new Uint8Array(await (await fetch(f)).arrayBuffer());
    py.FS.mkdirTree("/alphi/" + f.split("/").slice(0, -1).join("/"));
    py.FS.writeFile("/alphi/" + f, buf);
  }
  py.runPython('import sys; sys.path[:0] = ["/alphi/py/shims", "/alphi/py", "/alphi/core"]');
  status("Ready");
  postMessage({ type: "ready" });
}
const ready = init().catch((e) => postMessage({ type: "error", text: "Could not load: " + e }));

onmessage = async (e) => {
  if (e.data.type !== "run") return;
  await ready;
  try {
    status("Running ALPHI…");
    py.FS.mkdirTree("/input");
    py.FS.writeFile("/input/patients.json", e.data.patients);
    const opts = Object.assign({ patient_file: "/input/patients.json", workdir: "/work" }, e.data.options);
    py.globals.set("ALPHI_OPTIONS", JSON.stringify(opts));
    const out = py.runPython("import alphi_runner; alphi_runner.run(ALPHI_OPTIONS)");
    const result = JSON.parse(out);
    const zip = py.FS.readFile(result.zip);
    postMessage({ type: "done", result, zip }, [zip.buffer]);
  } catch (err) {
    postMessage({ type: "error", text: String(err) });
  }
};

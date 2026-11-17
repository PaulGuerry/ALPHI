// UMAP layout optimisation (stochastic gradient descent), port of umap-learn's
// optimize_layout_euclidean (umap/layouts.py, euclidean output, no densMAP).
// Sequential, as umap-learn with a fixed random_state. Works in the browser (called from
// Pyodide) and in Node (tests). The embedding is modified in place.
//
// embedding: Float32Array(nVertices * dim), row-major
// head, tail: Int32Array(nEdges); epochsPerSample: Float64Array(nEdges) (-1 = never sampled)
// seeds: Uint32Array(nVertices * 3), Tausworthe state of each head vertex
function tauRandInt(s, o) {
  s[o] = ((((s[o] & 0xFFFFFFFE) << 12) >>> 0) ^ ((((s[o] << 13) >>> 0) ^ s[o]) >>> 19)) >>> 0;
  s[o + 1] = ((((s[o + 1] & 0xFFFFFFF8) << 4) >>> 0) ^ ((((s[o + 1] << 2) >>> 0) ^ s[o + 1]) >>> 25)) >>> 0;
  s[o + 2] = ((((s[o + 2] & 0xFFFFFFF0) << 17) >>> 0) ^ ((((s[o + 2] << 3) >>> 0) ^ s[o + 2]) >>> 11)) >>> 0;
  return (s[o] ^ s[o + 1] ^ s[o + 2]) >>> 0;
}

function clip(v) { return v > 4.0 ? 4.0 : (v < -4.0 ? -4.0 : v); }

function optimizeLayout(embedding, head, tail, epochsPerSample, nVertices, dim, nEpochs,
                        a, b, seeds, gamma = 1.0, initialAlpha = 1.0, negativeSampleRate = 5) {
  const nEdges = head.length;
  const epochsPerNeg = new Float64Array(nEdges);
  const nextNeg = new Float64Array(nEdges);
  const nextSample = new Float64Array(nEdges);
  for (let i = 0; i < nEdges; i++) {
    epochsPerNeg[i] = epochsPerSample[i] / negativeSampleRate;
    nextNeg[i] = epochsPerNeg[i];
    nextSample[i] = epochsPerSample[i];
  }
  let alpha = initialAlpha;
  for (let n = 0; n < nEpochs; n++) {
    for (let i = 0; i < nEdges; i++) {
      if (epochsPerSample[i] <= 0 || nextSample[i] > n) continue;
      const j = head[i], k = tail[i];
      const cj = j * dim, ck = k * dim;
      let d2 = 0.0;
      for (let d = 0; d < dim; d++) { const diff = embedding[cj + d] - embedding[ck + d]; d2 += diff * diff; }
      let gc = 0.0;
      if (d2 > 0.0) gc = (-2.0 * a * b * Math.pow(d2, b - 1.0)) / (a * Math.pow(d2, b) + 1.0);
      for (let d = 0; d < dim; d++) {
        const g = clip(gc * (embedding[cj + d] - embedding[ck + d]));
        embedding[cj + d] += g * alpha;
        embedding[ck + d] -= g * alpha;
      }
      nextSample[i] += epochsPerSample[i];
      const nNeg = Math.floor((n - nextNeg[i]) / epochsPerNeg[i]);
      for (let p = 0; p < nNeg; p++) {
        const kk = tauRandInt(seeds, j * 3) % nVertices;
        const co = kk * dim;
        let e2 = 0.0;
        for (let d = 0; d < dim; d++) { const diff = embedding[cj + d] - embedding[co + d]; e2 += diff * diff; }
        let gn;
        if (e2 > 0.0) gn = (2.0 * gamma * b) / ((0.001 + e2) * (a * Math.pow(e2, b) + 1.0));
        else if (j === kk) continue;
        else gn = 0.0;
        if (gn > 0.0) {
          for (let d = 0; d < dim; d++) embedding[cj + d] += clip(gn * (embedding[cj + d] - embedding[co + d])) * alpha;
        }
      }
      nextNeg[i] += nNeg * epochsPerNeg[i];
    }
    alpha = initialAlpha * (1.0 - (n / nEpochs));
  }
  return embedding;
}

if (typeof module !== "undefined") {
  module.exports = { optimizeLayout, tauRandInt };
  // Node CLI used by the CPython tests: node umap_sgd.js <request.json>
  if (require.main === module) {
    const fs = require("fs");
    const req = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
    const rd = (f, T) => new T(fs.readFileSync(f).buffer.slice(0));
    const emb = rd(req.embedding, Float32Array);
    optimizeLayout(emb, rd(req.head, Int32Array), rd(req.tail, Int32Array), rd(req.eps, Float64Array),
                   req.n_vertices, req.dim, req.n_epochs, req.a, req.b, rd(req.seeds, Uint32Array),
                   req.gamma, req.initial_alpha, req.negative_sample_rate);
    fs.writeFileSync(req.out, Buffer.from(emb.buffer));
  }
}

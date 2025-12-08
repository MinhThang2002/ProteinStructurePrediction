import json
import sidechainnet as scn
import torch
from sidechainnet.structure.structure import inverse_trig_transform


def build_visualizable_structures(model, data, config, device, sample_idx=0):
    """Build visualizable structures for the requested sample_idx across dataloader."""
    with torch.no_grad():
        offset = sample_idx
        for batch in data:
            bsz = batch.int_seqs.shape[0]
            if offset >= bsz:
                offset -= bsz
                continue
            if config.mode == "seqs":
                model_input = batch.int_seqs.to(device)
            elif config.mode == "pssms":
                model_input = batch.seq_evo_sec.to(device)
            mask_ = batch.msks.to(device)
            predicted_angles_sincos = model(model_input, mask=mask_)
            predicted_angles = inverse_trig_transform(predicted_angles_sincos)

            sb_pred = scn.BatchedStructureBuilder(batch.int_seqs, predicted_angles.cpu())

            true_angles = batch.angs.clone()
            true_angles[~torch.isfinite(true_angles)] = 0
            try:
                sb_true = scn.BatchedStructureBuilder(batch.int_seqs, true_angles.cpu())
            except ValueError:
                gt_crds = batch.crds.clone()
                gt_crds[~torch.isfinite(gt_crds)] = 0
                sb_true = scn.BatchedStructureBuilder(batch.int_seqs, gt_crds.cpu())
            return sb_pred, sb_true, offset
    raise IndexError(f"sample_idx {sample_idx} is out of range for the dataloader.")


def plot_protein(exp1, exp2, html_path=None, show=True, label_pred="Predicted", label_true="Ground Truth"):
    """Visualize two PDBs in clipped panes. Writes a self-contained HTML that loads 3Dmol.js."""
    pdb1 = open(exp1, 'r').read()
    pdb2 = open(exp2, 'r').read()
    pdb1_js = json.dumps(pdb1)
    pdb2_js = json.dumps(pdb2)
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script src="https://3dmol.org/build/3Dmol.js"></script>
  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: Arial, sans-serif;
      background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 50%, #f8fafc 100%);
    }}
    #container {{
      width: 100vw;
      height: 100vh;
      position: relative;
    }}
    .pane {{
      position: absolute;
      top: 6%;
      bottom: 6%;
      width: 42%;
      overflow: hidden;
      border: 1.5px solid #d7dce2;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.04), 0 12px 28px rgba(0,0,0,0.06);
      border-radius: 10px;
      background: white;
    }}
    #pane-left {{ left: 6%; }}
    #pane-right {{ right: 6%; }}
    #controls {{
      position: absolute;
      bottom: 12px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 6px;
      align-items: center;
      z-index: 12;
      background: rgba(255,255,255,0.92);
      padding: 8px 10px;
      border-radius: 10px;
      box-shadow: 0 4px 10px rgba(0,0,0,0.12);
    }}
    #controls button,
    #controls select {{
      padding: 7px 10px;
      border: 1px solid #d0d7e2;
      border-radius: 6px;
      background: #fff;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      cursor: pointer;
      font-size: 12px;
    }}
    #controls button:hover,
    #controls select:hover {{
      border-color: #9fb3d8;
    }}
    .pane-title {{
      position: absolute;
      top: 12px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255,255,255,0.9);
      padding: 4px 10px;
      border-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
      font-size: 12px;
      font-weight: 600;
      color: #1f2937;
      z-index: 12;
      border: 1px solid #e2e8f0;
    }}
  </style>
</head>
<body>
  <div id="container">
    <div id="pane-left" class="pane">
      <div class="pane-title">{label_pred}</div>
    </div>
    <div id="pane-right" class="pane">
      <div class="pane-title">{label_true}</div>
    </div>
  </div>
  <div id="controls">
    <button id="prevBtn">&#8592; Prev</button>
    <button id="nextBtn">Next &#8594;</button>
    <button id="resetBtn">Reset</button>
    <select id="idxSelect">
      <option value="">Jump idx</option>
      <option value="0">0</option>
      <option value="1">1</option>
      <option value="2">2</option>
      <option value="3">3</option>
      <option value="4">4</option>
      <option value="5">5</option>
      <option value="6">6</option>
      <option value="7">7</option>
      <option value="8">8</option>
      <option value="9">9</option>
      <option value="10">10</option>
      <option value="11">11</option>
      <option value="12">12</option>
      <option value="13">13</option>
      <option value="14">14</option>
      <option value="15">15</option>
      <option value="16">16</option>
      <option value="17">17</option>
      <option value="18">18</option>
      <option value="19">19</option>
    </select>
  </div>
  <script>
    const pdb1 = {pdb1_js};
    const pdb2 = {pdb2_js};
    const config = {{backgroundColor: "white"}};
    const viewerLeft = $3Dmol.createViewer("pane-left", config);
    const viewerRight = $3Dmol.createViewer("pane-right", config);
    viewerLeft.addModel(pdb1, "pdb");
    viewerRight.addModel(pdb2, "pdb");
    viewerLeft.setStyle({{"cartoon": {{"color": "spectrum", "opacity": 0.9}}}});
    viewerRight.setStyle({{"cartoon": {{"color": "spectrum", "opacity": 1.0, "thickness": 1.0, "arrows": true}}}});
    viewerLeft.zoomTo(); viewerRight.zoomTo();
    const initialZoom = 0.6;
    viewerLeft.zoom(initialZoom); viewerRight.zoom(initialZoom);

    const clampZoom = (v) => {{
      if (!v || typeof v.getZoom !== 'function' || typeof v.zoom !== 'function') return;
      const minZ = 0.5, maxZ = 1.0;
      const z = v.getZoom();
      if (z < minZ) v.zoom(minZ / z);
      if (z > maxZ) v.zoom(maxZ / z);
    }};
    const clampAll = () => {{ clampZoom(viewerLeft); clampZoom(viewerRight); viewerLeft.render(); viewerRight.render(); }};
    setInterval(clampAll, 200);
    document.getElementById("pane-left").addEventListener('wheel', clampAll, {{passive:true}});
    document.getElementById("pane-right").addEventListener('wheel', clampAll, {{passive:true}});

    document.getElementById("resetBtn").onclick = () => {{ viewerLeft.zoomTo(); viewerRight.zoomTo(); viewerLeft.render(); viewerRight.render(); }};

    const getCurrentIdx = () => {{
      const match = window.location.pathname.match(/(\\d+)_compare\\.html$/);
      if (match) return parseInt(match[1], 10);
      const sel = document.getElementById("idxSelect").value;
      return sel ? parseInt(sel, 10) : 0;
    }};
    const jumpToIdx = (val) => {{
      if (val === "" || val === null || val === undefined) return;
      window.location.href = `${{val}}_compare.html`;
    }};
    document.getElementById("idxSelect").onchange = (e) => jumpToIdx(e.target.value);
    // Sync dropdown with current idx on load
    const currentIdx = getCurrentIdx();
    const selectEl = document.getElementById("idxSelect");
    if (selectEl) selectEl.value = String(currentIdx);
    document.getElementById("prevBtn").onclick = () => {{
      const current = getCurrentIdx();
      jumpToIdx(Math.max(0, current - 1));
    }};
    document.getElementById("nextBtn").onclick = () => {{
      const current = getCurrentIdx();
      jumpToIdx(Math.min(10, current + 1));
    }};
  </script>
</body>
</html>
"""
    if html_path:
        with open(html_path, 'w') as f:
            f.write(html)
    if show:
        try:
            import webbrowser, os
            webbrowser.open('file://' + os.path.realpath(html_path or 'plot.html'))
        except Exception:
            pass

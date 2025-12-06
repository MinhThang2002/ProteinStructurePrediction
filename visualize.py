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
            predicted_angles_sincos = model(model_input, mask = mask_)
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

def plot_protein(exp1, exp2, html_path=None, show=True):
    """Visualize two PDBs side-by-side. Writes a self-contained HTML that loads 3Dmol.js."""
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
    }}
    #container {{
      width: 100vw;
      height: 100vh;
    }}
  </style>
</head>
<body>
  <div id="container"></div>
  <script>
    const pdb1 = {pdb1_js};
    const pdb2 = {pdb2_js};
    let element = document.getElementById("container");
    let config = {{backgroundColor: "white"}};
    let viewergrid = $3Dmol.createViewerGrid($(element), {{rows: 1, cols: 2, control_all: true}}, config);
    viewergrid[0][0].addModel(pdb1, "pdb");
    viewergrid[0][1].addModel(pdb2, "pdb");
    viewergrid[0][0].setStyle({{"cartoon": {{"color": "spectrum"}}}});
    viewergrid[0][1].setStyle({{"cartoon": {{"color": "spectrum"}}}});
    viewergrid[0][0].zoomTo();
    viewergrid[0][1].zoomTo();
    viewergrid[0][0].render();
    viewergrid[0][1].render();
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

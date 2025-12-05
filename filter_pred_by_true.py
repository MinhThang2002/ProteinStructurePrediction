"""
Filter a predicted PDB to only the residues present in the reference PDB.

Usage:
    python filter_pred_by_true.py --true plots/0_true.pdb --pred plots/0_pred.pdb \
        --out plots/0_pred_masked.pdb

This keeps only ATOM/HETATM records whose (chain, resseq, icode) exist in the
reference file. Other records (REMARK/SEQRES/TER/END) are copied unchanged.
"""
import argparse
from typing import Set, Tuple


Key = Tuple[str, int, str]


def parse_key(line: str) -> Key:
    """Extract (chain, resseq, icode) from a PDB ATOM/HETATM line."""
    chain = line[21].strip() or " "        # column 22 in 1-based PDB indexing
    resseq = int(line[22:26])              # columns 23-26
    icode = line[26].strip() or ""         # column 27 insertion code
    return chain, resseq, icode


def residues_with_ca(path: str) -> Set[Key]:
    """Return residue keys that contain a CA atom in a PDB file."""
    keys: Set[Key] = set()
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                keys.add(parse_key(line))
    return keys


def filter_pred(true_path: str, pred_path: str, out_path: str) -> None:
    """Write a filtered prediction PDB containing only residues with CA in true."""
    keep_keys = residues_with_ca(true_path)
    kept = 0
    total = 0
    with open(pred_path, "r") as src, open(out_path, "w") as dst:
        for line in src:
            if line.startswith(("ATOM", "HETATM")):
                total += 1
                if parse_key(line) in keep_keys:
                    dst.write(line)
                    kept += 1
            else:
                dst.write(line)
    print(f"Kept {kept}/{total} ATOM/HETATM records -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter predicted PDB to residues with coords in true PDB.")
    parser.add_argument("--true", required=True, help="Reference PDB with known coords.")
    parser.add_argument("--pred", required=True, help="Predicted PDB to be filtered.")
    parser.add_argument("--out", required=True, help="Output filtered PDB path.")
    args = parser.parse_args()
    filter_pred(args.true, args.pred, args.out)


if __name__ == "__main__":
    main()

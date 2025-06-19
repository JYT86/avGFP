
import requests
import pandas as pd
from time import sleep,time
from Bio.PDB import PDBParser
from io import StringIO
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process structural features.")
parser.add_argument('--start_id', '-i', type=str, required=True, help="Input FASTA file")
parser.add_argument('--prediction', '-p', type=int, default=0, help='Prediction Mode')


args = parser.parse_args()


url = "https://api.esmatlas.com/foldSequence/v1/pdb/"

def read_fasta(file_path: str):
    """
    Read a FASTA file and return a dictionary {seq_id: sequence}.
    
    Args:
        file_path (str): Path to the FASTA file.
    
    Returns:
        dict: Mapping from sequence ID to sequence string.
    """
    sequences = {}
    seq_id = None
    seq_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            print(line)
            if not line:
                continue
            if line.startswith(">"):
                if seq_id:
                    sequences[seq_id] = ''.join(seq_lines)
                seq_id = line[1:].split()[0]  # Only use first token as ID
                seq_lines = []
            else:
                seq_lines.append(line)
        # Don't forget the last one
        if seq_id:
            sequences[seq_id] = ''.join(seq_lines)

    return sequences

if not args.prediction:
    seqs = list(read_fasta(f'../data/avGFP_var/avGFP_var_from_{int(args.start_id)}_{int(args.start_id)+100}.fasta').values())
else:
    seqs = list(read_fasta(f'../data/avGFP_var/avGFP_var_wait_for_prediction.fasta').values())
    print(len(seqs))


def get_strunctral_info(seq:str):
    star_ids = [i for i, char in enumerate(seq) if char == '*']
    seq = seq.replace('*', '')
    success = False
    for attempt in range(3):
        try:
            resp = requests.post(url, data=seq, headers={"Content-Type": "text/plain"})
            resp.raise_for_status()
            success = True

            parser = PDBParser(QUIET=True)
            pdb_str = resp.content.decode("utf-8")
            structure = parser.get_structure("protein", StringIO(pdb_str))

            plddt_scores = []
            ca_coords = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.get_name() == 'CA':
                                plddt_scores.append(atom.get_bfactor())
                                ca_coords.append(atom.get_coord())

            # 插入缺失位置（*）用 0 替代
            for pos in sorted(star_ids, reverse=True):
                plddt_scores = plddt_scores[:pos] + [0.] + plddt_scores[pos:]
                ca_coords = ca_coords[:pos] + [np.array([0., 0., 0.])] + ca_coords[pos:]

            # return np.array(plddt_scores), np.array(ca_coords)  # shape: (L,), (L, 3)
            combined = np.hstack([
                    np.array(plddt_scores).reshape(-1, 1),  # (L, 1)
                    np.array(ca_coords)                # (L, 3)
                ])  # → shape: (L, 4)
            return combined

        except requests.exceptions.HTTPError as e:
            print(f"Error (try {attempt+1}): {e}")
            sleep(60)

    if not success:
        print(f"Failed to process sequence after 3 attempts.")
        return None, None
    else:
        sleep(1)



begin = time()
plddts = []
for i in range(len(seqs)):
    print(f'Now handling {i+int(args.start_id)} protein')
    
    plddt = get_strunctral_info(seqs[i])
    plddts.append(plddt)
end = time()
print(f'{end-begin}s')


vars = {f'protein_{id+int(args.start_id)}':plddt for id, plddt in zip(range(0, 100), plddts)}

import json
if not args.prediction:
    with open(f"../data/avGFP_structure/plddt_data_from_{int(args.start_id)}_{int(args.start_id)+100}.jsonl", "w") as f:
        for prot_id, plddt in vars.items():
            record = {
                "id": prot_id,
                "plddt": plddt.tolist() if isinstance(plddt, np.ndarray) else plddt
            }
            f.write(json.dumps(record) + "\n")
else:
    with open(f"../data/avGFP_structure/plddt_data_prediction.jsonl", "w") as f:
        for prot_id, plddt in vars.items():
            record = {
                "id": prot_id,
                "plddt": plddt.tolist() if isinstance(plddt, np.ndarray) else plddt
            }
            f.write(json.dumps(record) + "\n")

    

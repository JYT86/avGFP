# Implemented by JYT on 19 June 2025 – Latest version
import pandas as pd
import numpy as np
import json
from typing import List, Literal, Optional

import torch
from torch.utils.data import Dataset

import pyrosetta
from pyrosetta import pose_from_sequence, get_fa_scorefxn

import subprocess

pyrosetta.init("-mute all")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
batch_converter = alphabet.get_batch_converter()
esm_model.eval()  
esm_model = esm_model.to(device)


class ESMFlorescenceDataset(Dataset):
    def __init__(self, csv_path:str = None, seqs:list[str] = None, scores:list[str] = None):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
            self.data = self.data[~self.data['FullSequence'].str.contains('\\*')]
            self.var_seqs = self.data['FullSequence'].tolist()
            self.brightness_scores = self.data['Brightness'].tolist()

        if csv_path is None:
            self.var_seqs = seqs
            self.brightness_scores = scores

    def __len__(self):
        return len(self.var_seqs)

    def __getitem__(self, index):
        var_seq = self.var_seqs[index].replace("*", "")

        inputs = [(f"var_{index}", var_seq)]
        _, _, tokens = batch_converter(inputs)
        with torch.no_grad():
            results = esm_model(tokens.to(device), repr_layers=[6], return_contacts=False)

        var_repr = results["representations"][6][0, 1:len(var_seq)+1]

        # Labels
        label = torch.tensor(self.brightness_scores[index], dtype=torch.float)
        return var_repr, label
    

class ESMLabelDataset(Dataset):
    def __init__(self, csv_path:str = None, seqs:list[str] = None, scores:list[str] = None):
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
            self.var_seqs = self.data['FullSequence'].tolist()
            self.brightness_scores = self.data['Brightness'].tolist()
            self.brightness_scores = [1 if i > 2.5 else 0 for i in self.brightness_scores]
        if csv_path is None:
            self.var_seqs = seqs
            self.brightness_scores = scores
            self.brightness_scores = [1 if i > 2.5 else 0 for i in self.brightness_scores]
    def __len__(self):
        return len(self.var_seqs)

    def __getitem__(self, index):
        var_seq = self.var_seqs[index].replace("*", "")

        inputs = [(f"var_{index}", var_seq)]
        _, _, tokens = batch_converter(inputs)
        with torch.no_grad():
            results = esm_model(tokens.to(device), repr_layers=[6], return_contacts=False)

        var_repr = results["representations"][6][0, 1:len(var_seq)+1]

        # Labels
        label = torch.tensor(self.brightness_scores[index], dtype=torch.float)
        return var_repr, label
    

def esm_collate_fn(batch, max_length=None):

    xs, ys = zip(*batch)
    embed_dim = xs[0].shape[1]
    B = len(xs)

    target_len = max_length if max_length is not None else max(x.shape[0] for x in xs)

    X_padded = torch.zeros(B, target_len, embed_dim)
    y_padded = torch.zeros(B, 1, dtype=torch.float)

    mask = torch.zeros(B, target_len, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        cur_len = min(x.shape[0], target_len)
        X_padded[i, :cur_len] = x[:cur_len]
        y_padded[i, :] = y
        mask[i, :cur_len] = 1

    return X_padded, y_padded, mask




def get_structural_info_from_file(id:int, prediction=False):
    i = int(id/100)*100
    if prediction:
        with open('../data/avGFP_structure/plddt_data_prediction.jsonl', 'r') as f:
            structure_list = [json.loads(line) for line in f]
        info = [structure_list[j]['plddt'] for j in range(len(structure_list))]
        return np.array(info)
            
    with open(f'../data/avGFP_structure/plddt_data_from_{i}_{i+100}.jsonl', 'r') as f:
        structure_list = [json.loads(line) for line in f]
    # info = structure_list[id%100]['plddt']
    info = [structure_list[j]['plddt'] for j in range(100)]
    return np.array(info)



def one_hot_encode_aa(sequence, aa_list="ACDEFGHIKLMNPQRSTVWY*"):
    aa_to_index = {aa: i for i, aa in enumerate(aa_list)}
    
    # Initialize one-hot matrix
    one_hot_matrix = np.zeros((len(sequence), len(aa_list)), dtype=np.float32)
    
    for i, aa in enumerate(sequence):
        if aa in aa_to_index:  # Valid amino acid
            one_hot_matrix[i, aa_to_index[aa]] = 1.0
        else:  # Unknown amino acid (X or others)
            pass  # Optionally, you can set a special encoding for unknowns

    return one_hot_matrix

def ml_encode(
    seq_list: List[str],
    y_list: List[float],
    encode_strategy: Literal['one_hot', 'feature'] = 'one_hot',
    parital_residues: Optional[List[int]] = None,
    if_structure: bool = False,
    prediction = False,
):
    if encode_strategy == 'one_hot':
        X = []
        for i in range(len(seq_list)):
            encoding = one_hot_encode_aa(seq_list[i])
        X.append(encoding)
        return np.array(X) if parital_residues is None else np.array(X)[:, parital_residues, :], np.array(y_list)

    if encode_strategy == 'feature':
        vdw_volume = {
            'A':67, 'C':86, 'D':91, 'E':109, 'F':135,
            'G':48, 'H':118, 'I':124, 'K':135, 'L':124, 
            'M':124, 'N':96, 'P':90, 'Q':114, 'R':148,
            'S':73, 'T':93, 'V':105, 'W':163, 'Y':141, '*': 0
        }
        pI = {
            'A':6.11,'C':6.31,'D':5.945,'E':5.785,'F':5.755,
            'G':6.065,'H':5.565,'I':6.04,'K':5.61,'L':6.035,
            'M':5.705,'N':5.43,'P':6.295,'Q':5.65,'R':5.405,
            'S':5.70,'T':5.595,'V':6.065,'W':5.935,'Y':5.705, '*': 0,
        }
        hydropathy = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5, '*': 0.,
        }
        X = []
        for i in range(len(seq_list)):
            if if_structure:
                encoding = np.zeros((len(seq_list[i]), 7))
            else:
                encoding = np.zeros((len(seq_list[i]), 3))
            for j, aa in enumerate(seq_list[i]):
                encoding[j, 0] = vdw_volume[aa]
                encoding[j, 1] = pI[aa]
                encoding[j, 2] = hydropathy[aa]

            X.append(encoding)
        X = np.array(X)   
        if if_structure:
            
            if prediction:
                with open('../data/avGFP_var/avGFP_var_wait_for_prediction.fasta', "w") as f:
                    for j, seq in enumerate(seq_list):
                        header = f"variantforpred_{j}"
                        f.write(f'>{header}\n{seq}\n')
                subprocess.run([
                        "python", "download_PDB.py",
                        "--start_id", "0",
                        "--prediction", "1"
                    ],
                    stdout=subprocess.DEVNULL,   
                    stderr=subprocess.DEVNULL
                )

                X[:, :, -4:] = get_structural_info_from_file(0, True)

            else:
                for i in range(0, len(seq_list), 100):
                    X[i:i+100, :, -4:] = get_structural_info_from_file(i)
    
        return X if parital_residues is None else X[:, parital_residues, :], np.array(y_list)


def dcg(scores):
    return np.sum((2**scores - 1) / np.log2(np.arange(2, scores.size + 2)))
def ndcg_at_k(preds, trues, k=10):

    if len(preds) < k:
        k = len(preds)

    topk_pred_indices = np.argsort(preds)[::-1][:k]
    topk_true_indices = np.argsort(trues)[::-1][:k]

    rel_pred = np.array(trues)[topk_pred_indices]
    rel_ideal = np.array(trues)[topk_true_indices]

    dcg_val = dcg(rel_pred)
    idcg_val = dcg(rel_ideal)

    return dcg_val / idcg_val if idcg_val > 0 else 0.0

def reg_2_cls_acc(preds, trues, tollerate_rate=0.05):
    assert preds.shape == trues.shape
    correct_counts = 0
    for i,j in zip(preds, trues):
        if abs(i-j) / j <= tollerate_rate:
            correct_counts += 1
    return correct_counts / len(preds)

def compute_energy_with_rosetta(seq:str):
    pose = pose_from_sequence(seq, 'fa_standard')  
    scorefxn = get_fa_scorefxn()
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)
    total_score = scorefxn(pose)
    return total_score



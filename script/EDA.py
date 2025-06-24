# Implemented by JYT on 19 June 2025 – Latest version

import numpy as np
import pandas as pd

import torch

from collections import Counter
from prediction import predict
from model import ItPred4Classification, ItPredConfig

import argparse
import pickle
import json
import logging

class ProbTable:
    def __init__(self, residues_list:list[str], init_probs:str="uni"):
        
        self.residues_list = residues_list
        self.aa_list = list("ACDEFGHIKLMNPQRSTVWY*")
        self.wt = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
        if init_probs == "uni":
            self.pt = np.ones((len(self.aa_list), len(self.residues_list)))
            self.pt *= 1/len(self.aa_list)

    def sampling(self, num_seqs:int):
        seqs = []
        for _ in range(num_seqs):
            seq = ""
            for i in range(len(self.residues_list)):
                aa = np.random.choice(self.aa_list, p=self.pt[:, i])
                seq += aa
            seqs.append(seq)

        results = []
        for seq in seqs:
            temp = self.wt
            for residue, aa in zip(self.residues_list, list(seq)):
                temp = temp[:residue] + aa + temp[residue+1:]
            results.append(temp)
        return results
    
    def update(self, winner_seqs:list[str]):
        new_pt = np.zeros_like(self.pt)
        for i, residue in enumerate(self.residues_list):
            aas = [seq[residue] for seq in winner_seqs]
            count = Counter(aas)
            total = sum(count.values())
            prob_list = [count.get(aa, 0) / total for aa in self.aa_list]
            new_pt[:, i] = prob_list
        
        self.gamma = 0.2
        self.pt = (1 - self.gamma) * self.pt + self.gamma * new_pt


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description="Perform EDA.")
    parser.add_argument('--structure', '-s', action='store_true', help="IF take account of structural feature")
    parser.add_argument('--file', '-f', type=str, default='../data/deduplicated_variants_above_4.csv')

    args = parser.parse_args()
    df = pd.read_csv(args.file)

    config = ItPredConfig(320)
    cls_model = ItPred4Classification(config)
    config.residue_mask = False
    cls_model = ItPred4Classification(config)
    cls_model.load_state_dict(torch.load('../checkpoints/cls_model100.pt'))


    if args.structure:
        with open('../checkpoints/ml_reg_w_structure.pkl', 'rb') as f:
            reg_model = pickle.load(f)
    else:
        with open('../checkpoints/ml_reg_wo_structure.pkl', 'rb') as f:
            reg_model = pickle.load(f)


    scores, var_seqs, actions = df['Brightness'].tolist(), df['FullSequence'].tolist(), df['aaMutations'].tolist()
    res = []
    for score, seq, action in zip(scores, var_seqs, actions):
        action = action.split(':')
        residues = [int(act[1:-1]) for act in action]

        logger.info(f'residues: {residues}')
        pt = ProbTable(residues)
        for i in range(10):
            seqs = pt.sampling(100)
            preds = [predict(seq, cls_model, reg_model, args.structure, True) for seq in seqs]
            pred_brightness = np.array([pred['Brightness'] for pred in preds])
            winner_ids = np.argsort(pred_brightness)[-10:]
            winners = [seqs[i] for i in winner_ids]

            pt.update(winners)
            winner_scores = pred_brightness[winner_ids]

            logger.info(f'mean: {winner_scores.mean()}')
            logger.info(f' std: {winner_scores.std()}')
            if winner_scores.std() == 0.:
                break

        preds = predict(winners[-1], cls_model, reg_model, args.structure, True, True)
        dataset_preds = predict(seq, cls_model, reg_model, args.structure, True, True)
        res.append({
            'Sequence': winners[-1],
            'Residues': residues, 
            'Predicted Brightness': preds['Brightness'],
            'Predicted Energy': preds['Energy'],
            'Dataset best': seq,
            'Dataset predicted Brightness': dataset_preds['Brightness'],
            'Dataset predicted Energy': dataset_preds['Energy'],
            'Sum of Score':  preds['Brightness'] - preds['Energy'] # energy the lower the better
        })

    with open('../Final_result.jsonl', 'w') as f:
        for item in res:
            f.write(json.dumps(item) + '\n')

        
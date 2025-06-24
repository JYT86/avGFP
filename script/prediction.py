# Implemented by JYT on 19 June 2025 – Latest version
import torch
from torch.utils.data import DataLoader

from functools import partial
import numpy as np

from model import ItPred4Classification, ItPredConfig
from utils import ESMLabelDataset, esm_collate_fn, device, ml_encode, compute_energy_with_rosetta

from sklearn.base import BaseEstimator

import logging
import pickle


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)



def predict(
        seq: str, 
        cls_model: ItPred4Classification, 
        reg_model: BaseEstimator,
        structure: bool = False,
        after_training: bool = False,
        pred_energy: bool = False,
    ):

    dataset = ESMLabelDataset(None, [seq], [0.])
    loader = DataLoader(dataset, 1, collate_fn=partial(esm_collate_fn, max_length=None), shuffle=True)
    cls_model.eval()
    cls_model = cls_model.to(device)
    preds = []
    with torch.no_grad():
        for batch in loader:
            x, _, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y_pred = cls_model(x, mask)

            preds.append(y_pred.cpu())

    preds = torch.cat(preds).numpy()
    preds = (preds > 0.5).astype(int)
    if preds[0] == 1:
        logger.debug(f'This variant is H-Class.')
        X, _ = ml_encode([seq], [0], 'feature', None, structure, after_training)
        
        brightness = reg_model.predict(X.reshape(X.shape[0], -1))
        if pred_energy:
            energy = np.array([compute_energy_with_rosetta(seq) for _ in range(5)]).mean()
        else:
            energy = 0
        return {
            'Sequence': seq,
            'Brightness': brightness.item(),
            'Energy': energy,
        }
    
    else:
        brightness = 1.5
        energy = None
        return {
            'Sequence': seq,
            'Brightness': brightness,
            'Energy': energy,
        }
    

if __name__ == '__main__':

    # predict if high or not
    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRDEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'

    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDVTYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGRKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSTMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQRNGIKVNFKIRHNLEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
    config = ItPredConfig(320)
    config.residue_mask = False
    cls_model = ItPred4Classification(config)
    cls_model.load_state_dict(torch.load('../checkpoints/cls_model100.pt'))

    structure = True
    with open('../checkpoints/ml_reg_w_structure.pkl', 'rb') as f:
        reg_model = pickle.load(f)


    print(predict(seq, cls_model, reg_model, structure, True))
    
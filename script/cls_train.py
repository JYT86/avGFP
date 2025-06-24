# Implemented by JYT on 19 June 2025 – Latest version

# We formulate this as a hierarchical task using cascade learning.
# This file implements the first stage: the classification step.
# The goal is to predict whether a variant's brightness exceeds 2.5,
# based on its sequence.

import torch
from torch.nn import BCELoss
from torch.optim import AdamW
from torch.utils.data import Subset, DataLoader

from utils import ESMLabelDataset, esm_collate_fn, device
from model import ItPredConfig, ItPred4Classification

from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import logging

if __name__ == '__main__':

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


    torch.manual_seed(42)
    config = ItPredConfig(dim=320) 
    model = ItPred4Classification(config).to(device)

    train_valid_dataset = ESMLabelDataset('../data/deduplicated_variants.csv')
    train_indices, val_indices = train_test_split(
        list(range(len(train_valid_dataset))),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    valset = Subset(train_valid_dataset, val_indices)
    trainset = Subset(train_valid_dataset, train_indices)
    trainloader = DataLoader(trainset, 4, collate_fn=partial(esm_collate_fn, max_length=None), shuffle=True)
    valloader = DataLoader(valset, 4, collate_fn=partial(esm_collate_fn, max_length=None), shuffle=False)


    loss = BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)


    for epoch in range(100): 
        model.train()
        logger.info(f'Start {epoch+1} epoch.')
        logger.info(f'Training.')
        total_loss = 0.
        for batch in trainloader:
            x, y, mask = batch[0].to(device),  batch[1].to(device), batch[2].to(device)
            y_pred = model(x, mask)
            optimizer.zero_grad()
            l = loss(y_pred, y)
            l.backward()
            optimizer.step()
        
            total_loss += l.item()
        total_loss /= len(trainloader)
        logger.debug(f'Epoch {epoch}, loss {total_loss:.4f}')

        model.eval()
        logger.info('Eval.')
        preds, trues = [], []
        with torch.no_grad():
            for batch in valloader:
                x, y, mask = batch[0].to(device),  batch[1].to(device), batch[2].to(device)
                y_pred = model(x, mask)

                preds.append(y_pred.cpu())
                trues.append(y.cpu())
        preds = torch.cat(preds).numpy()
        trues = torch.cat(trues).numpy()

        preds = (preds > 0.5).astype(int)
        logger.info(f'acc: {accuracy_score(trues, preds)}')
        logger.info(f'recall: {recall_score(trues, preds)}')
        logger.info(f'precision: {precision_score(trues, preds)}')
        logger.info(f'f1 score: {f1_score(trues, preds)}')

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f"../checkpoints/cls_model{epoch+1}.pt")


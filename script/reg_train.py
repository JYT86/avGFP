# Implemented by JYT on 19 June 2025 – Latest version

# This task is formulated as a hierarchical process using cascade learning.
# This file implements the second stage: the regression step.
# The goal is to estimate the brightness score for variants previously
# classified as high-brightness (H-class) in the first stage.


import pandas as pd

from utils import ml_encode, ndcg_at_k, reg_2_cls_acc
import argparse
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

import logging

if __name__ == '__main__':


    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(description="ML Regressor Training.")
    parser.add_argument('--structure', '-s', action='store_true', help="IF take account of structural feature")
    parser.add_argument('--file', '-f', type=str, default='../data/deduplicated_variants.csv')
    parser.add_argument('--n_estimators', '-n', type=int, default=600)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.05)
    parser.add_argument('--max_depth', '-d', type=int, default=5)
    parser.add_argument('--subsample', '-p', type=float, default=0.6)
    parser.add_argument('--max_features', '-e', type=float, default=0.7)

    args = parser.parse_args()

    df = pd.read_csv(args.file)

    seqs = df['FullSequence'].tolist()[:20000]
    brightness = df['Brightness'].tolist()[:20000]
    logger.debug(f'Got {len(seqs)} sequences.')
    X, y = ml_encode(seqs, brightness, 'feature', None, args.structure)
    logger.debug(f'Got {X.shape[-1]} features.')

    high_class_X = X[y>2.5, :, :]
    high_class_y = y[y>2.5]
    low_class_X = X[y<=2.5, :, :]
    low_class_y = y[y<=2.5]
    X_train, X_test, y_train, y_test = train_test_split(high_class_X, high_class_y, test_size=0.1, random_state=2025)


    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        max_features=args.max_features,
    )

    # scaler = StandardScaler()
    # X_train= scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    # X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    # model = MLPRegressor(
    #     hidden_layer_sizes=(512, 64),        
    #     activation='relu',
    #     solver='adam',
    #     learning_rate='adaptive',           
    #     learning_rate_init=1e-3,             
    #     alpha=1e-3,                           
    #     early_stopping=True,             
    #     validation_fraction=0.1,             
    #     n_iter_no_change=10,                
    #     max_iter=1000,                       
    #     random_state=42,
    #     verbose=False                         
    # )

    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
    # model.fit(X_train, y_train)


    preds_test = model.predict(X_test.reshape(X_test.shape[0], -1))
    preds_train = model.predict(X_train.reshape(X_train.shape[0], -1))

    # preds_test = model.predict(X_test)
    # preds_train = model.predict(X_train)

    r2_test = r2_score(y_test, preds_test)
    mae_test = mean_absolute_error(y_test, preds_test)
    sp_test, _ = spearmanr(y_test, preds_test)
    r2_train = r2_score(y_train, preds_train)
    mae_train = mean_absolute_error(y_train, preds_train)
    sp_train, _ = spearmanr(y_train, preds_train)

    error_rate = 0.05

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=False)

    axes[0].plot(y_test, preds_test, 'o', alpha=0.6)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0].set_xlabel("True y_test")
    axes[0].set_ylabel("Predicted y_test")
    axes[0].set_title(f"Test set: sp_corr = {sp_test:.3f}, MAE = {mae_test:.3f}")
    axes[0].text(0.65, 0.2, f"{error_rate*100}% acc = {reg_2_cls_acc(preds_test, y_test, error_rate):.3f}\nR² = {r2_test:.3f}\nNDCG = {ndcg_at_k(preds_test, y_test):.3f}",
                transform=axes[0].transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axes[0].set_xlim(y_test.min(), y_test.max())
    axes[0].set_ylim(y_test.min(), y_test.max())

  
    axes[1].plot(y_train, preds_train, 'o', alpha=0.6)
    axes[1].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    axes[1].set_xlabel("True y_train")
    axes[1].set_ylabel("Predicted y_train")
    axes[1].set_title(f"Train set: sp_corr = {sp_train:.3f}, MAE = {mae_train:.3f}")
    axes[1].text(0.65, 0.2, f"{error_rate*100}% acc = {reg_2_cls_acc(preds_train, y_train, error_rate):.3f}\nR² = {sp_train:.3f}\nNDCG = {ndcg_at_k(preds_train, y_train):.3f}",
                transform=axes[1].transAxes,
                verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    axes[1].set_xlim(y_test.min(), y_test.max())
    axes[1].set_ylim(y_test.min(), y_test.max())

    plt.tight_layout()
    if args.structure:
        plt.savefig('../graphs/GBR_w_structure.png')
        with open('../checkpoints/ml_reg_w_structure.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        plt.savefig('../graphs/GBR_wo_structure.png')
        with open('../checkpoints/ml_reg_wo_structure.pkl', 'wb') as f:
            pickle.dump(model, f)

    
        


    

    
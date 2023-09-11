import argparse
import os
import random
from datetime import datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from gensim.models import word2vec
from pytorch_lightning import seed_everything
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
from tqdm import tqdm

import wandb
from preprocess import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, nargs="*", help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    args = parser.parse_args()
    return args


def wandb_start(cfg):
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        config=cfg,
    )


def inference(cfg, valid_X):
    if cfg["noid"]:
        model = joblib.load(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_noid.ckpt"
        )
        preds = model.predict(valid_X, num_iteration=model.best_iteration)
    else:
        model = joblib.load(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt"
        )
        preds = model.predict(valid_X, num_iteration=model.best_iteration)
    return preds


def one_fold(seen, cfg, valid_X_cv):
    seed_everything(cfg["general"]["seed"], workers=True)

    if seen:
        # with user_id
        print("with user_id")
        cfg["noid"] = False
        vec_columns = [f"wo_score_user_factor_{i}" for i in range(64)]
        drop_columns = cfg["drop_features_with_id"] + vec_columns
        valid_X_cv = valid_X_cv.drop(drop_columns, axis=1)
        valid_preds = inference(cfg, valid_X_cv)
        return valid_preds
    else:
        # without user_id
        print("without user_id")
        drop_columns = cfg["drop_features_noid"]
        valid_X_cv = valid_X_cv.drop(drop_columns, axis=1)
        cfg["noid"] = True
        valid_preds = inference(cfg, valid_X_cv)
        return valid_preds


def main():
    # Read config
    args = get_args()
    cfg_list = []
    for c in args.config:
        with open(c, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg_list.append(cfg)
    cfg = cfg_list[0]
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
        print(f"fold: {cfg['general']['fold']}")

    # Set random seed
    seed_everything(cfg["general"]["seed"], workers=True)

    # Read csv
    anime_meta = pd.read_csv(f"{cfg['general']['input_path']}/anime.csv")
    train = pd.read_csv(f"{cfg['general']['input_path']}/train.csv")
    test = pd.read_csv(f"{cfg['general']['input_path']}/test.csv")

    train, test = anime2vec_without_score(cfg, train, test)
    # train, test = anime2vec_with_genres(cfg, train, test, anime_meta)

    # preprocess
    cfg["features"] = {}
    cfg["features"]["text"] = [
        "genres",
        "japanese_name",
        "aired",
        "producers",
        "licensors",
        "studios",
    ]
    cfg["features"]["categorical"] = ["type", "source", "rating"]
    cfg["features"]["numerical"] = [
        "episodes",
        "duration",
        "members",
        "watching",
        "completed",
        "on_hold",
        "dropped",
        "plan_to_watch",
    ]

    anime_meta = comma_count(anime_meta)
    anime_meta = unknown_to_nan(anime_meta)
    train, _ = count_encoding_id(train, test)
    anime_meta = count_encoding(anime_meta, cfg["features"]["categorical"])
    anime_meta = aired_to_time_features(anime_meta)
    anime_meta = create_anime_num_features(anime_meta)
    # anime_meta = create_type_source_features(anime_meta)

    # type change
    anime_meta["type"] = anime_meta["type"].astype("category")
    anime_meta["source"] = anime_meta["source"].astype("category")
    # anime_meta["rating"] = anime_meta["rating"].astype("category")
    anime_meta["episodes"] = anime_meta["episodes"].astype("float64")
    anime_meta["members"] = anime_meta["members"].astype("float64")
    anime_meta["watching"] = anime_meta["watching"].astype("float64")
    anime_meta["completed"] = anime_meta["completed"].astype("float64")
    anime_meta["on_hold"] = anime_meta["on_hold"].astype("float64")
    anime_meta["dropped"] = anime_meta["dropped"].astype("float64")
    anime_meta["plan_to_watch"] = anime_meta["plan_to_watch"].astype("float64")
    anime_meta["duration"] = anime_meta["duration"].apply(convert_to_seconds)

    anime_meta = label_encoding(anime_meta, ["rating"])

    # merge embs
    anime_genres = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_genres.csv"
    )
    anime_genres_columns = anime_genres.columns
    anime_meta = anime_meta.merge(anime_genres, on="anime_id", how="left")
    # """
    anime_names = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_japanese_name_svd50.csv"
    )
    anime_names_columns = anime_names.columns
    anime_meta = anime_meta.merge(anime_names, on="anime_id", how="left")
    # """
    """
    anime_studios = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_studios_svd50.csv"
    )
    anime_studios_columns = anime_genre.columns
    anime_meta = anime_meta.merge(anime_studios, on="anime_id", how="left")
    """
    """
    anime_licensors = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_licensors_svd50.csv"
    )
    anime_licensors_columns = anime_genre.columns
    anime_meta = anime_meta.merge(anime_licensors, on="anime_id", how="left")
    """
    """
    anime_producers = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_producers_svd50.csv"
    )
    anime_producers_columns = anime_genre.columns
    anime_meta = anime_meta.merge(anime_producers, on="anime_id", how="left")
    """

    # anime_meta = cat_groupby(anime_meta)

    # merge anime_meta
    train = train.merge(anime_meta, on="anime_id", how="left")
    test = test.merge(anime_meta, on="anime_id", how="left")

    train, test = user_groupby(train, test)
    # train, test = user_groupby_genres(train, test, anime_genres)

    # Split X/y
    train_X = train.copy()
    train_y = train["score"]

    if cfg["general"]["cv"]:
        skf = StratifiedKFold(
            n_splits=cfg["general"]["n_splits"],
            shuffle=True,
            random_state=cfg["general"]["seed"],
        )
        sgkf = StratifiedGroupKFold(
            n_splits=cfg["general"]["n_splits"],
            shuffle=True,
            random_state=cfg["general"]["seed"],
        )
        weights_list = []
        weights_list_noid = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            print(f"fold_{fold_n} start")
            # ------------------------------------------------------------------------#
            if cfg["train_with_id"]:
                print("seen")

                train_indices, valid_indices = list(skf.split(train_X, train_y))[fold_n]
                train_X_cv, train_y_cv = (
                    train_X.iloc[train_indices].reset_index(drop=True),
                    train_y.iloc[train_indices].reset_index(drop=True),
                )
                valid_X_cv, valid_y_cv = (
                    train_X.iloc[valid_indices].reset_index(drop=True),
                    train_y.iloc[valid_indices].reset_index(drop=True),
                )
                print(f"train len: {len(train_X_cv)}")
                print(f"valid len: {len(valid_X_cv)}")

                train_X_cv, valid_X_cv = target_encoding(
                    cfg,
                    ["user_id", "anime_id", "type"],
                    train_X_cv,
                    train_y_cv,
                    valid_X_cv,
                    test=None,
                )
                train_X_cv, valid_X_cv = anime2vec(
                    cfg, train_X_cv, valid_X_cv, test=None
                )

                train_X_cv = train_X_cv.drop(["score"], axis=1)
                valid_X_cv = valid_X_cv.drop(["score"], axis=1)
                train_X_cv["user_id"] = train_X_cv["user_id"].astype("category")
                valid_X_cv["user_id"] = valid_X_cv["user_id"].astype("category")
                train_X_cv["anime_id"] = train_X_cv["anime_id"].astype("category")
                valid_X_cv["anime_id"] = valid_X_cv["anime_id"].astype("category")

                # print info
                # print(train_X_cv.info())
                # print(train_X_cv.head())
                # print(valid_X_cv.head())

                preds_list = []
                for cfg_ in cfg_list:
                    cfg_["fold_n"] = fold_n
                    preds = one_fold(True, cfg_, valid_X_cv)
                    preds_list.append(preds)

                for i, p in enumerate(preds_list):
                    rmse = np.sqrt(mean_squared_error(valid_y_cv, p))
                    print(f"rmse_[{i}]:{rmse}")

                def f(x):
                    pred = 0
                    for i, p in enumerate(preds_list):
                        pred += p * x[i]
                    score = np.sqrt(mean_squared_error(valid_y_cv, pred))
                    return score

                init_state = np.ones((len(preds_list))) / len(preds_list)
                bounds = [(0.0, 1.0)] * len(preds_list)
                result = minimize(f, init_state, method="Nelder-Mead", bounds=bounds)
                print(f"optimized_rmse:{result['fun']}")

                weights = [[0] for _ in range(len(preds_list))]
                for i in range(len(preds_list)):
                    weights[i] = result["x"][i]
                weights_list.append(weights)
                print(weights)

            else:
                pass
            # ------------------------------------------------------------------------#
            if cfg["train_noid"]:
                print("unseen")

                train_indices, valid_indices = list(
                    sgkf.split(train_X, train_y, train_X["user_id"])
                )[fold_n]
                train_X_cv, train_y_cv = (
                    train_X.iloc[train_indices].reset_index(drop=True),
                    train_y.iloc[train_indices].reset_index(drop=True),
                )
                valid_X_cv, valid_y_cv = (
                    train_X.iloc[valid_indices].reset_index(drop=True),
                    train_y.iloc[valid_indices].reset_index(drop=True),
                )
                print(f"train len: {len(train_X_cv)}")
                print(f"valid len: {len(valid_X_cv)}")

                train_X_cv, valid_X_cv = target_encoding(
                    cfg,
                    ["type"],  # ["anime_id"],
                    train_X_cv,
                    train_y_cv,
                    valid_X_cv,
                    test=None,
                )
                # train_X_cv, valid_X_cv = anime2vec(
                #    cfg, train_X_cv, valid_X_cv, test=None
                # )

                train_X_cv = train_X_cv.drop(["score"], axis=1)
                valid_X_cv = valid_X_cv.drop(["score"], axis=1)
                train_X_cv["user_id"] = train_X_cv["user_id"].astype("category")
                valid_X_cv["user_id"] = valid_X_cv["user_id"].astype("category")
                train_X_cv["anime_id"] = train_X_cv["anime_id"].astype("category")
                valid_X_cv["anime_id"] = valid_X_cv["anime_id"].astype("category")

                # print info
                # print(train_X_cv.info())
                # print(train_X_cv.head())
                # print(valid_X_cv.head())

                preds_list = []
                for cfg_ in cfg_list:
                    cfg_["fold_n"] = fold_n
                    preds = one_fold(False, cfg_, valid_X_cv)
                    preds_list.append(preds)

                for i, p in enumerate(preds_list):
                    rmse = np.sqrt(mean_squared_error(valid_y_cv, p))
                    print(f"rmse_[{i}]:{rmse}")

                def f(x):
                    pred = 0
                    for i, p in enumerate(preds_list):
                        pred += p * x[i]
                    score = np.sqrt(mean_squared_error(valid_y_cv, pred))
                    return score

                init_state = np.ones((len(preds_list))) / len(preds_list)
                bounds = [(0.0, 1.0)] * len(preds_list)
                result = minimize(f, init_state, method="Nelder-Mead", bounds=bounds)
                print(f"optimized_rmse:{result['fun']}")

                weights = [[0] for _ in range(len(preds_list))]
                for i in range(len(preds_list)):
                    weights[i] = result["x"][i]
                weights_list_noid.append(weights)
                print(weights)
            else:
                pass

        weights_mean = np.mean(weights_list, axis=0)
        weights_mean_noid = np.mean(weights_list_noid, axis=0)
        print(f"cv weights mean: {weights_mean}")
        print(f"cv weights mean (without user_id): {weights_mean_noid}")

    else:
        pass


if __name__ == "__main__":
    main()

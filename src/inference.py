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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
from tqdm import tqdm

from preprocess import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    args = parser.parse_args()
    return args


def inference(cfg, test, noid=False):
    if cfg["noid"]:
        model = joblib.load(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_noid.ckpt"
        )
        preds = model.predict(test, num_iteration=model.best_iteration)
    else:
        model = joblib.load(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt"
        )
        preds = model.predict(test, num_iteration=model.best_iteration)
    return preds


def one_fold(cfg, test):
    seed_everything(cfg["general"]["seed"], workers=True)

    preds = inference(cfg, test)

    return preds


def main():
    # Read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
        print(f"fold: {cfg['general']['fold']}")

    # Set random seed
    seed_everything(cfg["general"]["seed"], workers=True)

    # Read csv
    anime_meta = pd.read_csv(f"{cfg['general']['input_path']}/anime.csv")
    train = pd.read_csv(f"{cfg['general']['input_path']}/train.csv")
    test = pd.read_csv(f"{cfg['general']['input_path']}/test.csv")
    sample_submission = pd.read_csv(
        f"{cfg['general']['input_path']}/sample_submission.csv"
    )

    train, test = anime2vec_without_score(cfg, train, test)
    train, test = anime2vec_with_genres(cfg, train, test, anime_meta)

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
    train, test = count_encoding_id(train, test)
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
    anime_meta = anime_meta.merge(anime_genres, on="anime_id", how="left")
    # """
    anime_names = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_japanese_name_svd50.csv"
    )
    anime_meta = anime_meta.merge(anime_names, on="anime_id", how="left")
    # """
    # """
    anime_studios = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_studios_svd50.csv"
    )
    anime_meta = anime_meta.merge(anime_studios, on="anime_id", how="left")
    # """
    # """
    anime_licensors = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_licensors_svd50.csv"
    )
    anime_meta = anime_meta.merge(anime_licensors, on="anime_id", how="left")
    # """
    # """
    anime_producers = pd.read_csv(
        f"{cfg['general']['input_path']}/created_features/tfidf/anime_producers_svd50.csv"
    )
    anime_meta = anime_meta.merge(anime_producers, on="anime_id", how="left")
    # """

    # merge anime_meta
    train = train.merge(anime_meta, on="anime_id", how="left")
    test = test.merge(anime_meta, on="anime_id", how="left")

    train, test = user_groupby(train, test)

    # Split X/y
    train_X = train.copy()
    train_y = train["score"]

    test_tmp = test.copy()
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
        preds_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            print(f"fold_{fold_n} start")
            cfg["fold_n"] = fold_n
            # ------------------------------------------------------------------------#
            test = test_tmp.copy()
            print("seen")

            train_indices, _ = list(skf.split(train_X, train_y))[fold_n]
            train_X_cv, train_y_cv = (
                train_X.iloc[train_indices].reset_index(drop=True),
                train_y.iloc[train_indices].reset_index(drop=True),
            )

            test = target_encoding(
                cfg,
                ["user_id", "anime_id", "type", "source", "rating", "season"],
                train_X_cv,
                train_y_cv,
                valid_X=None,
                test=test,
            )
            test = anime2vec(cfg, train_X_cv, valid_X=None, test=test)

            test_in_train = test[test["user_id"].isin(train["user_id"])].copy()

            test_in_train["user_id"] = test_in_train["user_id"].astype("category")
            test_in_train["anime_id"] = test_in_train["anime_id"].astype("category")

            # print info
            print(f"test len: {len(test_in_train)}")
            print(test_in_train.info())
            print(test_in_train.head())

            cfg["noid"] = False
            vec_columns = [f"wo_score_user_factor_{i}" for i in range(64)]
            drop_columns = cfg["drop_features_with_id"] + vec_columns
            test_in_train = test_in_train.drop(drop_columns, axis=1)
            preds = one_fold(cfg, test_in_train)
            test_in_train["preds"] = preds

            # ------------------------------------------------------------------------#
            test = test_tmp.copy()
            print("unseen")

            train_indices, _ = list(sgkf.split(train_X, train_y, train_X["user_id"]))[
                fold_n
            ]
            train_X_cv, train_y_cv = (
                train_X.iloc[train_indices].reset_index(drop=True),
                train_y.iloc[train_indices].reset_index(drop=True),
            )

            test = target_encoding(
                cfg,
                ["type", "source", "rating", "season"],
                train_X_cv,
                train_y_cv,
                valid_X=None,
                test=test,
            )

            test_not_in_train = test[~test["user_id"].isin(train["user_id"])].copy()

            test_not_in_train["user_id"] = test_not_in_train["user_id"].astype(
                "category"
            )
            test_not_in_train["anime_id"] = test_not_in_train["anime_id"].astype(
                "category"
            )

            # print info
            print(f"test len: {len(test_not_in_train)}")
            print(test_not_in_train.info())
            print(test_not_in_train.head())

            cfg["noid"] = True
            drop_columns = cfg["drop_features_noid"]
            test_not_in_train = test_not_in_train.drop(drop_columns, axis=1)
            preds = one_fold(cfg, test_not_in_train)
            test_not_in_train["preds"] = preds

            # concat
            concated = pd.concat([test_in_train, test_not_in_train]).sort_index()
            preds = concated["preds"]
            preds_list.append(preds)

        sub_preds = np.mean(preds_list, axis=0)
    else:
        # train all data
        cfg["fold_n"] = "all"

        sub_preds = one_fold(cfg, test)

    sample_submission["score"] = sub_preds
    print(sample_submission)
    os.makedirs(f"{cfg['general']['output_path']}/sub/", exist_ok=True)
    sample_submission.to_csv(
        f"{cfg['general']['output_path']}/sub/{cfg['general']['save_name']}_sub.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

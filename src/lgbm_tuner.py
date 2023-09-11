import argparse
import json
import os
import random
from datetime import datetime

import joblib
import numpy as np
import optuna.integration.lightgbm as lgb
import pandas as pd
import yaml
from gensim.models import word2vec
from pytorch_lightning import seed_everything
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder
from tqdm import tqdm

import wandb
from preprocess import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
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


def train_and_inference(cfg, train_X, train_y, valid_X=None, valid_y=None):
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)

    params = cfg["model_noid"]["params"] if cfg["noid"] else cfg["model"]["params"]

    callbacks = [
        lgb.early_stopping(cfg["lightgbm"]["patience"]),
        lgb.log_evaluation(100),
    ]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    best_params = model.params
    sfx = "_noid" if cfg["noid"] else ""
    path = f"{cfg['general']['output_path']}/tuned_dict/{cfg['general']['save_name']}{sfx}.json"
    json_file = open(path, mode="w")
    json.dump(best_params, json_file)
    json_file.close()

    plot = lgb.plot_importance(model, importance_type="gain", max_num_features=10)
    plot = wandb.Image(plot)
    if cfg["noid"]:
        wandb.log({"feature_importance_noid": plot})
    else:
        wandb.log({"feature_importance": plot})

    if cfg["model_save"]:
        os.makedirs(
            f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}",
            exist_ok=True,
        )
        if cfg["noid"]:
            joblib.dump(
                model,
                f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}_noid.ckpt",
                compress=3,
            )
        else:
            joblib.dump(
                model,
                f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/fold{cfg['fold_n']}.ckpt",
                compress=3,
            )

    if valid_X is None:
        return None
    else:
        # valid_preds = rf.predict(valid_X)
        valid_preds = model.predict(valid_X, num_iteration=model.best_iteration)
        return valid_preds


def one_fold(seen, cfg, train_X_cv, train_y_cv, valid_X_cv=None, valid_y_cv=None):
    wandb_start(cfg)
    seed_everything(cfg["general"]["seed"], workers=True)

    if seen:
        # with user_id
        print("with user_id")
        cfg["noid"] = False
        vec_columns = [f"wo_score_user_factor_{i}" for i in range(64)]
        drop_columns = cfg["drop_features_with_id"] + vec_columns
        train_X_cv_ = train_X_cv.drop(drop_columns, axis=1)
        valid_X_cv_ = valid_X_cv.drop(drop_columns, axis=1)
        valid_preds = train_and_inference(
            cfg, train_X_cv_, train_y_cv, valid_X_cv_, valid_y_cv
        )
        if valid_preds is None:
            return None
        else:
            rmse = np.sqrt(mean_squared_error(valid_y_cv, valid_preds))
            print(f"fold{cfg['fold_n']} rmse: {rmse}")
            wandb.log({"rmse": rmse})
            wandb.finish()
            return rmse
    else:
        # without user_id
        print("without user_id")
        drop_columns = cfg["drop_features_noid"]
        train_X_cv_ = train_X_cv.drop(drop_columns, axis=1)
        valid_X_cv_ = valid_X_cv.drop(drop_columns, axis=1)
        cfg["noid"] = True
        valid_preds_noid = train_and_inference(
            cfg, train_X_cv_, train_y_cv, valid_X_cv_, valid_y_cv
        )
        if valid_preds_noid is None:
            return None
        else:
            rmse_noid = np.sqrt(mean_squared_error(valid_y_cv, valid_preds_noid))
            print(f"fold{cfg['fold_n']} rmse (without user_id): {rmse_noid}")
            wandb.log({"rmse_noid": rmse_noid})
            wandb.finish()
            return rmse_noid


def main():
    # Read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
        print(f"fold: {cfg['general']['fold']}")
    else:
        print("all train")

    # Set jobtype for wandb
    cfg["job_type"] = "train"

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
    anime_meta = anime_meta.merge(anime_genres, on="anime_id", how="left")
    # anime_names = pd.read_csv(
    #    f"{cfg['general']['input_path']}/created_features/embs/anime_genres_embs50.csv"
    # )
    # anime_meta = anime_meta.merge(anime_names, on="anime_id", how="left")

    # anime_meta = cat_groupby(anime_meta)

    # merge anime_meta
    train = train.merge(anime_meta, on="anime_id", how="left")
    test = test.merge(anime_meta, on="anime_id", how="left")

    train, test = user_groupby(train, test)

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
        rmse_list = []
        rmse_noid_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            print(f"fold_{fold_n} start")
            cfg["fold_n"] = fold_n
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
                    ["user_id", "anime_id"],
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
                print(train_X_cv.info())
                print(train_X_cv.head())
                print(valid_X_cv.head())

                rmse = one_fold(
                    True, cfg, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv
                )
                rmse_list.append(rmse)
            else:
                rmse_list.append(0)
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

                # train_X_cv, valid_X_cv = target_encoding(
                #    cfg,
                #    ["anime_id"],
                #    train_X_cv,
                #    train_y_cv,
                #    valid_X_cv,
                #    test=None,
                # )
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
                print(train_X_cv.info())
                print(train_X_cv.head())
                print(valid_X_cv.head())

                rmse_noid = one_fold(
                    False, cfg, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv
                )
                rmse_noid_list.append(rmse_noid)
            else:
                rmse_noid_list.append(0)

        cfg["fold_n"] = "summary"
        wandb_start(cfg)
        rmse_mean = np.mean(rmse_list, axis=0)
        rmse_noid_mean = np.mean(rmse_noid_list, axis=0)
        wandb.log({"mean_rmse": rmse_mean, "mean_rmse_without_user_id": rmse_noid_mean})
        print(
            f"cv mean rmse: {rmse_mean}, cv mean rmse (without user_id): {rmse_noid_mean}"
        )
        wandb.finish()

    else:
        # train all data
        cfg["fold_n"] = "all"

        one_fold(cfg, train_X, train_y)


if __name__ == "__main__":
    main()

import random
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import word2vec
from pytorch_lightning import seed_everything
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, TargetEncoder


def convert_to_seconds(duration):
    if duration is np.nan:
        return None

    parts = duration.split(" ")
    time_in_seconds = 0

    for part in parts:
        if part.isdigit():
            time_in_seconds_tmp = int(part)
        elif part == "min.":
            time_in_seconds += time_in_seconds_tmp * 60
        elif part == "hr.":
            time_in_seconds += time_in_seconds_tmp * 3600

    return time_in_seconds


def parse_date_range(date_range_string):
    if date_range_string is np.nan:
        return {
            "start_year": np.nan,
            "end_year": np.nan,
            "start_month": np.nan,
            "end_month": np.nan,
            "duration_days": np.nan,
            "season": np.nan,
            # "duration_months": np.nan,
            # "duration_years": np.nan,
        }

    if " to " in date_range_string:
        start_date_str, end_date_str = date_range_string.split(" to ")
        try:
            start_date = datetime.strptime(start_date_str, "%b %d, %Y")
        except ValueError:
            try:
                start_date = datetime.strptime(start_date_str, "%b, %Y")
            except ValueError:
                start_date = datetime.strptime(start_date_str, "%Y")
        try:
            end_date = (
                datetime.strptime(end_date_str, "%b %d, %Y")
                if end_date_str != "?"
                else None
            )
        except ValueError:
            try:
                end_date = (
                    datetime.strptime(end_date_str, "%b, %Y")
                    if end_date_str != "?"
                    else None
                )
            except ValueError:
                end_date = (
                    datetime.strptime(end_date_str, "%Y")
                    if end_date_str != "?"
                    else None
                )
    else:
        try:
            start_date_str = date_range_string
            start_date = datetime.strptime(start_date_str, "%b %d, %Y")
            end_date = None
        except ValueError:
            try:
                start_date_str = date_range_string
                start_date = datetime.strptime(start_date_str, "%b, %Y")
                end_date = None
            except ValueError:
                start_date_str = date_range_string
                start_date = datetime.strptime(start_date_str, "%Y")
                end_date = None

    if start_date.month == 4.0:
        season = "spring"
    elif start_date.month == 7.0:
        season = "summer"
    elif start_date.month == 10.0:
        season = "fall"
    elif start_date.month == 1.0:
        season = "winter"
    else:
        season = np.nan

    return {
        "start_year": start_date.year,
        "end_year": end_date.year
        if end_date and end_date_str != "?"
        else start_date.year,
        "start_month": start_date.month,
        "end_month": end_date.month
        if end_date and end_date_str != "?"
        else start_date.month,
        "duration_days": (end_date - start_date).days
        if end_date and end_date_str != "?"
        else np.nan,
        "season": season,
        # "duration_months": (end_date - start_date).days / 30
        # if end_date and end_date_str != "?"
        # else np.nan,
        # "duration_years": (end_date - start_date).days / 365
        # if end_date and end_date_str != "?"
        # else np.nan,
    }


def create_anime_num_features(anime):
    anime["members_digit_count"] = (
        anime["members"].apply(lambda x: len(str(x))).astype("float64")
    )
    anime["watching_digit_count"] = (
        anime["watching"].apply(lambda x: len(str(x))).astype("float64")
    )
    anime["completed_digit_count"] = (
        anime["completed"].apply(lambda x: len(str(x))).astype("float64")
    )
    anime["on_hold_digit_count"] = (
        anime["on_hold"].apply(lambda x: len(str(x))).astype("float64")
    )
    anime["dropped_digit_count"] = (
        anime["dropped"].apply(lambda x: len(str(x))).astype("float64")
    )
    anime["plan_to_watch_digit_count"] = (
        anime["plan_to_watch"].apply(lambda x: len(str(x))).astype("float64")
    )

    # anime["not_watching"] = (anime["members"] - anime["watching"]).astype("float64")
    # anime["not_completed"] = (anime["members"] - anime["completed"]).astype("float64")
    # anime["not_on_hold"] = (anime["members"] - anime["on_hold"]).astype("float64")
    # anime["not_dropped"] = (anime["members"] - anime["dropped"]).astype("float64")
    # anime["not_plan_to_watch"] = (anime["members"] - anime["plan_to_watch"]).astype(
    #    "float64"
    # )

    # anime["positive_members"] = (anime["watching"] + anime["completed"] + anime["plan_to_watch"]).astype("float64")
    # anime["negative_members"] = (anime["dropped"] + anime["on_hold"]).astype("float64")

    anime["watching_ratio"] = anime["watching"] / anime["members"]
    anime["completed_ratio"] = anime["completed"] / anime["members"]
    anime["on_hold_ratio"] = anime["on_hold"] / anime["members"]
    anime["dropped_ratio"] = anime["dropped"] / anime["members"]
    anime["plan_to_watch_ratio"] = anime["plan_to_watch"] / anime["members"]

    # anime["not_watching_ratio"] = anime["not_watching"] / anime["members"]
    # anime["not_completed_ratio"] = anime["not_completed"] / anime["members"]
    # anime["not_on_hold_ratio"] = anime["not_on_hold"] / anime["members"]
    # anime["not_dropped_ratio"] = anime["not_dropped"] / anime["members"]
    # anime["not_plan_to_watch_ratio"] = anime["not_plan_to_watch"] / anime["members"]

    # anime["positive_members_ratio"] = anime["positive_members"] / anime["members"]
    # anime["negative_members_ratio"] = anime["negative_members"] / anime["members"]

    # anime["completed-dropped"] = anime["completed_ratio"] - anime["dropped_ratio"]

    return anime


def add_w2v_features_with_score(cfg, train_df, val_df, test_df=None):
    seed_everything(cfg["general"]["seed"], workers=True)
    anime_ids = train_df["anime_id"].unique().tolist()
    user_anime_list_dict = {
        user_id: anime_ids.tolist()
        for user_id, anime_ids in train_df.groupby("user_id")["anime_id"]
    }

    # スコアを考慮する場合
    # 今回は1～10のレーティングなので、スコアが5のアニメは5回、スコアが10のアニメは10回、タイトルをリストに追加する
    title_sentence_list = []
    for user_id, user_df in train_df.groupby("user_id"):
        user_title_sentence_list = []
        for anime_id, anime_score in user_df[["anime_id", "score"]].values:
            for i in range(anime_score):
                user_title_sentence_list.append(anime_id)
        title_sentence_list.append(user_title_sentence_list)

    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [
        random.sample(sentence, len(sentence)) for sentence in title_sentence_list
    ]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": cfg["general"]["seed"],
        "min_count": 1,
        "workers": 1,
    }

    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

    # ユーザーごとの特徴ベクトルと対応するユーザーID
    user_factors = {
        user_id: np.mean([model.wv[anime_id] for anime_id in user_anime_list], axis=0)
        for user_id, user_anime_list in user_anime_list_dict.items()
    }

    # アイテムごとの特徴ベクトルと対応するアイテムID
    item_factors = {aid: model.wv[aid] for aid in anime_ids}

    # データフレームを作成
    user_factors_df = (
        pd.DataFrame(user_factors).T.reset_index().rename(columns={"index": "user_id"})
    )
    item_factors_df = (
        pd.DataFrame(item_factors).T.reset_index().rename(columns={"index": "anime_id"})
    )

    # データフレームのカラム名をリネーム
    user_factors_df.columns = ["user_id"] + [
        f"user_factor_{i}" for i in range(vector_size)
    ]
    item_factors_df.columns = ["anime_id"] + [
        f"item_factor_{i}" for i in range(vector_size)
    ]

    train_df = train_df.merge(user_factors_df, on="user_id", how="left")
    train_df = train_df.merge(item_factors_df, on="anime_id", how="left")

    if val_df is not None:
        val_df = val_df.merge(user_factors_df, on="user_id", how="left")
        val_df = val_df.merge(item_factors_df, on="anime_id", how="left")
    else:
        val_df = None

    if test_df is not None:
        test_df = test_df.merge(user_factors_df, on="user_id", how="left")
        test_df = test_df.merge(item_factors_df, on="anime_id", how="left")
        return train_df, val_df, test_df

    return train_df, val_df


def add_w2v_features_without_score(cfg, train_test_df):
    seed_everything(cfg["general"]["seed"], workers=True)
    anime_ids = train_test_df["anime_id"].unique().tolist()
    user_anime_list_dict = {
        user_id: anime_ids.tolist()
        for user_id, anime_ids in train_test_df.groupby("user_id")["anime_id"]
    }

    title_sentence_list = (
        train_test_df.groupby("user_id")["anime_id"].apply(list).tolist()
    )

    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [
        random.sample(sentence, len(sentence)) for sentence in title_sentence_list
    ]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = title_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": cfg["general"]["seed"],
        "min_count": 1,
        "workers": 1,
    }

    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

    # ユーザーごとの特徴ベクトルと対応するユーザーID
    user_factors = {
        user_id: np.mean([model.wv[anime_id] for anime_id in user_anime_list], axis=0)
        for user_id, user_anime_list in user_anime_list_dict.items()
    }

    # アイテムごとの特徴ベクトルと対応するアイテムID
    item_factors = {aid: model.wv[aid] for aid in anime_ids}

    # データフレームを作成
    user_factors_df = (
        pd.DataFrame(user_factors).T.reset_index().rename(columns={"index": "user_id"})
    )
    item_factors_df = (
        pd.DataFrame(item_factors).T.reset_index().rename(columns={"index": "anime_id"})
    )

    # データフレームのカラム名をリネーム
    user_factors_df.columns = ["user_id"] + [
        f"wo_score_user_factor_{i}" for i in range(vector_size)
    ]
    item_factors_df.columns = ["anime_id"] + [
        f"wo_score_item_factor_{i}" for i in range(vector_size)
    ]

    train_test_df = train_test_df.merge(user_factors_df, on="user_id", how="left")
    # train_test_df = train_test_df.merge(item_factors_df, on="anime_id", how="left")

    return train_test_df


def add_w2v_features_with_genres(cfg, train_test_df):
    seed_everything(cfg["general"]["seed"], workers=True)

    genre_set = set()
    for g in train_test_df["genres"]:
        each_set = set(g.split(", "))
        genre_set = genre_set.union(each_set)
    genres_list = list(genre_set)

    user_genres_list_dict = {
        user_id: [
            item
            for sublist in (elem.split(", ") for elem in genres)
            for item in sublist
        ]
        for user_id, genres in train_test_df.groupby("user_id")["genres"]
    }

    genres_sentence_list = (
        train_test_df.groupby("user_id")["genres"].apply(list).tolist()
    )
    formatted_list = []
    for genres_sentence in genres_sentence_list:
        formatted_list.append(
            [
                item
                for sublist in (elem.split(", ") for elem in genres_sentence)
                for item in sublist
            ]
        )

    # ユーザごとにshuffleしたリストを作成
    shuffled_sentence_list = [
        random.sample(sentence, len(sentence)) for sentence in formatted_list
    ]  ## <= 変更点

    # 元のリストとshuffleしたリストを合わせる
    train_sentence_list = genres_sentence_list + shuffled_sentence_list

    # word2vecのパラメータ
    vector_size = 64
    w2v_params = {
        "vector_size": vector_size,  ## <= 変更点
        "seed": cfg["general"]["seed"],
        "min_count": 1,
        "workers": 1,
    }

    # word2vecのモデル学習
    model = word2vec.Word2Vec(train_sentence_list, **w2v_params)

    # ユーザーごとの特徴ベクトルと対応するユーザーID
    user_factors = {
        user_id: np.mean([model.wv[genre] for genre in user_genres_list], axis=0)
        for user_id, user_genres_list in user_genres_list_dict.items()
    }

    # アイテムごとの特徴ベクトルと対応するアイテムID
    genre_factors = {genre_elem: model.wv[genre_elem] for genre_elem in genres_list}

    # データフレームを作成
    user_factors_df = (
        pd.DataFrame(user_factors).T.reset_index().rename(columns={"index": "user_id"})
    )
    genre_factors_df = (
        pd.DataFrame(genre_factors).T.reset_index().rename(columns={"index": "genre"})
    )

    # データフレームのカラム名をリネーム
    user_factors_df.columns = ["user_id"] + [
        f"wo_score_genre_user_factor_{i}" for i in range(vector_size)
    ]
    genre_factors_df.columns = ["genre"] + [
        f"wo_score_genre_factor_{i}" for i in range(vector_size)
    ]

    train_test_df = train_test_df.merge(user_factors_df, on="user_id", how="left")
    # train_test_df = train_test_df.merge(genre_factors_df, on="anime_id", how="left")

    return train_test_df


def comma_count(anime_meta):
    anime_meta["genres_num_count"] = [
        genre_row.count(",") + 1 for genre_row in anime_meta["genres"]
    ]
    anime_meta["producers_num_count"] = [
        genre_row.count(",") + 1 for genre_row in anime_meta["producers"]
    ]
    anime_meta["licensors_num_count"] = [
        genre_row.count(",") + 1 for genre_row in anime_meta["licensors"]
    ]
    anime_meta["studios_num_count"] = [
        genre_row.count(",") + 1 for genre_row in anime_meta["studios"]
    ]

    return anime_meta


def unknown_to_nan(anime_meta):
    anime_meta = anime_meta.replace("Unknown", np.nan)

    return anime_meta


def count_encoding_id(train, test):
    train_and_test = pd.concat([train, test])
    user_id_counts = train_and_test["user_id"].value_counts()
    train_and_test["user_id_counts"] = train_and_test["user_id"].map(user_id_counts)
    train_and_test["user_id_counts"] = train_and_test["user_id_counts"].astype(
        "float64"
    )
    # train_and_test["user_id_counts_ratio"] = (
    #    train_and_test["user_id_counts"] / train_and_test["user_id_counts"].max()
    # )
    train = train_and_test[: len(train)]
    test = train_and_test[len(train) :].drop(["score"], axis=1)
    train_and_test = pd.concat([train, test])
    anime_id_counts = train_and_test["anime_id"].value_counts()
    train_and_test["anime_id_counts"] = train_and_test["anime_id"].map(anime_id_counts)
    train_and_test["anime_id_counts"] = train_and_test["anime_id_counts"].astype(
        "float64"
    )
    # train_and_test["anime_id_counts_ratio"] = (
    #    train_and_test["anime_id_counts"] / train_and_test["anime_id_counts"].max()
    # )

    train = train_and_test[: len(train)]
    test = train_and_test[len(train) :].drop(["score"], axis=1)
    train["score"] = train["score"].astype("int64")

    return train, test


def count_encoding(anime_meta, features):
    for col in features:
        encoder = anime_meta[col].value_counts()
        anime_meta[f"{col}_count"] = anime_meta[col].map(encoder)
        anime_meta[f"{col}_count"] = anime_meta[f"{col}_count"].astype("float64")

    return anime_meta


def aired_to_time_features(anime_meta):
    time_features = anime_meta["aired"].apply(parse_date_range)
    time_features = pd.DataFrame(time_features.to_list())
    anime_meta = pd.concat([anime_meta, time_features], axis=1)
    anime_meta["season"] = anime_meta["season"].astype("category")

    return anime_meta


def create_type_source_features(anime_meta):
    anime_meta["type_source"] = anime_meta["type"] + "_" + anime_meta["source"]
    anime_meta["type_source"] = anime_meta["type_source"].astype("category")

    return anime_meta


def label_encoding(anime_meta, enc_categorical):
    for category in enc_categorical:
        le = LabelEncoder()
        le.fit(anime_meta[category])
        anime_meta[category] = le.transform(anime_meta[category])

    return anime_meta


def target_encoding(cfg, enc_categorical, train_X, train_y, valid_X=None, test=None):
    for col in enc_categorical:
        enc_auto = TargetEncoder(
            smooth="auto",
            target_type="continuous",
            cv=5,
            random_state=cfg["general"]["seed"],
        )
        col_train = np.array(train_X[col]).reshape(-1, 1)
        train_X[f"{col}_ts"] = enc_auto.fit_transform(col_train, train_y)
        if test is None:
            col_valid = np.array(valid_X[col]).reshape(-1, 1)
            valid_X[f"{col}_ts"] = enc_auto.transform(col_valid)
        else:
            col_test = np.array(test[col]).reshape(-1, 1)
            test[f"{col}_ts"] = enc_auto.transform(col_test)

    if test is None:
        return train_X, valid_X
    else:
        return test


def user_groupby(train, test):
    train_and_test = pd.concat([train, test])

    agg_cols = ["min", "max", "mean", "std"]

    numeric_col = [
        "members",
        "watching",
        "completed",
        "on_hold",
        "dropped",
        "plan_to_watch",
    ]
    # ["members_digit_count", "watching_digit_count", "completed_digit_count", "on_hold_digit_count", "dropped_digit_count", "plan_to_watch_digit_count",]
    # ["watching_ratio" "completed_ratio" "on_hold_ratio", "dropped_ratio", "plan_to_watch_ratio",]

    for col in numeric_col:
        grp_df = train_and_test.groupby("user_id")[col].agg(agg_cols)
        grp_df.columns = [f"user_{col}_{agg}" for agg in agg_cols]
        train = train.merge(grp_df, on="user_id", how="left")
        test = test.merge(grp_df, on="user_id", how="left")

    return train, test


def user_groupby_genres(train, test, genres_df):
    train_and_test = pd.concat([train, test])

    agg_cols = ["min", "max", "mean", "std"]

    numeric_col = genres_df.columns[1:]

    for col in numeric_col:
        grp_df = train_and_test.groupby("user_id")[col].agg(agg_cols)
        grp_df.columns = [f"user_{col}_{agg}" for agg in agg_cols]
        train = train.merge(grp_df, on="user_id", how="left")
        test = test.merge(grp_df, on="user_id", how="left")

    return train, test


def cat_groupby(anime_meta):
    agg_cols = ["min", "max", "mean", "std"]

    cat_col = ["type", "source", "rating"]

    numeric_col = [
        "members",
        "watching",
        "completed",
        "on_hold",
        "dropped",
        "plan_to_watch",
    ]

    for cat in cat_col:
        for num in numeric_col:
            grp_df = anime_meta.groupby(cat)[num].agg(agg_cols)
            grp_df.columns = [f"{cat}_{num}_{agg}" for agg in agg_cols]
            anime_meta = anime_meta.merge(grp_df, on=cat, how="left")
            print(grp_df)
            print(anime_meta)

    return anime_meta


def anime2vec(cfg, train_X, valid_X=None, test=None):
    if test is None:
        train_X, valid_X = add_w2v_features_with_score(cfg, train_X, valid_X)
        return train_X, valid_X
    else:
        _, _, test = add_w2v_features_with_score(cfg, train_X, valid_X, test)
        return test


def anime2vec_without_score(cfg, train, test):
    test["score"] = 0
    train_test_df = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test_df = add_w2v_features_without_score(cfg, train_test_df)
    train_df = train_test_df[train_test_df["score"] != 0].copy().reset_index(drop=True)
    test_df = train_test_df[train_test_df["score"] == 0].copy().reset_index(drop=True)
    test_df = test_df.drop(["score"], axis=1)
    return train_df, test_df


def anime2vec_with_genres(cfg, train, test, anime_meta):
    test["score"] = 0
    train_test_df = pd.concat([train, test], axis=0).reset_index(drop=True)
    train_test_df = train_test_df.merge(
        anime_meta[["anime_id", "genres"]], on="anime_id", how="left"
    )
    train_test_df = add_w2v_features_with_genres(cfg, train_test_df)
    train_df = train_test_df[train_test_df["score"] != 0].copy().reset_index(drop=True)
    test_df = train_test_df[train_test_df["score"] == 0].copy().reset_index(drop=True)
    train_df = train_df.drop(["genres"], axis=1)
    test_df = test_df.drop(["score", "genres"], axis=1)
    print(train_df)
    print(test_df)
    return train_df, test_df

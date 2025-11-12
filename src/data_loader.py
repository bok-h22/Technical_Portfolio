import re
import html
import logging
from pathlib import Path
from typing import List, Tuple, Any, Callable, Dict

import pandas as pd
from cornac.eval_methods import BaseMethod
from sklearn.model_selection import train_test_split

def normalize_category_name(name: str) -> str:
    if not isinstance(name, str):
        return name
        
    name = html.unescape(name)
    name = re.sub(r"['\"\\\-\s/&]", "", name.lower())
    
    correction_map = {
        "childrens": "children",
        "childrns": "children",
        "childrn": "children",
        "childerns": "children"
    }
    return correction_map.get(name, name)

def load_movielens_1m(path: str) -> pd.DataFrame:
    p = Path(path)
    ratings_cols = ["uid", "iid", "rating", "timestamp"]
    ratings = pd.read_csv(
        p / "ratings.dat", sep="::", header=None, names=ratings_cols,
        engine="python", encoding="Latin-1"
    )
    items_cols = ["iid", "item_name", "categories"]
    items = pd.read_csv(
        p / "movies.dat", sep="::", header=None, names=items_cols,
        engine="python", encoding="Latin-1"
    )

    items["categories"] = items["categories"].str.split('|')
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit='s')
    
    exploded_cats = items.explode("categories")
    exploded_cats["normalized_cat"] = exploded_cats["categories"].apply(normalize_category_name)
    normalized_items = exploded_cats.groupby("iid").agg({
        "item_name": "first",
        "normalized_cat": lambda x: list(x)
    }).rename(columns={"normalized_cat": "categories"}).reset_index()

    movielens_df = pd.merge(ratings, normalized_items, how="inner", on="iid")
    
    return movielens_df[["uid", "iid", "rating", "timestamp", "item_name", "categories"]]

def load_bookcrossing(path: str) -> pd.DataFrame:
    p = Path(path)
    raw_data = pd.read_csv(p / "BookCrossingThemes.csv", sep=";")
    
    ratings_cols = ["uid", "iid", "rating"]
    items_cols = ["iid", "item_name", "categories"]
    
    ratings = raw_data[["User-ID", "ISBN", "Book-Rating"]]
    items = raw_data[["ISBN", "Book-Title", "Theme"]].drop_duplicates("ISBN")
    
    ratings.columns = ratings_cols
    items.columns = items_cols

    bookcrossing_df = pd.merge(ratings, items, how="inner", on="iid")
    return bookcrossing_df

def load_yahoo_music(path: str) -> pd.DataFrame:
    p = Path(path)
    ratings_cols = ['uid', 'iid', 'rating']
    
    files_to_load = list(p.glob("train_*.txt")) + list(p.glob("test_*.txt"))
    
    if not files_to_load:
        raise FileNotFoundError(f"'{path}' 경로에서 train/test .txt 파일을 찾을 수 없습니다.")
        
    df_list = [
        pd.read_csv(
            f, sep='\t', header=None, names=ratings_cols, engine='pyarrow'
        )
        for f in files_to_load
    ]
    ratings_df = pd.concat(df_list, ignore_index=True)

    hierarchy_df = pd.read_csv(
        p / "genre-hierarchy.txt",
        sep='\t', header=None, usecols=[0, 1, 3],
        names=['genre_id', 'parent_genre_id', 'genre_name'],
        engine='pyarrow', dtype={0: int, 1: int, 3: str}
    )

    hierarchy_df['genre_name'] = hierarchy_df['genre_name'].apply(normalize_category_name)
    genre_to_parent_map = hierarchy_df.set_index('genre_id')['parent_genre_id']
    id_to_name_map = hierarchy_df.set_index('genre_id')['genre_name']

    item_df = pd.read_csv(
        p / "song-attributes.txt",
        sep='\t', header=None, usecols=[0, 3], 
        names=['iid', 'genre_id'], engine='pyarrow'
    )

    merged_df = pd.merge(ratings_df, item_df, on='iid', how='inner')
    merged_df['parent_genre_id'] = merged_df['genre_id'].map(genre_to_parent_map)
    merged_df = merged_df.dropna(subset=['parent_genre_id'])
    merged_df['parent_genre_id'] = merged_df['parent_genre_id'].astype(int)
    
    filtered_df = merged_df[merged_df['parent_genre_id'] != 0].copy()
    filtered_df['categories'] = filtered_df['parent_genre_id'].map(id_to_name_map)
    filtered_df['categories'] = filtered_df['categories'].fillna('Unknown')

    final_cols = ['uid', 'iid', 'rating', 'categories']
    return filtered_df[final_cols]

DATASET_LOADERS: Dict[str, Callable[[str], pd.DataFrame]] = {
    "movielens_1m": load_movielens_1m,
    "book_crossing": load_bookcrossing,
    "yahoo_music": load_yahoo_music
}

def split_data(
    df: pd.DataFrame, 
    time_aware: bool = False, 
    test_size: float = 0.2, 
    val_size: float = 0.0,
    min_interactions: int = 20,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_filtered = df.groupby('uid').filter(lambda x: len(x) >= min_interactions)
    
    if df_filtered.empty:
        raise ValueError(
            f"최소 상호작용 {min_interactions}회 이상인 사용자가 없음"
        )
    
    splits = []
    val_ratio = val_size / (1.0 - test_size)

    if time_aware:
        if "timestamp" not in df.columns:
            raise ValueError("'time_aware=True'이지만 'timestamp' 컬럼이 없음")
        
        # 시간 기반 분할 진행 시작
        for _, group in df_filtered.groupby('uid'):
            group_sorted = group.sort_values('timestamp', ascending=True)
            
            train_val_df, test_df = train_test_split(
                group_sorted,
                test_size=test_size, 
                shuffle=False,
            )
            if val_ratio > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_ratio,
                    shuffle=False,
                )
            else:
                train_df = train_val_df 
                val_df = group_sorted.iloc[0:0] 
            
            splits.append((train_df, val_df, test_df))
    else:
        for _, group in df_filtered.groupby('uid'):
            train_val_df, test_df = train_test_split(
                group,
                test_size=test_size, 
                shuffle=True,
                random_state=random_seed
            )
            if val_ratio > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_ratio,
                    shuffle=True,
                    random_state=random_seed
                )
            else:
                train_df = train_val_df
                val_df = group.iloc[0:0]
                
            splits.append((train_df, val_df, test_df))

    logging.info("분할된 데이터프레임 결합")
    train_list, val_list, test_list = zip(*splits)
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    logging.info("분할된 데이터프레임 결합 완료")
    
    return train_df, val_df, test_df

def load_data(
    dataset_name: str, 
    path: str,
    split_config: Dict,
    seed: int, 
    exclude_unknowns: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, BaseMethod, Dict[int, List[str]]]:
    loader_func = DATASET_LOADERS.get(dataset_name)
    if not loader_func:
        raise ValueError(f"알 수 없는 데이터셋: {dataset_name}")
    
    logging.info(f"Raw 데이터 로드 시작: '{dataset_name}' from '{path}'")
    raw_df = loader_func(path=path)
    
    logging.info(f"데이터 분할 진행 (최소 상호작용 수={split_config.get('min_interactions')})...")
    train_df, val_df, test_df = split_data(
        raw_df,  
        time_aware=split_config.get("time_aware", False),
        test_size=split_config.get("test_size", 0.2),
        val_size=split_config.get("val_size", 0.0),
        min_interactions=split_config.get("min_interactions", 20),
        random_seed=seed
    )
    
    logging.info("Cornac 평가 방법 설정")
    train_data = list(train_df[["uid", "iid", "rating"]].itertuples(index=False, name=None))
    test_data = list(test_df[["uid", "iid", "rating"]].itertuples(index=False, name=None))
    
    eval_method = BaseMethod.from_splits(
        train_data=train_data, 
        test_data=test_data,
        val_data=None, 
        rating_threshold=1.0,  
        exclude_unknowns=exclude_unknowns,
        random_seed=seed,
        verbose=verbose
    )
    
    logging.info("아이템 메타 정보")
    item_df = raw_df[['iid', 'categories']].drop_duplicates(subset=['iid']).copy()
    item_df['iidx'] = item_df['iid'].map(eval_method.global_iid_map)
    item_df.dropna(subset=['iidx'], inplace=True) 
    item_df['iidx'] = item_df['iidx'].astype(int)

    item_info = item_df.set_index('iidx')['categories'].to_dict()
    
    logging.info("데이터 로드 및 준비 완료.")
    
    return raw_df, eval_method, item_info
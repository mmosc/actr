import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import ranky as rk
from paths.paths import *

from config.config import CALC_REWARD, USE_CONTENT

from recbole.utils import (
    init_logger,
    get_model,
    init_seed,
)


from recbole.data import (
    create_dataset,
    data_preparation,
)

def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


def aggregate_scores(score_dfs, return_softmax=True, weights='no', method='scoring', number_of_items=101836):
    """
    Aggregates scores of items into a single score per item

    Attributes
    ----------
    score_dfs: list
      a list of score dataframes for items
    return_softmax: Boolean
      if True softmax scores are returned, default: true
    weights: list
      list of weighting for each score
    method: string
        the aggregation method in [rank, scoring, scoring_no_softmax, gauss_normalization, range_normalization]
    number_of_items
        the number_of_items to be scored

    returns aggregated scores

    """

    scores = pd.Series(dtype='float64')
    if weights == 'no':
        weights = [1 for ii in range(len(score_dfs))]
    i = 0
    for score in score_dfs:
        score = score[score > 0]
        if method == 'scoring':
            scores = pd.concat([scores, softmax(score) * weights[i]])

        if method == 'scoring_no_softmax':
            scores = pd.concat([scores, score * weights[i]])

        if method == 'gauss_normalization':
            std = score.std(ddof=0)
            if std != 0:
                scores = pd.concat([scores, ((score - score.mean()) / std) * weights[i]])
            else:
                scores = pd.concat([scores, score * weights[i]])

        if method == 'range_normalization':
            ran = score.max() - score.min()
            if ran != 0:
                scores = pd.concat([score, (score / ran) * weights[i]])
            else:
                scores = pd.concat([scores, score * weights[i]])

        if method == 'ranking':
            score = rk.rank(score, reverse=True, ascending=False)
            scores = pd.concat([scores, ((score - score.max()) + number_of_items) * weights[i]])

        i += 1

    scores = scores.to_frame()

    scores['item'] = scores.index
    scores = scores.reset_index()
    if method == 'scoring' or method == 'scoring_no_softmax' or method == 'range_normalization' or method == 'gauss_normalization':
        scores = scores.groupby('index')[0].sum()

    if method == 'ranking':
        scores = scores.groupby('index')[0].sum()

    if return_softmax:
        scores = softmax(scores)

    return scores


def get_users_embeddings(model, rating_matrix, device='cpu'):
    """
    Given a RecBole MultVAE model,
    get the embedding of
    all users as a matrix
    of dimensions
    rating_matrix_users x self.lat_dim / 2.

    Params
    :model: the RecBole model instance
    :rating_matrix:
    :device: the device on which the model should be loaded

    Returns:
    :mu: The (mean of the distribution of the) latent representation
        according to the pretrained MultVAE.
    """
    model = model.cpu()
    rating_matrix = torch.from_numpy(rating_matrix).float().to('cpu')
    h = F.normalize(rating_matrix)
    h = model.encoder(h).detach()
    mu = h[:, : int(model.lat_dim / 2)]
    return mu


def forward_custom(model, rating_matrix, device='cuda'):
    """
    Given a RecBole MultVAE model,
    get the embedding of
    all users as a matrix
    of dimensions
    rating_matrix_users x self.lat_dim / 2.

    Params
    :model: the RecBole model instance
    :rating_matrix:
    :device: the device on which the model should be loaded

    Returns:
    :z, mu, logvar: The parameters of the latent rep of MultVAE
    """
    model = model.to(device)
    rating_matrix = torch.from_numpy(rating_matrix).float().to(device)
    h = F.normalize(rating_matrix)
    h = model.encoder(h)
    mu = h[:, : int(model.lat_dim / 2)]
    logvar = h[:, int(model.lat_dim / 2):]
    z = model.reparameterize(mu, logvar)
    z = model.decoder(z)
    return z, mu, logvar


def get_key(my_dict, val):
    for key, value in my_dict.items():
        if val == value:
            return key


def ratings_to_vae_ids_2(model, rating_matrix, dictionary_track_id, dataset):
    """
    Convert item-ratings from rating matrix to the items to the vae ids of items
    """
    ratings = rating_matrix.copy()
    dictionary_track_id_ = dictionary_track_id.reset_index()
    dictionary_track_id_.index += 1
    relevant_index = dictionary_track_id_['track_id'].astype(str).values
    field = dataset.iid_field
    relevant_index = dataset.token2id(field, relevant_index)
    ratings[:, relevant_index] = rating_matrix.copy()[:, dictionary_track_id_.index.tolist()]
    return ratings


def forward_custom_gru(model, rating_matrix, item_seq_len, device='cuda'):
    """
    adapt forward pass for gru for our situation
    """
    model = model.to(device)
    rating_matrix = torch.from_numpy(rating_matrix).long().to(device)
    item_seq_emb = model.item_embedding(rating_matrix)

    item_seq_emb_dropout = model.emb_dropout(item_seq_emb)
    gru_output, _ = model.gru_layers(item_seq_emb_dropout)
    gru_output = model.dense(gru_output)
    rating_matrix.size()
    item_seq_len = item_seq_len.to(device)

    seq_output = model.gather_indexes(gru_output, item_seq_len - 1)
    test_items_emb = model.item_embedding.weight
    scores = torch.matmul(
        seq_output, test_items_emb.transpose(0, 1)
    )
    return scores


def load_data_and_model(
        model_file,
        vae=False,
        gru=False,
        device='cpu',
):
    r"""
    Adapted version of the RecBole loader, in order to be able to load on cpu.
    Also see
    https://recbole.io/docs/_modules/recbole/quick_start/quick_start.html#load_data_and_model

    Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file, map_location=torch.device(device))
    config = checkpoint["config"]
    if vae:
        config.final_config_dict['data_path'] = f'{BASE_FOLDER}/actr_data/actr_data_bin'
    elif gru:
        config.final_config_dict['data_path'] = f'{BASE_FOLDER}//actr_data/actr_data_gru'
    config.final_config_dict['device'] = device
    config.final_config_dict['eval_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 'none'}
    config.final_config_dict['eval_neg_sample_args']['distribution'] = 'uniform'
    config.final_config_dict['eval_neg_sample_args']['sample_num'] = 5
    config.final_config_dict['checkpoint_dir'] = 'saved'
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    # logger = getLogger()
    # logger.info(config)

    dataset = create_dataset(config)
    # logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data


def load_file(filename):
    """
    loads for a user the listening data and calculates durations, gaps between songs, sessions
    """
    col_names = ['user', 'track_id', 'album_id', 'timestamp']
    events = pd.read_csv(filename, names=col_names,
                         quoting=3, sep="\t", header=None, encoding='utf-8'
                         )
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    return events


def preprocess(events, use_content=USE_CONTENT, calc_reward=CALC_REWARD):
    """
    preprocess user-data
    """
    #events=events.merge(df_users, left_on='user', right_on='user_id')
    #events=events.merge(df_albums, on='album_id')
    #events=events.merge(df_tracks, on='track_id')
    events["prev_timestamp"] = events.groupby("user")["timestamp"].shift()
    events["gap"] = events["timestamp"] - events["prev_timestamp"]
    events["new_session"] = events["gap"] > pd.Timedelta("30min")
    events["new_session_int"] = events["new_session"].astype(int)
    events["session"] = events.groupby("user")["new_session_int"].cumsum()
    events["session_duration"] = events.groupby(["user", "session"])["timestamp"].transform(
        lambda x: x.iloc[-1] - x.iloc[0])
    #events["item"] = list(zip(events["track"], events["artist_x"])) #, events["album"]))
    events["item"] = events["track_id"]
    events["all_pos"] = 1
    if len(events) > 0:
        user = events["user"].iloc[0]
    events.index = [user for _ in range(len(events))]

    if calc_reward:
        dur_cols = ["track", "artist", "playcount", "track_listeners", "duration"]
        durations = pd.read_csv(datadir + "LFM-2b_track_artist_pc_ls_dur.txt", sep="\t", names=dur_cols)
        durations["item"] = list(zip(durations["track"], durations["artist"]))
        durations["duration_td"] = pd.to_timedelta(durations["duration"], unit="ms")
        durations = durations.set_index("item").drop(columns=["track", "artist", "playcount", "track_listeners"])

        events = events.merge(durations, on="track_id", how="left")
        events["timestamp_end"] = events["timestamp"] + events["duration_td"]
        events["timestamp_start_next"] = events["timestamp"].shift(-1)
        events["play_duration"] = (events["timestamp_start_next"] - events["timestamp"]).dt.seconds * 1000
        events["gap"] = (events["timestamp_start_next"] - events["timestamp_end"]).dt.seconds
        events["min_duration"] = events[["play_duration", "duration"]].min(axis=1)
        events["play_ratio"] = events["min_duration"] / events["duration"]

        #this function assigns reward 1 to all listened song, user's popularity values for songs
        def reward_function(play_duration):
            if play_duration >= 30000:
                return 1
            elif play_duration > 12000 and play_duration < 30000:
                return 1
            else:  # play_ratio <= 0.33:
                return 1

        events["reward"] = events["play_duration"].apply(reward_function)

    return events


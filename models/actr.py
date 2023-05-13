# Contains the ACT-R models
# Code from https://github.com/socialcomplab/recsys21-relistening-actr/
import numpy as np
import pandas as pd
from scipy import stats, special
import operator
from models.bpr import bpr_item
from utils.utils import aggregate_scores, softmax


class DecayFitterMixin:
    def fit(self, events):
        delta = events.groupby(["user", "item"])["timestamp"].diff().dropna().dt.total_seconds() / 3600
        delta = delta[delta != 0]
        delta_bins = delta.value_counts()
        log_x = np.log10(delta_bins.index.tolist())
        log_y = np.log10(delta_bins.values.tolist())
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        self.decay = -slope
        return slope


class ScoreToRecommenderMixin:
    """Requires a score(self, user_events) function."""

    def recommend_next(self, user_events):
        item_scores = self.score(user_events)
        return item_scores.idxmax()

    def recommend(self, user_events, topn):
        item_scores = self.score(user_events)
        return item_scores.nlargest(topn).index.tolist()


class BaseLevelComponent(ScoreToRecommenderMixin, DecayFitterMixin):
    """Models occurence."""

    def __init__(self, decay=0.5, time_col="timestamp"):
        self.decay = decay
        self.time_col = time_col

    def __str__(self):
        if self.decay == 0.5:
            return type(self).__name__
        else:
            return type(self).__name__ + str(self.decay)

    def score(self, user_events):
        user_events = user_events.copy()
        ts_ref = user_events["timestamp"].iloc[-1]

        user_events["ts_diff"] = (-(user_events[self.time_col] - ts_ref) + pd.Timedelta(
            "1hour")).dt.total_seconds() / 3600
        bll_scores = user_events.groupby("item", sort=False)["ts_diff"].apply(
            lambda x: np.sum(np.power(x.values, -self.decay)))
        return bll_scores


class AssociativeComponent(ScoreToRecommenderMixin):
    """Models co-occurence."""

    def __init__(self, session_col="session"):
        self.session_col = session_col

    def __str__(self):
        return type(self).__name__

    def score(self, user_events):
        context_item = user_events["item"].iloc[-1]
        context_sessions = set(user_events[user_events["item"] == context_item][self.session_col].unique())

        num_sessions = user_events[self.session_col].nunique()
        probability_of_item = user_events.groupby("item")[self.session_col].nunique() / num_sessions

        def overlap(sessions):
            return len(set(sessions.unique()).intersection(context_sessions))

        overlap_sessions = user_events.groupby("item")[self.session_col].apply(overlap)
        condidtional_probability = overlap_sessions / len(context_sessions)

        return condidtional_probability / probability_of_item


class PartialMatchingComponent(ScoreToRecommenderMixin):
    """Models similarity."""

    def __init__(self, name=None, feature_cols=None, similarity_function=np.dot):
        self.name = name if name else type(self).__name__
        self.feature_cols = feature_cols
        self.similarity_function = similarity_function

    def __str__(self):
        return self.name

    def score(self, user_events):
        context_features = user_events[self.feature_cols].iloc[-1]

        items = user_events.drop_duplicates(subset=["item"])
        item_index = items["item"].values
        cand_features = items[self.feature_cols].values

        pm_scores = self.similarity_function(cand_features, context_features)
        return pd.Series(data=pm_scores, index=item_index)


class ValuationComponent(ScoreToRecommenderMixin):
    """Models affect."""

    def __init__(self, name=None, learning_rate=0.2, initial_valuation=0, reward_col="reward"):
        self.name = name if name else type(self).__name__
        self.learning_rate = learning_rate
        self.initial_valuation = initial_valuation
        self.reward_col = reward_col

    def __str__(self):
        return self.name

    def score(self, user_events):
        def update_valuation(prev, reward=1, lr=0.05):
            return prev + lr * (reward - prev)

        def aggreagte_valuation(reward_s):
            valuation = self.initial_valuation
            for reward in reward_s.values:
                valuation = update_valuation(valuation, reward, self.learning_rate)
            return valuation

        valuation_scores = user_events.groupby("item")[self.reward_col].apply(aggreagte_valuation)
        return valuation_scores


class NoiseComponent(ScoreToRecommenderMixin):
    """Adds randomnes."""

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def __str__(self):
        return type(self).__name__

    def score(self, user_events):
        return pd.Series(data=self.rng.random(user_events["item"].nunique()), index=user_events["item"].unique())


class ActrRecommender(ScoreToRecommenderMixin):
    """Combines multiple components."""

    def __init__(self, components, weights=None, softmax=True, name=None, use_normalize_trick=False):
        self.components = components
        self.weights = weights if weights else [1] * len(components)
        self.softmax = softmax
        self.name = name if name else type(self).__name__ + "(" + ",".join(map(str, self.components)) + ")"
        self.use_normalize_trick = use_normalize_trick

    def __str__(self):
        return self.name

    def score(self, user_events):
        scores = pd.Series()

        for comp, w_c in zip(self.components, self.weights):
            comp_scores = comp.score(user_events)
            if self.softmax:
                if self.use_normalize_trick:
                    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
                    comp_scores = comp_scores - np.max(comp_scores)
                comp_scores = special.softmax(comp_scores)
            comp_scores = comp_scores * w_c
            scores = scores.combine(comp_scores, operator.add, 0)
            # print(scores)
        return scores


# This defines the Act-R score of bll+spreading, valu, bpr and pm

def actr_score(bll, assoc, val, train, item_embedding_matrix_spotify=None, item_dictionary_spotify=None,
               item_dictionary_df_spotify=None, item_embedding_matrix=None, item_dictionary=None,
               item_dictionary_df=None,
               valuation='no_valu', bpr='no_bpr', pm='no_pm', non_spotify_feat=[]):
    """
    """
    score_bll = bll.score(train)
    score_assoc = assoc.score(train)
    score_list = [score_assoc, score_bll]
    if valuation == 'valu':
        score_val = val.score(train)
        score_list.append(score_val)

    train_items = train["item"].values.tolist()

    if bpr == 'bpr':
        bpr = bpr_item(item_embedding_matrix, item_dictionary, item_dictionary_df, non_spotify_feat=non_spotify_feat)
        score_bpr = bpr.score(train_items=train_items)
        score_list.append(score_bpr)
    if pm == 'pm':
        pm_it = bpr_item(item_embedding_matrix_spotify, item_dictionary_spotify, item_dictionary_df_spotify)
        # pm_it=pm_item(item_embedding_matrix_spotify, item_dictionary_spotify, item_dictionary_df_spotify, item_dictionary, non_spotify_feat=non_spotify_feat)
        score_pm = pm_it.score(train_items=train_items)
        score_list.append(score_pm)

    scores = aggregate_scores(score_list, method='scoring')

    return scores


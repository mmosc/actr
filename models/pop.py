# determines most popular songs
import numpy as np
from utils.utils import softmax


class Pop:
    """
    Recommends most popular items relativ to a rating matrix

    Attributes
    ----------
    dictionary_track_id: dictionary
      a dictionary converting item ids to their position in the rating_matrix

    rating_matrix: matrix
      a matrix containing users rating for a set of items

    Methods
    ----------

      recommend_next(user_events)
        returns returns most popular items

      recommend(user_events, topn)
        return topn most popular items

    """

    def __init__(self, rating_matrix, dictionary_track_id):
        self.rating_matrix = rating_matrix
        self.dictionary_track_id = dictionary_track_id

    def __str__(self):
        return type(self).__name__

    def recommend_next(self, user_events):
        self.rec_rating = self.rating_matrix.mean(axis=0)  # needs to weighted
        self.rec_rating = np.squeeze(np.asarray(self.rec_rating))
        self.sorti = np.argsort(self.rec_rating)
        sorti = np.squeeze(np.asarray(self.sorti))[-1]
        score = softmax(self.rec_rating[sorti])
        return self.dictionary_track_id.iloc[sorti]['track_id'].values.tolist()

    def recommend(self, user_events, topn):
        self.rec_rating = self.rating_matrix.mean(axis=0)  # needs to weighted
        self.rec_rating = np.squeeze(np.asarray(self.rec_rating))
        self.sorti = np.argsort(self.rec_rating)
        sorti = np.squeeze(np.asarray(self.sorti))[-topn:]
        score = softmax(self.rec_rating[sorti])
        return self.dictionary_track_id.iloc[sorti]['track_id'].values.tolist()

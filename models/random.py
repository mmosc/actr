# randomly samples from all items
import random


class Random:
    """
    Recommends random items from a list of items

    Attributes
    ----------
    dictionary_track_id: dictionary
      a dictionary converting item ids to their position in the rating_matrix

    Methods
    ----------

      recommend_next(user_events)
        returns a randomly selected item

      recommend(user_events, topn)
        returns topn many random items

    """

    def __init__(self, dictionary_track_id):
        self.items = dictionary_track_id['track_id'].values.tolist()

    def __str__(self):
        return type(self).__name__

    def recommend_next(self, user_events):
        return random.sample(self.items, 1)

    def recommend(self, user_events, topn):
        return random.sample(self.items, topn)

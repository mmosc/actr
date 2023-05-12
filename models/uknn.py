# This the main collaborative filtering function
import numpy as np
import pandas as pd


class CF:
    """
    This class calculates item scores on basis of user-similarities

    Attributes
    ----------
    top k : int
        number of users to be considered for evaluating score
    users : list
        a list of users
    dictionary_track_id : dict
        a dictionary of tracks to track_id

    Methods
    -------
    set sim_rat(similarities, rating_matrix)
        sets similarities and rating_matrix and sets top k similarity scores
    
    get_closest_users()
        returns users with highest similarity scores

    prepare_scores(user_events)
        returns the CF score for all items based on an initial segment user_eventss
    
    get_similarity_scores()
    returns similarity scores in list format
    
    score(user_events)
        returns the CF score for all items based on an initial segment user_events_events
    
    recommend(user_events, topn)
        returns a list of topn events
    
    """

    def __init__(self, top_k, users, dictionary_track_id):
        self.top_k = top_k
        self.users = users
        self.dictionary_track_id = dictionary_track_id

    def __str__(self):
        return type(self).__name__

    def set_sim_rat(self, similarities, rating_matrix):

        self.similarities = similarities
        self.rating_matrix = rating_matrix
        # get the top k scores of the remainder
        self.closest_users_similarities = self.similarities.sort_values(ascending=False, by=0)[:self.top_k]

    def get_closest_users(self):
        # returns real names of closest users

        closest_users_ = [self.users[i] for i in self.closest_users_similarities.index.tolist()]

        return closest_users_

    def get_similarity_scores(self):
        # return similarities as a list
        return self.closest_users_similarities.tolist()

    def prepare_score(self, user_events):
        # calculates score in preparation for score dataframe for aggregation and for recommendation
        closest_users_sim_values = self.closest_users_similarities[0].to_numpy()  # transform to array
        closest_users = self.closest_users_similarities.index.tolist()

        # get users with highest similarity scores
        ratings_top = self.rating_matrix[closest_users, :]  # get ratings of closest users
        # scores for items are similarity weighted ratings, if all similarities are zero, then take most popular items
        ratings_top = ratings_top.todense()

        if np.sum(closest_users_sim_values) == 0:
            scores = ratings_top.mean(axis=0)
        else:
            scores = np.average(ratings_top, axis=0, weights=closest_users_sim_values)

        scores = np.squeeze(np.asarray(scores))

        return scores

    def score(self, user_events):
        # transform prepared scores to a dataframe to enable aggregating with other scores
        df_scores = pd.DataFrame(self.prepare_score(user_events))
        df_scores.index = self.dictionary_track_id['track_id']

        return df_scores.squeeze()

    def recommend(self, user_events, topn):
        rec_rating = self.prepare_score(user_events)  # get scores
        rec_rating = np.squeeze(np.asarray(rec_rating))
        sorti = np.argsort(rec_rating)  # sort scores
        sorti = np.squeeze(np.asarray(sorti))[-topn:]  # get top ranked items and return real name

        return self.dictionary_track_id.iloc[sorti]['track_id'].values.tolist()

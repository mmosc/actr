import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from utils.utils import get_users_embeddings


class Recency:
    """
    This class defines all recency weighted versions of CF (knn, vae) and also the ACT-R weighted similarity calculation 

    ...

    Attributes
    ----------
    algo : cf
        an algorithm, here cf
    users : list
        a list of users
    dictionary_track_id : dict
        a dictionary of tracks to track_id
    total_number_of_items : int
        the number of items
    mode: str
        knn... normal user-knn
        vae... similarity is based on similarity of user embeddings

    recency_para: float
        parameter d for recency weight

    rating_matrix: sparse matrix 
        a user-item rating_matrix

    vae_matrix: matrix
        a matrix of vae-embeddings of any rating_matrix

    model: pytorch model
        a model used for the dimensions of the number of vae-items

    scores: series 
        a series of scores to be used in the downweight modus of recen (often Actr scores)

    dataset: recbole dataset
        a dataset used to retrieve the correct item ids for recommendations

    recen: str
        recency... recency weight the initial segment
        downweight... weight the initial segment by scores 



    Methods
    -------
    get simlarity scores(train)
        returns the similarity scores of rating_matrix rows to an initial segment train

    recommend(train, prediction_length)
        returns prediction_length number recommended items based an initial segment train

    get get_closest_users_and_scores(train)
        returns closest_users and their similarity scores based on an initial segment train

   scores(train)
        returns the CF score for all items based on an initial segment train

    """

    def __init__(self, algo, users, dictionary_track_id, total_number_of_items, mode, recency_para, rating_matrix,
                 vae_matrix, model, scores, dataset, recen='recency'):
        self.total_number_of_items = total_number_of_items
        self.dictionary_track_id = dictionary_track_id
        self.algo = algo
        self.mode = mode
        self.rating_matrix = rating_matrix
        self.vae_matrix = vae_matrix
        self.recency_para = recency_para
        self.users = users
        self.model = model
        self.recen = recen
        self.scores = scores
        self.dataset = dataset

    def get_similarity_scores(self, train):
        # weights train items by their recency
        if self.recen == 'recency':
            if self.mode == 'vae':
                i = 0
                array = np.zeros(self.model.n_items)
                field = self.dataset.iid_field
                for ii, row in train.iterrows():
                    relevant_index = self.dataset.token2id(field, '{}'.format(row['track_id']))
                    array[relevant_index] += (len(train) - i) ** (-self.recency_para)
                    i += 1
                array = array.reshape(-1, array.shape[0])
            if self.mode == 'knn':
                i = 0
                array = np.zeros(self.model.n_items - 1)
                for ii, row in train.iterrows():
                    relevant_index = self.dictionary_track_id.index[
                        self.dictionary_track_id['track_id'] == row['track_id']].tolist()
                    array[relevant_index] += (len(train) - i) ** (-self.recency_para)
                    i += 1
        # weights train items by scores (ACTR in many cases)
        if self.recen == 'downweight':

            if self.mode == 'knn':
                array = np.zeros(self.model.n_items - 1)
                for ii, row in train.iterrows():
                    relevant_index = self.dictionary_track_id.index[
                        self.dictionary_track_id['track_id'] == row['track_id']].tolist()
                    array[relevant_index] += self.scores[self.scores.index == row['track_id']]
                    # array=array.reshape(-1,array.shape[0])

            if self.mode == 'vae':
                array = np.zeros(self.model.n_items)
                field = self.dataset.iid_field
                for ii, row in train.iterrows():
                    relevant_index = self.dataset.token2id(field, '{}'.format(row['track_id']))
                    array[relevant_index] += self.scores[self.scores.index == row['track_id']]
                array = array.reshape(-1, array.shape[0])
        # fetches vae embeddings or rating matrix
        if self.mode == 'vae':
            matrix = self.vae_matrix
            array = get_users_embeddings(self.model, array)
        if self.mode == 'knn':
            matrix = self.rating_matrix
            array = csr_matrix(array)

        # calculates cosine_similarity of manipulated train and the rating_matrix/vae matrix and returns scores
        w_normalized = normalize(matrix, norm='l2', axis=1)
        array_normalized = normalize(array, norm='l2', axis=1)
        similarities_sparse_c = w_normalized.dot(array_normalized.T)

        if self.mode == 'vae':
            similarities_c = pd.DataFrame(similarities_sparse_c)
        if self.mode == 'knn':
            similarities_c = pd.DataFrame(similarities_sparse_c.todense())

        self.similarities_c = similarities_c.sort_index().reset_index(drop=True)
        return similarities_c

    def recommend(self, train, prediction_length):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)

        res = self.algo.recommend(train, prediction_length)
        return res

    def get_closest_users_and_scores(self, train):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)
        closest_users = self.algo.get_closest_users()
        return closest_users, self.algo.closest_users_similarities

    def score(self, train):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)
        return self.algo.score(train)




class MartaRecency:
    """
    """

    def __init__(self, algo, dictionary_track_id, rating_matrix, model, scores):
        self.model = model
        self.dictionary_track_id = dictionary_track_id
        self.scores = scores
        self.rating_matrix = rating_matrix
        self.algo = algo

    def get_similarity_scores(self, train):

        array = np.zeros(self.model.n_items - 1)
        for ii, row in train.iterrows():
            relevant_index = self.dictionary_track_id.index[
                self.dictionary_track_id['track_id'] == row['track_id']].tolist()
            array[relevant_index] += self.scores[self.scores.index == row['track_id']]
            # array=array.reshape(-1,array.shape[0])
        matrix = self.rating_matrix
        array = csr_matrix(array)

        # calculates cosine_similarity of manipulated train and the rating_matrix/vae matrix and returns scores
        w_normalized = normalize(matrix, norm='l2', axis=1)
        array_normalized = normalize(array, norm='l2', axis=1)
        similarities_sparse_c = w_normalized.dot(array_normalized.T)

        similarities_c = pd.DataFrame(similarities_sparse_c.todense())

        self.similarities_c = similarities_c.sort_index().reset_index(drop=True)
        return similarities_c

    def recommend(self, train, prediction_length):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)

        res = self.algo.recommend(train, prediction_length)
        return res

    def get_closest_users_and_scores(self, train):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)
        closest_users = self.algo.get_closest_users()
        return closest_users, self.algo.closest_users_similarities

    def score(self, train):
        self.get_similarity_scores(train)
        self.algo.set_sim_rat(self.similarities_c, self.rating_matrix)
        return self.algo.score(train)

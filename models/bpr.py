import pandas as pd
from sklearn.preprocessing import normalize
from utils.utils import get_key


class bpr_item:
    """
    This class implements the bpr-based item based recommender


    Attributes
    ----------
    item_embedding_matrix: matrix
      a matrix of item embeddings (dim number of items * embedding dimension)

    item_dictionary: dictionary
      a dictionary converting item ids to their position in the item_embedding_matrix

    item_dictionary_df: dataframe
      a dataframe converting item ids to their position in the item item_embedding_matrix

    Methods
    ----------
      get_similarity_scores(train_items)
        returns similarity scores of items with respect to the last element of initial_segment

      recommend(train_items, prediction_length)
        returns prediction_length many recommended items conditional on initial_segment

      score(train_items)
        fetches similarity scores for the initial_segment's last element and returns prepared scores for aggregation with other algorithms

    """

    def __init__(self, item_embedding_matrix, item_dictionary, item_dictionary_df):
        self.item_dictionary = item_dictionary
        self.item_embedding_matrix = item_embedding_matrix
        self.item_dictionary_df = item_dictionary_df

    def get_similarity_scores(self, train_items):
        context_item = train_items[-1]
        item_index = self.item_dictionary[context_item]
        array = self.item_embedding_matrix[item_index]
        array = array.reshape(-1, array.shape[0])
        array_normalized = normalize(array, norm='l2', axis=1)
        w_normalized = normalize(self.item_embedding_matrix, norm='l2', axis=1)
        similarities_sparse_c = w_normalized.dot(array_normalized.T)  # score obtained
        similarities_sparse_c = pd.DataFrame(similarities_sparse_c)
        similarities_sparse_c.at[item_index, 0] = -1  # the train item itself should not be recommended
        return similarities_sparse_c

    def recommend(self, train_items, prediction_length):
        similarities_sparse_c = self.get_similarity_scores(train_items)
        resi = similarities_sparse_c.nlargest(prediction_length, columns=0).index.tolist()
        res = [int(get_key(self.item_dictionary, ii)) for ii in resi]
        return res

    def score(self, train_items):
        similarities_sparse_c = self.get_similarity_scores(train_items)
        similarities = pd.merge(similarities_sparse_c, pd.DataFrame(self.item_dictionary_df), left_index=True,
                                right_on='item_id')
        similarities.index = similarities['id']
        similarities = similarities.drop(['id', 'item_id'], axis=1)
        return similarities.squeeze()

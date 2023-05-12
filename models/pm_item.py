# Ignore for now
import pandas as pd
from sklearn.preprocessing import normalize
from utils.utils import get_key

# This is an implementation of the partial matching algorithm
class pm_item:
    def __init__(self, item_embedding_matrix, item_dictionary, item_dictionary_df, item_dictionary_all,
                 non_spotify_feat=[]):
        self.item_dictionary = item_dictionary
        self.item_embedding_matrix = item_embedding_matrix
        self.item_dictionary_df = item_dictionary_df
        self.item_dictionary_all = item_dictionary_all
        self.non_spotify_feat = non_spotify_feat

    def get_similarity_scores(self, train_items):
        context_item = train_items[-1]
        item_index = self.item_dictionary[context_item]

        array = self.item_embedding_matrix[item_index]

        array = array.reshape(-1, array.shape[0])

        # l = [i for i in range(self.item_embedding_matrix.shape[0])]
        # l.pop(item_index)
        # w_normalized = normalize(self.item_embedding_matrix[l, :], norm='l2', axis=1)
        array_normalized = normalize(array, norm='l2', axis=1)
        # array_normalized=array
        w_normalized = normalize(self.item_embedding_matrix, norm='l2', axis=1)
        similarities_sparse_c = w_normalized.dot(array_normalized.T)  # score obtained
        # similarities_sparse_c=np.insert(similarities_sparse_c, item_index, 0)
        # similarities_sparse_c[item_index]=0

        similarities_sparse_c = pd.DataFrame(similarities_sparse_c)

        similarities_sparse_c.at[item_index, 0] = -1  # the train item itself should not be recommended
        if self.non_spotify_feat != []:
            for items in self.non_spotify_feat:
                item_index_ = self.item_dictionary_all[items]
                similarities_sparse_c.at[item_index_, 0] = -1

        return similarities_sparse_c

    def recommend(self, train_items, prediction_length):
        similarities_sparse_c = self.get_similarity_scores(train_items)
        resi = similarities_sparse_c.nlargest(prediction_length, columns=0).index.tolist()
        res = [int(get_key(self.item_dictionary, ii)) for ii in resi]
        return res

    def score(self, train_items):
        similarities_sparse_c = self.get_similarity_scores(train_items)

        # similarities=pd.merge(similarities_sparse_c,pd.DataFrame(self.item_dictionary_df), left_index=True, right_on=0)
        similarities = pd.merge(similarities_sparse_c, pd.DataFrame(self.item_dictionary_df), left_index=True,
                                right_on='item_id')
        similarities.index = similarities['id']
        similarities = similarities.drop(['id', 'item_id'], axis=1)
        # similarities=similarities.drop(['0_x', 'key_0'], axis=1)
        return similarities.squeeze()

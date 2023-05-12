# Code from https://github.com/socialcomplab/recsys21-relistening-actr/
class MostRecent:
    def __str__(self):
        return type(self).__name__

    def recommend_next(self, user_events):
        return user_events["item"].values[-1]

    def recommend(self, user_events, topn):
        return user_events["item"].iloc[::-1].unique().tolist()[:topn]

# Code from https://github.com/socialcomplab/recsys21-relistening-actr/
class UserBasedTransitionProbability:
    def __str__(self):
        return type(self).__name__

    def recommend_next(self, user_events):
        cur_item = user_events["item"].iloc[-1]
        events_on_cur_item = user_events[(user_events["item"] == cur_item).shift().fillna(False)]
        if not events_on_cur_item.empty:
            return events_on_cur_item["item"].mode().values[-1]
        else:
            # Return no recommendation
            return -1

    def recommend(self, user_events, topn):
        cur_item = user_events["item"].iloc[-1]
        events_on_cur_item = user_events[(user_events["item"] == cur_item).shift().fillna(False)]
        return events_on_cur_item["item"].value_counts().index.tolist()[:topn]

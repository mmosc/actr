# the data splitter
# Code from https://github.com/socialcomplab/recsys21-relistening-actr/
import pandas as pd
from collections import namedtuple

TTPair = namedtuple("TTPair", ["train", "test"])


class Slider:
    def __init__(self, offset=0, step=1):
        self.offset = offset
        self.step = step

    def __call__(self, last_pos):
        for pos in range(self.offset, last_pos, self.step):
            yield pos


class TrainAll:
    def __call__(self, user_df, pos):
        return user_df.iloc[:pos + 1]


class TrainLastK:
    def __init__(self, lastk=1):
        self.lastk = lastk

    def __call__(self, user_df, pos):
        start = max(0, pos + 1 - self.lastk)
        return user_df.iloc[start:pos + 1]


class TrainTimeDelta:
    def __init__(self, timedelta=pd.Timedelta("7days"), time_col="timestamp"):
        self.timedelta = timedelta
        self.time_col = time_col

    def __call__(self, user_df, pos):
        split_time = user_df.iloc[pos][self.time_col]
        within_window = (user_df[self.time_col] <= split_time) & (user_df[self.time_col] > split_time - self.timedelta)
        return user_df[within_window]


class TestNextK:
    def __init__(self, nextk=1):
        self.nextk = nextk

    def __call__(self, user_df, pos):
        return user_df.iloc[pos + 1:pos + 1 + self.nextk]


class TestTimeDelta:
    def __init__(self, timedelta=pd.Timedelta("30min"), time_col="timestamp"):
        self.timedelta = timedelta
        self.time_col = time_col

    def __call__(self, user_df, pos):
        split_time = user_df.iloc[pos][self.time_col]
        within_window = (user_df[self.time_col] > split_time) & (user_df[self.time_col] <= split_time + self.timedelta)
        return user_df[within_window]


class TestRemainingSession:
    def __init__(self, session_col="session"):
        self.session_col = session_col

    def __call__(self, user_df, pos):
        current_session = user_df.iloc[pos][self.session_col].max()
        after_pos = user_df.iloc[pos + 1:]
        return after_pos[after_pos[self.session_col] == current_session]


class ValidSessionDuration:
    def __init__(self, min_duration=pd.Timedelta("60min"), duration_col="session_duration"):
        self.min_duration = min_duration
        self.duration_col = duration_col

    def __call__(self, user_df, pos):
        return user_df.iloc[pos][self.duration_col] >= self.min_duration


class DataSplitter:
    def __init__(self, slider, train_processor, test_processor, valid_checker=None):
        self.slider = slider
        self.train_processor = train_processor
        self.test_processor = test_processor
        self.valid_checker = valid_checker

    def __str__(self):
        return type(self.train_processor).__name__ + "_" + type(self.test_processor).__name__

    def __call__(self, user_df):
        for pos in self.slider(len(user_df)):
            if self.valid_checker and not self.valid_checker(user_df, pos):
                continue  # Not a valid session to consider

            train = self.train_processor(user_df, pos)

            assert len(train)  # At least 1 item in training data
            test = self.test_processor(user_df, pos)
            if not len(test):
                continue  # Ignore session end items
            yield TTPair(train, test)

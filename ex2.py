import abc
from typing import Tuple
import pandas as pd
import numpy as np


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.n_users = ratings['user'].nunique()
        self.n_items = ratings['item'].nunique()
        self.avg_users_dict = dict()
        self.avg_items_dict = dict()
        self.r_matrix_avg = 0
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        n_users = true_ratings['user'].nunique()
        n_items = true_ratings['item'].nunique()
        item1 = 1 / (n_users * n_items)
        curr_sum = 0
        for user_id in range(n_users):
            for item_id in range(n_items):
                true_val = true_ratings[(true_ratings['user'] == user_id) & (true_ratings['item'] == item_id)]['rating'].values[0]
                pred_val = self.r_matrix_avg + (self.avg_users_dict[user_id] - self.r_matrix_avg) + (self.avg_items_dict[item_id] - self.r_matrix_avg)
                curr_sum += (true_val - pred_val)**2

        return (item1*curr_sum)**0.5


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            self.avg_users_dict[user_idx] = sum(r_user) / len(r_user)

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            self.avg_items_dict[item_idx] = sum(r_item) / len(r_item)

        self.r_matrix_avg = sum(self.avg_users_dict.values()) / len(self.avg_users_dict.values())


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        pass


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        pass


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        pass

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        pass

import abc
from typing import Tuple
import pandas as pd
import numpy as np
import sklearn as sk


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.avg_users_dict = dict()
        self.avg_items_dict = dict()
        self.r_matrix_avg = 0
        self.r_matrix = None
        self.b_i_dict = dict()
        self.b_u_dict = dict()
        self.similarity_matrix = dict()
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
        if type(self) == NeighborhoodRecommender:
            for user in true_ratings['user'].unique():
                for other_user in self.r_matrix['user'].unique():
                    if user < other_user:
                        self.similarity_matrix[(user, other_user)] = self.user_similarity(user, other_user)
                        self.similarity_matrix[(other_user, user)] = self.similarity_matrix[(user, other_user)]



        sum_diff_2 = 0
        for idx, row in true_ratings.iterrows():
            sum_diff_2 += (row['rating'] - self.predict(row['user'], row['item'], 0))**2

        r_size = 1 / len(true_ratings['user'])
        return (r_size * sum_diff_2)**0.5


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            self.avg_users_dict[user_idx] = sum(r_user) / len(r_user)

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            self.avg_items_dict[item_idx] = sum(r_item) / len(r_item)

        self.r_matrix_avg = sum(ratings['rating']) / len(ratings['rating'])


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        b_u = self.avg_users_dict[user] - self.r_matrix_avg if user in self.avg_users_dict else 0
        b_i = self.avg_items_dict[item] - self.r_matrix_avg if item in self.avg_items_dict else 0
        predicted_rating = self.r_matrix_avg + b_u + b_i
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            self.avg_users_dict[user_idx] = sum(r_user) / len(r_user)

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            self.avg_items_dict[item_idx] = sum(r_item) / len(r_item)

        self.r_matrix_avg = sum(ratings['rating']) / len(ratings['rating'])

        for idx, row in ratings.iterrows():
            user = row['user']
            item = row['item']
            self.b_u_dict[user] = self.avg_users_dict[user] - self.r_matrix_avg
            self.b_i_dict[item] = self.avg_items_dict[item] - self.r_matrix_avg

        self.r_matrix = ratings

        """for user1 in ratings['user'].unique():
            for user2 in ratings['user'].unique():
                if user1 < user2:
                    self.similarity_matrix[(user1, user2)] = self.user_similarity(user1, user2)"""



    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        similarities_lst = list()
        for other_user in self.r_matrix['user'].unique():
            similarities_lst.append((self.similarity_matrix[(user, other_user)], other_user))

        top_3_lst = similarities_lst.sort(key=lambda x: x[0], reverse=True)[:3]
        numerator = 0
        denominator = 0
        for similarity, user_idx in top_3_lst:
            pred_other_user = self.r_matrix_avg + self.b_u_dict[user_idx] + self.b_i_dict[item]
            numerator += pred_other_user * similarity
            denominator += abs(similarity)

        corr_val = numerator / denominator

        pred_val = self.r_matrix_avg + self.b_u_dict[user] + self.b_i_dict[item] + corr_val
        return pred_val

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        data_user1 = self.r_matrix[self.r_matrix['user'] == user1][['rating', 'item']]
        r_user1_centered = [row['rating'] - (self.r_matrix_avg + self.b_u_dict[user1] + self.b_i_dict[row['item']]) for idx, row in data_user1.iterrows()]
        i_user1 = list(data_user1['item'])

        data_user2 = self.r_matrix[self.r_matrix['user'] == user2][['rating', 'item']]
        r_user2_centered = [row['rating'] - (self.r_matrix_avg + self.b_u_dict[user2] + self.b_i_dict[row['item']]) for idx, row in data_user2.iterrows()]
        i_user2 = list(data_user2['item'])

        numerator = 0
        r_1_norm = 0
        r_2_norm = 0
        for idx in range(len(i_user1)):
            curr_item = i_user1[idx]
            if curr_item in i_user2:
                curr_item_user_2 = i_user2.index(curr_item)
                numerator += r_user1_centered[idx]*r_user2_centered[curr_item_user_2]
                r_1_norm += r_user1_centered[idx]**2
                r_2_norm += r_user2_centered[curr_item_user_2]**2

        denominator = (r_1_norm * r_2_norm)**0.5

        return numerator / denominator if denominator > 0 else 0






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

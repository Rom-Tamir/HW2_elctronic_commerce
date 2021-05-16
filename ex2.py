import abc
import datetime
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
        self.users_to_calc = None
        self.rating_centered = None
        self.sol = None
        self.n_users = 0
        self.corr_matrix = None
        self.matrix_to_corr = None
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
        sum_diff_2 = 0
        for idx, row in true_ratings.iterrows():
            sum_diff_2 += (row['rating'] - self.predict(row['user'], row['item'], row['timestamp']))**2

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
        self.r_matrix_avg = sum(ratings['rating']) / len(ratings['rating'])
        unique_users = ratings['user'].unique()
        for user_idx in unique_users:
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            avg_user = sum(r_user) / len(r_user)
            self.b_u_dict[user_idx] = avg_user - self.r_matrix_avg

        unique_items = ratings['item'].unique()
        for item_idx in unique_items:
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            avg_item = sum(r_item) / len(r_item)
            self.b_i_dict[item_idx] = avg_item - self.r_matrix_avg

        max_user = int(max(unique_users) + 1)
        max_item = int(max(unique_items) + 1)
        self.matrix_to_corr = np.zeros((max_item, max_user))
        # matrix_to_corr[:] = np.nan
        for idx, row in ratings.iterrows():
            self.matrix_to_corr[int(row['item'])][int(row['user'])] = row['rating'] - self.r_matrix_avg

        df_to_corr = pd.DataFrame(self.matrix_to_corr)
        self.corr_matrix = df_to_corr.corr()
        # self.corr_matrix = self.corr_matrix.fillna(0)
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
        potential_users = self.r_matrix[(self.r_matrix.user != user) & (self.r_matrix.item == item)]['user']
        potential_users_similarities = []
        for other_user in potential_users:
            potential_users_similarities.append((self.corr_matrix[user][other_user], other_user))
            # print(f"matrix val: {self.corr_matrix[user][other_user]}")
            # print(self.user_similarity(user, other_user))
            # print("new row!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        potential_users_similarities.sort(key=lambda x: x[0], reverse=True)
        top_3_lst = potential_users_similarities[:3]
        numerator = 0
        denominator = 0
        for similarity, user_idx in top_3_lst:
            r_wave_other_user = self.matrix_to_corr[int(item)][int(user_idx)]
            numerator += r_wave_other_user * similarity
            denominator += abs(similarity)

        corr_val = numerator / denominator if denominator != 0 else 0

        predicted_rating = self.r_matrix_avg + self.b_u_dict[user] + self.b_i_dict[item] + corr_val
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5

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
        self.n_users = int(max(ratings['user']) + 1)
        n_items = int(max(ratings['item']) + 1)
        n_records = len(ratings['item'])
        self.r_matrix = np.zeros((n_records, self.n_users + n_items + 3))
        for idx, row in ratings.iterrows():
            curr_user = int(row['user'])
            curr_item = int(row['item'])
            curr_timestamp = datetime.datetime.fromtimestamp(row['timestamp'])
            self.r_matrix[idx][curr_user] = 1
            self.r_matrix[idx][self.n_users + curr_item] = 1
            self.r_matrix[idx][-1] = 1 if 3 < curr_timestamp.weekday() < 6 else 0
            self.r_matrix[idx][-3] = 1 if 6 <= curr_timestamp.hour < 18 else 0
            self.r_matrix[idx][-2] = 1 if self.r_matrix[idx, -3] == 0 else 0

        self.r_matrix_avg = sum(ratings['rating']) / len(ratings['rating'])
        self.rating_centered = np.array(ratings['rating']) - self.r_matrix_avg

    def predict(self, user: int, item: int, timestamp: int) -> float:
        dt = datetime.datetime.fromtimestamp(timestamp)
        b_w_indicator = 1 if 3 < dt.weekday() < 6 else 0
        b_d_indicator = 1 if 6 <= dt.hour < 18 else 0
        b_n_indicator = 1 if b_d_indicator == 0 else 0
        pred = self.r_matrix_avg + self.sol[int(user)] + self.sol[int(self.n_users + item)] + b_w_indicator*self.sol[-1] + b_d_indicator*self.sol[-3] + b_n_indicator*self.sol[-2]
        return pred if 0.5 <= pred <= 5 else 0.5 if pred < 0.5 else 5

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.sol = np.linalg.lstsq(self.r_matrix, np.array(self.rating_centered))[0]

        return self.r_matrix, self.sol, self.rating_centered


class CompetitionRecommender(Recommender):
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


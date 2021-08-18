import abc
import datetime
from typing import Tuple
import pandas as pd
import numpy as np


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.r_matrix_avg = sum(ratings['rating']) / len(ratings['rating'])
        self.r_matrix = None
        self.b_i_dict = dict()
        self.b_u_dict = dict()
        self.rating_centered = None
        self.sol = None
        self.n_users = 0
        self.similarity_dict = dict()
        self.r_wave_matrix = None
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

    def omer_rmse(self, true_ratings) -> float:
        user_col = true_ratings["user"].tolist()
        item_col = true_ratings["item"].tolist()
        rating_col = true_ratings["rating"].tolist()
        timestamp_col = true_ratings["timestamp"].tolist()

        sum_rmse = 0
        for idx in range(len(user_col)):
            sum_rmse += (rating_col[idx] - self.predict(int(user_col[idx]), int(item_col[idx]), int(timestamp_col[idx]))) ** 2

        if len(user_col) != 0:
            rmse = np.sqrt((1 / len(user_col)) * sum_rmse)
        else:
            rmse = 0

        return rmse


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            avg_user = sum(r_user) / len(r_user)
            self.b_u_dict[user_idx] = avg_user - self.r_matrix_avg

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            avg_item = sum(r_item) / len(r_item)
            self.b_i_dict[item_idx] = avg_item - self.r_matrix_avg


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        b_u = self.b_u_dict[user] if user in self.b_u_dict else 0
        b_i = self.b_i_dict[item] if item in self.b_i_dict else 0
        predicted_rating = self.r_matrix_avg + b_u + b_i
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        unique_users = ratings['user'].unique()
        max_user = int(max(unique_users)) + 1
        for user_idx in unique_users:
            r_user = ratings[ratings['user'] == user_idx]['rating']
            avg_user = sum(r_user) / len(r_user)
            self.b_u_dict[user_idx] = avg_user - self.r_matrix_avg

        unique_items = ratings['item'].unique()
        max_item = int(max(unique_items)) + 1
        for item_idx in unique_items:
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            avg_item = sum(r_item) / len(r_item)
            self.b_i_dict[item_idx] = avg_item - self.r_matrix_avg

        # similarity part

        self.r_wave_matrix = np.zeros((max_user, max_item))
        for idx, row in ratings.iterrows():
            self.r_wave_matrix[int(row['user'])][int(row['item'])] = row['rating'] - self.r_matrix_avg

        for user_idx in unique_users:
            self.similarity_dict[(user_idx, user_idx)] = 1
            for other_user_idx in unique_users:
                if user_idx > other_user_idx:
                    user1_items = np.where(self.r_wave_matrix[int(user_idx)] != 0)[0]
                    user2_items = np.where(self.r_wave_matrix[int(other_user_idx)] != 0)[0]
                    mutual_items = np.intersect1d(user1_items, user2_items, assume_unique=True)
                    if len(mutual_items) == 0:
                        self.similarity_dict[(user_idx, other_user_idx)] = 0
                        self.similarity_dict[(other_user_idx, user_idx)] = 0
                        continue
                    user_1_rating_consider = self.r_wave_matrix[int(user_idx)][mutual_items]
                    user_2_rating_consider = self.r_wave_matrix[int(other_user_idx)][mutual_items]
                    self.similarity_dict[(user_idx, other_user_idx)] = np.inner(user_1_rating_consider, user_2_rating_consider) / (np.linalg.norm(user_1_rating_consider) * np.linalg.norm(user_2_rating_consider))
                    self.similarity_dict[(other_user_idx, user_idx)] = self.similarity_dict[(user_idx, other_user_idx)]

        self.r_matrix = ratings


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        potential_users = self.r_matrix[(self.r_matrix.user != user) & (self.r_matrix.item == item)]['user'].unique()
        potential_users_similarities = []
        for other_user in potential_users:
            potential_users_similarities.append((self.similarity_dict[(user, other_user)], other_user))
        potential_users_similarities.sort(key=lambda x: abs(x[0]), reverse=True)
        top_3_lst = potential_users_similarities[:3]
        numerator = 0
        denominator = 0
        for similarity, user_idx in top_3_lst:
            r_wave_other_user = self.r_wave_matrix[int(user_idx)][int(item)]
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
        return self.similarity_dict[(user1, user2)] if (user1, user2) in self.similarity_dict else 0


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
        self.sol = np.linalg.lstsq(self.r_matrix, np.array(self.rating_centered), rcond=None)[0]

        return self.r_matrix, self.sol, self.rating_centered


class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            avg_user = sum(r_user) / len(r_user)
            self.b_u_dict[user_idx] = avg_user - self.r_matrix_avg

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            avg_item = sum(r_item) / len(r_item)
            self.b_i_dict[item_idx] = avg_item - self.r_matrix_avg

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        b_u = self.b_u_dict[user] if user in self.b_u_dict else 0
        b_i = self.b_i_dict[item] if item in self.b_i_dict else 0
        predicted_rating = self.r_matrix_avg + b_u + b_i
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5



class MFRecommender(Recommender):
    def __init__(self, R, K=10, alpha=0.01, beta=0.01, iterations=10):
        """
               Perform matrix factorization to predict empty
               entries in a matrix.
               Arguments
               - R (ndarray)   : user-item rating matrix
               - K (int)       : number of latent dimensions
               - alpha (float) : learning rate
               - beta (float)  : regularization parameter
               """
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.count = 0
        super().__init__(R)


    def initialize_predictor(self, ratings):
        n_users = int(max(ratings['user']) + 1)
        n_items = int(max(ratings['item']) + 1)
        self.b_u = np.zeros(n_users)
        self.b_m = np.zeros(n_items)

        for user_idx in ratings['user'].unique():
            r_user = list(ratings[ratings['user'] == user_idx]['rating'])
            avg_user = sum(r_user) / len(r_user)
            self.b_u[int(user_idx)] = avg_user - self.r_matrix_avg

        for item_idx in ratings['item'].unique():
            r_item = list(ratings[ratings['item'] == item_idx]['rating'])
            avg_item = sum(r_item) / len(r_item)
            self.b_m[int(item_idx)] = avg_item - self.r_matrix_avg



        self.r_matrix = np.zeros((n_users, n_items))
        for idx, row in ratings.iterrows():
            curr_user = int(row['user'])
            curr_item = int(row['item'])
            self.r_matrix[curr_user][curr_item] = float(row['rating'])

        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(n_users, self.K))
        self.Q = np.random.normal(scale=1. / self.K, size=(n_items, self.K))

        self.samples = [
            (i, j, self.r_matrix[i, j])
            for i in range(n_users)
            for j in range(n_items)
            if self.r_matrix[i, j] > 0
        ]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.omer_rmse(ratings)
            training_process.append((i, rmse, mf_params(self.Q, self.P, self.b_u, self.b_m)))
            # if (i+1) % 100 == 0:
            #    print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        print(f"SGD num: {self.count}")
        self.count += 1
        for user, item, rating in self.samples:
            # Computer prediction and error
            prediction = self.predict(user, item, 0)
            error = rating - prediction

            # Update biases
            self.b_u[user] += self.alpha * (error - self.beta * self.b_u[user])
            self.b_m[item] += self.alpha * (error - self.beta * self.b_m[item])

            # Update user and item latent feature matrices
            self.P[user, :] += self.alpha * (error * self.Q[item, :] - self.beta * self.P[user, :])
            self.Q[item, :] += self.alpha * (error * self.P[user, :] - self.beta * self.Q[item, :])

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        predicted_rating = self.r_matrix_avg + self.b_u[int(user)] + self.b_m[int(item)] + self.P[int(user), :].dot(self.Q[int(item), :].T)
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5

    def mf_rmse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.r_matrix.nonzero()
        predicted_matrix = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += (self.r_matrix[x, y] - predicted_matrix[x, y])**2
        return error**0.5

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.r_matrix_avg + self.b_u[:, np.newaxis] + self.b_m[np.newaxis:, ] + self.P.dot(self.Q.T)


class mf_params:
    def __init__(self, Q, P, b_u, b_m):
        self.Q = np.copy(Q)
        self.P = np.copy(P)
        self.b_u = np.copy(b_u)
        self.b_m = np.copy(b_m)





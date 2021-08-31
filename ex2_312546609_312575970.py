import abc
import datetime
from typing import Tuple
import pandas as pd
import numpy as np
from itertools import product
import ast


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame, num_neighbors=0):
        self.num_neighbors = num_neighbors
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
        top_k_lst = potential_users_similarities[:self.num_neighbors]
        numerator = 0
        denominator = 0
        for similarity, user_idx in top_k_lst:
            r_wave_other_user = self.r_wave_matrix[int(user_idx)][int(item)]
            numerator += r_wave_other_user * similarity
            denominator += abs(similarity)

        corr_val = numerator / denominator if denominator != 0 else 0

        self.b_u_dict[user] = self.b_u_dict[user] if user in self.b_u_dict else 0
        self.b_i_dict[item] = self.b_i_dict[item] if item in self.b_i_dict else 0

        predicted_rating = self.r_matrix_avg + self.b_u_dict[user] + self.b_i_dict[item] + corr_val
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        return self.similarity_dict[(user1, user2)] if (user1, user2) in self.similarity_dict else 0

    @staticmethod
    def cross_validation_error(df, num_of_neighbors, folds):
        # Create folds
        X_folds = np.array_split(df, folds)
        train_results, val_results = [], []

        for i in range(folds):
            # Create train, validation for current fold
            X_val_fold = X_folds[i]
            X_train_fold = pd.concat([other_df for other_df in X_folds if not other_df.equals(X_val_fold)])

            # Fit the model on the current fold training set
            model = NeighborhoodRecommender(X_train_fold, num_of_neighbors)

            # Evaluate on the fold validation set
            val_results.append(model.rmse(X_val_fold))

        return np.array(val_results).mean()


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
    def __init__(self, R, K=100, alpha=0.01, beta=0.01, iterations=50):
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
        self.b_company = dict()

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
            rmse = self.rmse(ratings)
            training_process.append((i, rmse, mf_params(self.Q, self.P, self.b_u, self.b_m)))
            # if (i+1) % 100 == 0:
            #    print("Iteration: %d ; error = %.4f" % (i+1, mse))

        return training_process

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        #print(f"SGD num: {self.count}")
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

    @staticmethod
    def cross_validation_error(df, combination_of_params, folds):
        # Create folds
        X_folds = np.array_split(df, folds)
        val_results = []

        for i in range(folds):
            # Create train, validation for current fold
            X_val_fold = X_folds[i]
            X_train_fold = pd.concat([other_df for other_df in X_folds if not other_df.equals(X_val_fold)])

            # Fit the model on the current fold training set
            model = MFRecommender(X_train_fold, combination_of_params[0], combination_of_params[1], combination_of_params[2], combination_of_params[3])

            # Evaluate on the fold validation set
            val_results.append(model.rmse(X_val_fold))

        return np.array(val_results).mean()

    @staticmethod
    def hyperparameters_tuning(df):
        possible_k = [10, 100, 250]
        possible_alpha = [0.01, 0.1]
        possible_beta = [0.01, 0.1]
        possible_iterations = [10, 50, 100]

        list_of_lists = [possible_k, possible_alpha, possible_beta, possible_iterations]
        all_combinations = list(product(*list_of_lists))

        # shuffle the df
        df = df.sample(frac=1)
        mf_rmse_dict = dict()
        for combination_of_params in all_combinations:
            mf_rmse_dict[combination_of_params] = MFRecommender.cross_validation_error(df, combination_of_params, 4)

        keys_list = list(mf_rmse_dict.keys())
        val_list = list(mf_rmse_dict.values())
        optimal_mf_rmse = min(val_list)
        optimal_params = keys_list[val_list.index(optimal_mf_rmse)]
        print(f'The optimal params for the MF Recommender according to cross validation is: {optimal_params} and the optimal RMSE is: {optimal_mf_rmse}')

        return optimal_params


class HybridMFRecommender(Recommender):
    def __init__(self, R, K=100, alpha=0.01, beta=0.01, iterations=10, movies_metadata=None):
        """
               Perform matrix factorization to predict empty
               entries in a matrix.
               Arguments
               - R (ndarray)   : user-item rating matrix
               - K (int)       : number of latent dimensions
               - alpha (float) : learning rate
               - beta (float)  : regularization parameter
               """
        # region hyper-parameters
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        # endregion

        # region movies metadata biases
        self.movies_metadata = movies_metadata
        self.b_company = None
        self.b_genre = None
        self.b_country = None
        self.b_spoken = None
        self.b_runtime = None
        self.b_revenue = None
        self.b_budget = None
        self.b_popularity = None
        self.b_vote_count = None
        self.b_vote_avg = None
        self.b_years = None
        # endregion

        # region other fields
        self.count = 0
        self.ratings = R
        self.r_matrix_avg = sum(R['rating']) / len(R['rating'])
        # endregion

        super().__init__(R)

    def initialize_predictor(self, ratings):
        n_users = int(max(ratings['user']) + 1)
        n_items = int(max(ratings['item']) + 1)

        # region build b_u, b_m, r_matrix and metadata biases
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

        self.b_company = np.zeros(n_items)
        self.b_genre = np.zeros(n_items)
        self.b_country = np.zeros(n_items)
        self.b_spoken = np.zeros(n_items)
        self.b_runtime = np.zeros(n_items)
        self.b_revenue = np.zeros(n_items)
        self.b_budget = np.zeros(n_items)
        self.b_popularity = np.zeros(n_items)
        self.b_vote_count = np.zeros(n_items)
        self.b_vote_avg = np.zeros(n_items)
        self.b_years = np.zeros(n_items)

        self.build_biases()
        # endregion

        # region Init P, Q, samples and train by SGD
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
            rmse = self.rmse(ratings)
            training_process.append((i, rmse, mf_params(self.Q, self.P, self.b_u, self.b_m)))

        # endregion

        return training_process

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for user, item, rating in self.samples:
            prediction = self.predict(user, item, 0)
            error = rating - prediction
            # region update biases
            self.b_u[user] += self.alpha * (error - self.beta * self.b_u[user])
            self.b_m[item] += self.alpha * (error - self.beta * self.b_m[item])
            self.b_company[item] += self.alpha * (error - self.beta * self.b_company[item])
            self.b_genre[item] += self.alpha * (error - self.beta * self.b_genre[item])
            self.b_country[item] += self.alpha * (error - self.beta * self.b_country[item])
            self.b_spoken[item] += self.alpha * (error - self.beta * self.b_spoken[item])
            self.b_runtime[item] += self.alpha * (error - self.beta * self.b_runtime[item])
            self.b_revenue[item] += self.alpha * (error - self.beta * self.b_revenue[item])
            self.b_budget[item] += self.alpha * (error - self.beta * self.b_budget[item])
            self.b_popularity[item] += self.alpha * (error - self.beta * self.b_popularity[item])
            self.b_vote_count[item] += self.alpha * (error - self.beta * self.b_vote_count[item])
            self.b_vote_avg[item] += self.alpha * (error - self.beta * self.b_vote_avg[item])
            self.b_years[item] += self.alpha * (error - self.beta * self.b_years[item])

            # endregion
            # region update P,Q
            self.P[user, :] += self.alpha * (error * self.Q[item, :] - self.beta * self.P[user, :])
            self.Q[item, :] += self.alpha * (error * self.P[user, :] - self.beta * self.Q[item, :])
            # endregion

    def predict(self, user: int, item: int, original_item=0) -> float:
        """
        :param original_item:
        :param user: User identifier
        :param item: Item identifier
        :return: Predicted rating of the user for the item
        """
        predicted_rating = self.r_matrix_avg + self.b_u[int(user)] + self.b_m[int(item)] + self.b_company[item] + self.b_genre[item] + self.b_spoken[item] + self.b_runtime[item] + self.b_revenue[item] + self.b_budget[item] + self.b_popularity[item] + self.b_vote_count[item] + self.b_vote_avg[item] + self.b_years[item] + self.P[int(user), :].dot(self.Q[int(item), :].T)
        return predicted_rating if 0.5 <= predicted_rating <= 5 else 0.5 if predicted_rating < 0.5 else 5

    def mf_rmse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.r_matrix.nonzero()
        predicted_matrix = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            error += (self.r_matrix[x, y] - predicted_matrix[x, y]) ** 2
        return error ** 0.5

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        return self.r_matrix_avg + self.b_u[:, np.newaxis] + self.b_m[np.newaxis:, ] + self.P.dot(self.Q.T)

    def build_biases(self):
        self.build_companies_biases()
        self.build_genres_biases()
        self.build_countries_biases()
        self.build_spoken_languages_biases()
        self.build_runtimes_biases()
        self.build_revenues_biases()
        self.build_budgets_biases()
        self.build_popularity_biases()
        self.build_vote_counts_biases()
        self.build_vote_avg_biases()
        self.build_years_biases()
        #return

    def build_companies_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build company_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        company_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            company_str = row["production_companies"]
            company_dict = ast.literal_eval(company_str)
            for item in company_dict:
                company_rating_dict[item["name"]] = company_rating_dict.get(item["name"], 0) + movie_id_avg_rating_dict[
                    movie_id]
                counter_dict[item["name"]] = counter_dict.get(item["name"], 0) + 1
        # endregion

        # region build company_bias_dict
        keys_list = list(company_rating_dict.keys())
        company_bias_dict = {}
        for company in keys_list:
            company_bias_dict[company] = (company_rating_dict[company] / counter_dict[company]) - self.r_matrix_avg
        # endregion

        # region build b_company (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_company_str = movie_metadata["production_companies"].values[0]
            movie_company_dict = ast.literal_eval(movie_company_str)
            for company in movie_company_dict:
                self.b_company[counter] += company_bias_dict[company["name"]] / len(movie_company_dict)

            counter += 1

        # endregion

    def build_genres_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build genres_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        genres_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            genre_str = row["genres"]
            genre_dict = ast.literal_eval(genre_str)
            for genre in genre_dict:
                genres_rating_dict[genre["name"]] = genres_rating_dict.get(genre["name"], 0) + movie_id_avg_rating_dict[
                    movie_id]
                counter_dict[genre["name"]] = counter_dict.get(genre["name"], 0) + 1
        # endregion

        # region build genres_bias_dict
        keys_list = list(genres_rating_dict.keys())
        genres_bias_dict = {}
        for genre in keys_list:
            genres_bias_dict[genre] = (genres_rating_dict[genre] / counter_dict[genre]) - self.r_matrix_avg
        # endregion

        # region build b_genre (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_genre_str = movie_metadata["genres"].values[0]
            movie_genre_dict = ast.literal_eval(movie_genre_str)
            for genre in movie_genre_dict:
                self.b_genre[counter] += genres_bias_dict[genre["name"]] / len(movie_genre_dict)

            counter += 1

        # endregion

    def build_countries_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build countries_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        countries_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            country_str = row["production_countries"]
            country_dict = ast.literal_eval(country_str)
            for country in country_dict:
                countries_rating_dict[country["name"]] = countries_rating_dict.get(country["name"], 0) + movie_id_avg_rating_dict[
                    movie_id]
                counter_dict[country["name"]] = counter_dict.get(country["name"], 0) + 1
        # endregion

        # region build countries_bias_dict
        keys_list = list(countries_rating_dict.keys())
        countries_bias_dict = {}
        for country in keys_list:
            countries_bias_dict[country] = (countries_rating_dict[country] / counter_dict[country]) - self.r_matrix_avg
        # endregion

        # region build b_country (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_country_str = movie_metadata["production_countries"].values[0]
            movie_country_dict = ast.literal_eval(movie_country_str)
            for country in movie_country_dict:
                self.b_country[counter] += countries_bias_dict[country["name"]] / len(movie_country_dict)

            counter += 1

        # endregion

    def build_spoken_languages_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build languages_rating_dict & languages_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        languages_dict = {}
        languages_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            language_str = row["spoken_languages"]
            language_dict = ast.literal_eval(language_str)
            for language in language_dict:
                languages_rating_dict[language["name"]] = languages_rating_dict.get(language["name"], 0) + movie_id_avg_rating_dict[
                    movie_id]
                languages_dict[language["name"]] = languages_dict.get(language["name"], 0) + 1
        # endregion

        # region build languages_bias_dict
        keys_list = list(languages_rating_dict.keys())
        languages_bias_dict = {}
        for language in keys_list:
            languages_bias_dict[language] = (languages_rating_dict[language] / languages_dict[language]) - self.r_matrix_avg
        # endregion

        # region build b_spoken (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_language_str = movie_metadata["spoken_languages"].values[0]
            movie_language_dict = ast.literal_eval(movie_language_str)
            for language in movie_language_dict:
                self.b_spoken[counter] += languages_bias_dict[language["name"]] / len(movie_language_dict)

            counter += 1

        # endregion

    def build_runtimes_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build runtimes_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        runtimes_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            runtime = row["runtime"]
            if runtime == 0:
                continue
            elif runtime > 135:
                runtimes_rating_dict["long movie"] = runtimes_rating_dict.get("long movie", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["long movie"] = counter_dict.get("long movie", 0) + 1
            elif runtime > 80:
                runtimes_rating_dict["medium movie length"] = runtimes_rating_dict.get("medium movie length", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["medium movie length"] = counter_dict.get("medium movie length", 0) + 1
            else:
                runtimes_rating_dict["short movie"] = runtimes_rating_dict.get("short movie", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["short movie"] = counter_dict.get("short movie", 0) + 1
        # endregion

        # region build runtimes_biases_dict
        keys_list = list(runtimes_rating_dict.keys())
        runtimes_biases_dict = {}
        for runtime in keys_list:
            runtimes_biases_dict[runtime] = (runtimes_rating_dict[runtime] / counter_dict[runtime]) - self.r_matrix_avg
        # endregion

        # region build b_runtime (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_runtime = movie_metadata["runtime"].values[0]
            self.b_runtime[counter] = runtimes_biases_dict['long movie'] if movie_runtime > 135 else runtimes_biases_dict['medium movie length'] if movie_runtime > 80 else runtimes_biases_dict['short movie']
            counter += 1

        # endregion

    def build_revenues_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build revenues_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        revenues_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            revenue = row["revenue"]
            if revenue == 0:
                continue
            elif revenue > 180000000:
                revenues_rating_dict["high revenue"] = revenues_rating_dict.get("high revenue", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["high revenue"] = counter_dict.get("high revenue", 0) + 1
            elif revenue > 4000000:
                revenues_rating_dict["medium revenue"] = revenues_rating_dict.get("medium revenue", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["medium revenue"] = counter_dict.get("medium revenue", 0) + 1
            else:
                revenues_rating_dict["low revenue"] = revenues_rating_dict.get("low revenue", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["low revenue"] = counter_dict.get("low revenue", 0) + 1
        # endregion

        # region build revenues_biases_dict
        keys_list = list(revenues_rating_dict.keys())
        revenues_biases_dict = {}
        for revenue in keys_list:
            revenues_biases_dict[revenue] = (revenues_rating_dict[revenue] / counter_dict[revenue]) - self.r_matrix_avg
        # endregion

        # region build b_revenue (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_revenue = movie_metadata["revenue"].values[0]
            self.b_revenue[counter] = revenues_biases_dict['high revenue'] if movie_revenue > 180000000 else revenues_biases_dict['medium revenue'] if movie_revenue > 4000000 else revenues_biases_dict['low revenue']
            counter += 1

        # endregion

    def build_budgets_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build budgets_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        budgets_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            budget = row["budget"]
            if budget == 0:
                continue
            elif budget > 54000000:
                budgets_rating_dict["high budget"] = budgets_rating_dict.get("high budget", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["high budget"] = counter_dict.get("high budget", 0) + 1
            elif budget > 2000000:
                budgets_rating_dict["medium budget"] = budgets_rating_dict.get("medium budget", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["medium budget"] = counter_dict.get("medium budget", 0) + 1
            else:
                budgets_rating_dict["low budget"] = budgets_rating_dict.get("low budget", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["low budget"] = counter_dict.get("low budget", 0) + 1
        # endregion

        # region build budgets_biases_dict
        keys_list = list(budgets_rating_dict.keys())
        budgets_biases_dict = {}
        for budget in keys_list:
            budgets_biases_dict[budget] = (budgets_rating_dict[budget] / counter_dict[budget]) - self.r_matrix_avg
        # endregion

        # region build b_budget (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_budget = movie_metadata["budget"].values[0]
            self.b_budget[counter] = budgets_biases_dict['high budget'] if movie_budget > 54000000 else budgets_biases_dict['medium budget'] if movie_budget > 2000000 else budgets_biases_dict['low budget']
            counter += 1

        # endregion

    def build_popularity_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build popularity_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        popularity_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            popularity = row["popularity"]
            if popularity == 0:
                continue
            elif popularity > 15:
                popularity_rating_dict["very high popularity"] = popularity_rating_dict.get("very high popularity", 0) + \
                                                             movie_id_avg_rating_dict[movie_id]
                counter_dict["very high popularity"] = counter_dict.get("very high popularity", 0) + 1
            elif popularity > 10:
                popularity_rating_dict["high popularity"] = popularity_rating_dict.get("high popularity", 0) + \
                                                            movie_id_avg_rating_dict[movie_id]
                counter_dict["high popularity"] = counter_dict.get("high popularity", 0) + 1
            elif popularity > 5:
                popularity_rating_dict["medium popularity"] = popularity_rating_dict.get("medium popularity", 0) + \
                                                              movie_id_avg_rating_dict[movie_id]
                counter_dict["medium popularity"] = counter_dict.get("medium popularity", 0) + 1
            elif popularity > 1:
                popularity_rating_dict["low popularity"] = popularity_rating_dict.get("low popularity", 0) + \
                                                           movie_id_avg_rating_dict[movie_id]
                counter_dict["low popularity"] = counter_dict.get("low popularity", 0) + 1
            else:
                popularity_rating_dict["very low popularity"] = popularity_rating_dict.get("very low popularity", 0) + \
                                                                movie_id_avg_rating_dict[movie_id]
                counter_dict["very low popularity"] = counter_dict.get("very low popularity", 0) + 1
        # endregion

        # region build popularity_biases_dict
        keys_list = list(popularity_rating_dict.keys())
        popularity_biases_dict = {}
        for popularity in keys_list:
            popularity_biases_dict[popularity] = (popularity_rating_dict[popularity] / counter_dict[popularity]) - self.r_matrix_avg
        # endregion

        # region build b_popularity (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_popularity = movie_metadata["popularity"].values[0]
            self.b_popularity[counter] = popularity_biases_dict['very high popularity'] if movie_popularity > 15 else popularity_biases_dict['high popularity'] if movie_popularity > 10 else popularity_biases_dict['medium popularity'] if movie_popularity > 5 else popularity_biases_dict['low popularity'] if movie_popularity > 1 else popularity_biases_dict['very low popularity']
            counter += 1

        # endregion

    def build_vote_counts_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build votes_count_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        votes_count_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            vote_count = row["vote_count"]
            if vote_count == 0:
                continue
            elif vote_count > 550:
                votes_count_rating_dict["high vote_count"] = votes_count_rating_dict.get("high vote_count", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["high vote_count"] = counter_dict.get("high vote_count", 0) + 1
            elif vote_count > 5:
                votes_count_rating_dict["medium vote_count"] = votes_count_rating_dict.get("medium vote_count", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["medium vote_count"] = counter_dict.get("medium vote_count", 0) + 1
            else:
                votes_count_rating_dict["low vote_count"] = votes_count_rating_dict.get("low vote_count", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["low vote_count"] = counter_dict.get("low vote_count", 0) + 1
        # endregion

        # region build votes_count_biases_dict
        keys_list = list(votes_count_rating_dict.keys())
        votes_count_biases_dict = {}
        for vote_count in keys_list:
            votes_count_biases_dict[vote_count] = (votes_count_rating_dict[vote_count] / counter_dict[vote_count]) - self.r_matrix_avg
        # endregion

        # region build b_vote_count (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_vote_count = movie_metadata["vote_count"].values[0]
            self.b_vote_count[counter] = votes_count_biases_dict['high vote_count'] if movie_vote_count > 550 else votes_count_biases_dict['medium vote_count'] if movie_vote_count > 5 else votes_count_biases_dict['low vote_count']
            counter += 1

        # endregion

    def build_vote_avg_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build votes_avg_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        votes_avg_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            vote_average = row["vote_average"]
            if vote_average == 0:
                continue
            elif vote_average > 7.5:
                votes_avg_rating_dict["high vote_average"] = votes_avg_rating_dict.get("high vote_average", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["high vote_average"] = counter_dict.get("high vote_average", 0) + 1
            elif vote_average > 5:
                votes_avg_rating_dict["medium vote_average"] = votes_avg_rating_dict.get("medium vote_average", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["medium vote_average"] = counter_dict.get("medium vote_average", 0) + 1
            else:
                votes_avg_rating_dict["low vote_average"] = votes_avg_rating_dict.get("low vote_average", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["low vote_average"] = counter_dict.get("low vote_average", 0) + 1
        # endregion

        # region build votes_avg_biases_dict
        keys_list = list(votes_avg_rating_dict.keys())
        votes_avg_biases_dict = {}
        for vote_average in keys_list:
            votes_avg_biases_dict[vote_average] = (votes_avg_rating_dict[vote_average] / counter_dict[vote_average]) - self.r_matrix_avg
        # endregion

        # region build b_vote_avg (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_vote_avg = movie_metadata["vote_average"].values[0]
            self.b_vote_avg[counter] = votes_avg_biases_dict['high vote_average'] if movie_vote_avg > 7.5 else votes_avg_biases_dict['medium vote_average'] if movie_vote_avg > 5 else votes_avg_biases_dict['low vote_average']
            counter += 1

        # endregion

    def build_years_biases(self):

        # region calculate each movie avg
        unique_original_movie_ids = self.ratings['original_movie_id'].unique()
        movie_id_avg_rating_dict = {}
        for movie_id in unique_original_movie_ids:
            movie_id_avg_rating_dict[movie_id] = self.ratings[(self.ratings["original_movie_id"] == movie_id)][
                "rating"].mean()

        # endregion

        # region build years_rating_dict & counter_dict
        self.movies_metadata = self.movies_metadata[self.movies_metadata.id.isin(unique_original_movie_ids)]
        counter_dict = {}
        years_rating_dict = {}
        for index, row in self.movies_metadata.iterrows():
            movie_id = row["id"]
            release_date = row["release_date"]

            if type(release_date) is str and len(release_date) > 3 and release_date[-4:].isdigit() and 0 < int(release_date[-4:]) < 2022:
                release_date = int(release_date[-4:])
            else:
                continue

            if release_date >= 2010:
                years_rating_dict["2010's"] = years_rating_dict.get("2010's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["2010's"] = counter_dict.get("2010's", 0) + 1
            elif release_date >= 2000:
                years_rating_dict["2000's"] = years_rating_dict.get("2000's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["2000's"] = counter_dict.get("2000's", 0) + 1
            elif release_date >= 1990:
                years_rating_dict["1990's"] = years_rating_dict.get("1990's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1990's"] = counter_dict.get("1990's", 0) + 1
            elif release_date >= 1980:
                years_rating_dict["1980's"] = years_rating_dict.get("1980's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1980's"] = counter_dict.get("1980's", 0) + 1
            elif release_date >= 1970:
                years_rating_dict["1970's"] = years_rating_dict.get("1970's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1970's"] = counter_dict.get("1970's", 0) + 1
            elif release_date >= 1960:
                years_rating_dict["1960's"] = years_rating_dict.get("1960's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1960's"] = counter_dict.get("1960's", 0) + 1
            elif release_date >= 1950:
                years_rating_dict["1950's"] = years_rating_dict.get("1950's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1950's"] = counter_dict.get("1950's", 0) + 1
            elif release_date >= 1940:
                years_rating_dict["1940's"] = years_rating_dict.get("1940's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1940's"] = counter_dict.get("1940's", 0) + 1
            elif release_date >= 1930:
                years_rating_dict["1930's"] = years_rating_dict.get("1930's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1930's"] = counter_dict.get("1930's", 0) + 1
            elif release_date >= 1920:
                years_rating_dict["1920's"] = years_rating_dict.get("1920's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1920's"] = counter_dict.get("1920's", 0) + 1
            elif release_date >= 1910:
                years_rating_dict["1910's"] = years_rating_dict.get("1910's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1910's"] = counter_dict.get("1910's", 0) + 1
            else:
                years_rating_dict["1900's"] = years_rating_dict.get("1900's", 0) + movie_id_avg_rating_dict[movie_id]
                counter_dict["1900's"] = counter_dict.get("1900's", 0) + 1
        # endregion

        # region build years_biases_dict
        keys_list = list(years_rating_dict.keys())
        years_biases_dict = {}
        for release_date in keys_list:
            years_biases_dict[release_date] = (years_rating_dict[release_date] / counter_dict[release_date]) - self.r_matrix_avg
        # endregion

        # region build b_years (movie id as index)
        counter = 0
        for movie_id in unique_original_movie_ids:
            movie_metadata = self.movies_metadata[self.movies_metadata.id == movie_id]
            if len(movie_metadata) == 0:
                counter += 1
                continue
            movie_year_raw = movie_metadata["release_date"].values[0]
            if type(movie_year_raw) is str and len(movie_year_raw) > 3 and movie_year_raw[-4:].isdigit() and 0 < int(movie_year_raw[-4:]) < 2022:
                movie_year = int(movie_year_raw[-4:])
            else:
                counter += 1
                continue
            self.b_years[counter] = years_biases_dict["2010's"] if movie_year > 2009 else years_biases_dict["2000's"] if movie_year > 1999 else years_biases_dict["1990's"] if movie_year > 1989 else years_biases_dict["1980's"] if movie_year > 1979 else years_biases_dict["1970's"] if movie_year > 1969 else years_biases_dict["1960's"] if movie_year > 1959 else years_biases_dict["1950's"] if movie_year > 1949 else years_biases_dict["1940's"] if movie_year > 1939 else years_biases_dict["1930's"] if movie_year > 1929 else years_biases_dict["1920's"] if movie_year > 1919 else years_biases_dict["1910's"] if movie_year > 1909 else years_biases_dict["1900's"]
            counter += 1

        # endregion



class mf_params:
    def __init__(self, Q, P, b_u, b_m):
        self.Q = np.copy(Q)
        self.P = np.copy(P)
        self.b_u = np.copy(b_u)
        self.b_m = np.copy(b_m)



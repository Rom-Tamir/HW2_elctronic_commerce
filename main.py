import ex2_312546609_312575970
import pandas as pd
import numpy as np
import time


def transform(ratings, min_items=5, min_users=7):
    """
    Transforms ratings DataFrame
    :param ratings: Ratings DataFrame
    :param min_items: Minimum items per user
    :param min_users: Minimum users per item
    :return: Transformed DataFrame
    """
    ratings = ratings.groupby('user').filter(lambda items: len(items) >= min_items)
    ratings = ratings.groupby('item').filter(lambda users: len(users) >= min_users)
    unique_users = ratings['user'].unique()
    unique_items = ratings['item'].unique()
    user_mapping = {u: k for k, u in enumerate(unique_users)}
    item_mapping = {i: k for k, i in enumerate(unique_items)}
    ratings = ratings.replace({'user': user_mapping, 'item': item_mapping})
    return ratings


def train_test_split(ratings, train_ratio=0.8):
    """
    Splits ratings per user
    :param ratings: Ratings DataFrame
    :param train_ratio: Percentage of ratings in the train set
    :return: A tuple of train and test DataFrames
    """
    train, test = [], []
    for user in ratings.groupby('user'):
        rows = user[1].values.tolist()
        split = int(train_ratio * len(rows))
        indices = np.random.permutation(len(rows))
        for i, row in enumerate(rows):
            if i in indices[:split]:
                train.append(row)
            else:
                test.append(row)
    train = pd.DataFrame(train, columns=ratings.columns, index=None)
    test = pd.DataFrame(test, columns=ratings.columns, index=None)
    return train, test


def main():
    ratings = transform(pd.read_csv('ratings.csv'))
    train, test = train_test_split(ratings)

    """ start = time.time()
    # shuffle the df
    train = train.sample(frac=1)
    neighbors_rmse_dict = dict()
    for k in range(1, 20):
        neighbors_rmse_dict[k] = ex2_312546609_312575970.NeighborhoodRecommender.cross_validation_error(train, k, 5)
    optimal_k = min(neighbors_rmse_dict)
    optimal_neighbors_rmse = neighbors_rmse_dict[optimal_k]
    print(f'the optimal number of neighbors according to cross validation is : {optimal_k} and the optimal rmse is : {optimal_neighbors_rmse}')
    neighborhood_recommender = ex2_312546609_312575970.NeighborhoodRecommender(train, optimal_k)
    print(neighborhood_recommender.rmse(test))
    print(f'Took {time.time() - start:.2f}s')"""

    start = time.time()
    ls_recommender = ex2_312546609_312575970.LSRecommender(train)
    ls_recommender.solve_ls()
    print(ls_recommender.rmse(test))
    print(f'Took {time.time() - start:.2f}s')

    """start = time.time()
    optimal_params = ex2_312546609_312575970.MFRecommender.hyperparameters_tuning(train)
    mf_recommender = ex2_312546609_312575970.MFRecommender(train, optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3])
    print(mf_recommender.omer_rmse(test))
    print(f'Took {time.time() - start:.2f}s')"""


if __name__ == '__main__':
    np.random.seed(0)
    main()

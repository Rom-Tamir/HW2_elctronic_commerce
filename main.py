import ex2_312546609_312575970
import pandas as pd
import numpy as np
import time


def transform(ratings, min_items=5, min_users=5):
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


def data_handling_for_hybrid(transformed_ratings):

    # region transform handling
    original_ratings = pd.read_csv('ratings.csv')
    original_ratings = original_ratings.groupby('user').filter(lambda items: len(items) >= 5)
    original_ratings = original_ratings.groupby('item').filter(lambda users: len(users) >= 5)
    original_movie_ids = original_ratings['item'].tolist()
    transformed_ratings.insert(0, 'original_movie_id', original_movie_ids)
    # endregion

    # region movies metadata handling
    movies_metadata = pd.read_csv('movies_metadata.csv')
    movies_metadata = movies_metadata.drop(
        columns=['adult', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path',
                 'spoken_languages', 'status', 'tagline', 'title', 'video'])
    movies_metadata['id'] = movies_metadata[movies_metadata['id'].map(lambda x: "/" not in x)]['id']
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'])
    # endregion

    return transformed_ratings, movies_metadata


def main():
    ratings = transform(pd.read_csv('ratings.csv'))
    # train, test = train_test_split(ratings)

    """start = time.time()
    # cross validation
    train_sample = train.sample(frac=1)
    neighbors_rmse_dict = dict()
    print('neighbors_rmse_dict = {')
    for k in range(1, 150, 2):
        neighbors_rmse_dict[k] = ex2_312546609_312575970.NeighborhoodRecommender.cross_validation_error(train_sample, k, 5)
        print(f'{k}: {neighbors_rmse_dict[k]}')
    print('}')
    optimal_k = min(neighbors_rmse_dict)
    optimal_neighbors_rmse = neighbors_rmse_dict[optimal_k]
    print(f'The optimal number of neighbors according to cross validation is: {optimal_k} and the optimal RMSE is: {optimal_neighbors_rmse}')

    # RMSE of test
    neighborhood_recommender = ex2_312546609_312575970.NeighborhoodRecommender(train, optimal_k)
    print(f'The Neighborhood Recommender model RMSE on test set, with the optimal k is: {neighborhood_recommender.omer_rmse(test)}')
    print(f'Took {(time.time() - start)/60:.2f} minutes')
    print()
    print("-----------------------------------------------------------------")
    print()

####################################################################################

    start = time.time()
    ls_recommender = ex2_312546609_312575970.LSRecommender(train)
    ls_recommender.solve_ls()
    print(f'The Least Squares Recommender model RMSE is: {ls_recommender.omer_rmse(test)}')
    print(f'Took {(time.time() - start)/60:.2f} minutes')
    print()
    print("-----------------------------------------------------------------")
    print()"""

    ####################################################################################

    """start = time.time()
    #optimal_params = ex2_312546609_312575970.MFRecommender.hyperparameters_tuning(train)
    #mf_recommender = ex2_312546609_312575970.MFRecommender(train, optimal_params[0], optimal_params[1], optimal_params[2], optimal_params[3])
    mf_recommender = ex2_312546609_312575970.MFRecommender(train, 200, 0.01, 0.1, 10)
    original_data = pd.read_csv('ratings.csv')
    original_data = original_data.groupby('user').filter(lambda items: len(items) >= 5)
    original_data = original_data.groupby('item').filter(lambda users: len(users) >= 5)
    original_movies_id = original_data['item'].unique()
    temp_res = np.where(original_movies_id == 31)
    print(mf_recommender.b_m[temp_res])

    print(f'The Matrix Factorization Recommender model RMSE on test set, with the optimal params is: {mf_recommender.omer_rmse(test)}')
    print(f'Took {(time.time() - start)/60:.2f} minutes')
    print()
    print("-----------------------------------------------------------------")
    print()"""

    #####################################################################################

    start = time.time()
    hybrid_data_transformed, movies_metadata = data_handling_for_hybrid(ratings)
    train, test = train_test_split(hybrid_data_transformed)
    hybrid_mf_recommender = ex2_312546609_312575970.HybridMFRecommender(train, 100, 0.01, 0.1, 10, movies_metadata)

    print(
        f'The Matrix Factorization Recommender model RMSE on test set, with the optimal params is: {hybrid_mf_recommender.omer_rmse(test)}')
    print(f'Took {(time.time() - start) / 60:.2f} minutes')


if __name__ == '__main__':
    #np.random.seed(0)
    main()

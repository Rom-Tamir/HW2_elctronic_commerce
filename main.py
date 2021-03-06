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
    #ratings = ratings.groupby('user').filter(lambda items: len(items) >= min_items)
    #ratings = ratings.groupby('item').filter(lambda users: len(users) >= min_users)
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


def data_handling_for_hybrid():

    # region transform handling
    movies_metadata = pd.read_csv('movies_metadata.csv')
    movies_metadata['id'] = movies_metadata[movies_metadata['id'].map(lambda x: "/" not in x)]['id']
    movies_metadata['id'] = pd.to_numeric(movies_metadata['id'])
    original_ratings = pd.read_csv('ratings.csv')
    unique_original_movie_ids = movies_metadata['id'].unique().tolist()
    original_ratings = original_ratings[original_ratings.item.isin(unique_original_movie_ids)]
    original_ratings = original_ratings.groupby('user').filter(lambda items: len(items) >= 5)
    original_ratings = original_ratings.groupby('item').filter(lambda users: len(users) >= 5)
    original_movie_ids = original_ratings['item'].tolist()
    original_ratings = transform(original_ratings)
    original_ratings.insert(0, 'original_movie_id', original_movie_ids)


    # endregion

    # region movies metadata handling
    movies_metadata = movies_metadata.drop(
        columns=['adult', 'belongs_to_collection', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path',
                 'status', 'tagline', 'title', 'video'])
    movies_metadata['runtime'] = pd.to_numeric(movies_metadata['runtime'], errors='coerce')
    movies_metadata['budget'] = pd.to_numeric(movies_metadata['budget'], errors='coerce')
    movies_metadata['popularity'] = pd.to_numeric(movies_metadata['popularity'], errors='coerce')
    #movies_metadata['release_date'] = movies_metadata[movies_metadata['release_date'].map(lambda x: type(x) is str and len(x) > 3 and x[-4:].isdigit())]['release_date']
    # endregion

    return original_ratings, movies_metadata


def main():

    """start = time.time()
    # cross validation
    train_sample = train.sample(frac=1)
    neighbors_rmse_dict = dict()
    print('neighbors_rmse_dict = {')
    for k in range(1, 150, 2):
        neighbors_rmse_dict[k] = ex2_312546609_312575970.NeighborhoodRecommender.cross_validation_error(train_sample, k, 5)
        print(f'{k}: {neighbors_rmse_dict[k]}')
    print('}')

    keys_list = list(neighbors_rmse_dict.keys())
    val_list = list(neighbors_rmse_dict.values())
    optimal_neighbors_rmse = min(val_list)
    optimal_k = keys_list[val_list.index(optimal_neighbors_rmse)]
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
    hybrid_data_transformed, movies_metadata = data_handling_for_hybrid()
    train, test = train_test_split(hybrid_data_transformed)
    hybrid_mf_recommender = ex2_312546609_312575970.HybridMFRecommender(train, 200, 0.01, 0.1, 100, movies_metadata)

    print(
        f'The Matrix Factorization Recommender model RMSE on test set, with the optimal params is: {hybrid_mf_recommender.rmse(test)}')
    print(f'Took {(time.time() - start) / 60:.2f} minutes')


if __name__ == '__main__':
    np.random.seed(0)
    main()

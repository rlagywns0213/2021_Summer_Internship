import pandas as pd
import numpy as np

class Dataset(object):
    #Load ml-100k movie data into pandas with labels.
    df = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    item = pd.read_csv('data/ml-100k/u.item', sep="|", encoding='latin-1', header=None)
    item.columns = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 
                'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 
                'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    user = pd.read_csv('data/ml-100k/u.user', sep="|", encoding='latin-1', names=['user_id','age','sex','occupation','zip_code'],header=None)
    #Declare number of users and movies.
    n_users = df.user_id.unique().shape[0]  #943
    n_movies = df.movie_id.unique().shape[0]  #1682
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    train_df = pd.read_csv('data/ml-100k/ua.base', sep='\t', names=r_cols)
    test_df = pd.read_csv('data/ml-100k/ua.test', sep='\t', names=r_cols)

    n_users = df.user_id.unique().shape[0]
    n_movies = df.movie_id.unique().shape[0]

    #Set rating user-movies matrix
    ml_100k = np.zeros((n_users, n_movies))
    train = np.zeros((n_users, n_movies))
    test = np.zeros((n_users, n_movies))

    for row in df.itertuples():
        ml_100k[row[1]-1, row[2]-1] = row[3]

    for row in train_df.itertuples():
        train[row[1]-1, row[2]-1] = row[3]
        
    for row in test_df.itertuples():
        test[row[1]-1, row[2]-1] = row[3]
import numpy as np
from sklearn.feature_extraction import DictVectorizer

# Read in data
def loadData(filename,path="../data/ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

def vectorize():
    (train_data, y_train, train_users, train_items) = loadData("ua.base")
    (test_data, y_test, test_users, test_items) = loadData("ua.test")
    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)
    return X_train, X_test, y_train, y_test, train_users, train_items, test_users, test_items
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

"""### **Load data**"""

#Reading user file:
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

n_users = users.shape[0]
print('Number of users:', n_users)
users.head() #uncomment this to see some few examples

#Reading ratings file:
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

rate_train = pd.read_csv('./data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
rate_test = pd.read_csv('./data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

# rate_train = ratings_base.as_matrix()
# rate_test = ratings_test.as_matrix()

print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])

print(rate_train.head())
print(rate_test.head())

plt.boxplot(rate_train['rating'])
plt.xticks(rotation=90)

"""### **Establish item profiles**"""

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
        'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')

n_items = items.shape[0]
print('Number of items:', n_items)
items.head(3)

"""* #### *Mark the genres that the films belongs*"""

item_counter = items.iloc[:,5:]
item_counter

"""* #### Erect the TF-IDF matrix for the importance percentage of each word in a film"""

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA

transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
tfidf = transformer.fit_transform(item_counter).toarray()

"""* #### Reduce dimension by PCA"""

pca = PCA(n_components = 10)
tfidf_pca = pca.fit_transform(tfidf)

print(tfidf_pca.shape)
tfidf_pca

"""* #### Reduce dimension by removing high-correlated columns"""

tfidf_df = pd.DataFrame(tfidf)
tfidf_df.head()

plt.figure(figsize=(15,15))
sns.heatmap(tfidf_df.corr(), cmap="YlGnBu", annot=True)

"""* Because of high-level correlation, I decide to remove these columns: Adventure, Animation, Action"""

tfidf_df.drop(columns=[1,2,3,6,10,16], inplace=True)

tfidf_corr = tfidf_df.to_numpy()
tfidf_corr.shape

"""### **Build the recommender model by using the predicted rating of users**"""

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

class ContentBasedWithRating():
    def __init__(self, tfidf:TfidfTransformer, n_users) -> None:
        self.tfidf = tfidf
        self.d = tfidf.shape[1]
        self.W = np.zeros((self.d, n_users))
        self.b = np.zeros((1, n_users))
        self.Yhat = None

    def get_items_rated_by_user(self, rate_matrix, user_id) -> tuple:
        """
        in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
        we care about the first three values
        return (item_ids, scores) rated by user user_id
        """
        y = rate_matrix.iloc[:,0] # all users

        ids = np.where(y == user_id)[0] # users = user_id
        item_ids = rate_matrix.iloc[ids, 1] # movie_ids of user_id
        scores = rate_matrix.iloc[ids, 2] # rates of those movie_ids
        return item_ids, scores

    def regression_func(self, movie_ids, user_id):
        return abs(self.tfidf[movie_ids-1, :].dot(self.W[:, user_id-1]) + self.b[0, user_id-1])

    def fit_transform(self, rate_train:iter, n_users:int, lamda=3):
        for n in range(1, n_users+1):
            ids, scores = self.get_items_rated_by_user(rate_train, n)   # choose the movie_ids and scores of rated movies by user_n

            clf = Ridge(alpha=lamda, fit_intercept = True)   # Predict rating of this user by the linear model + regularization
            Xhat = self.tfidf[ids-1, :]  # choose the feature vectors of movies were rated by user_n
            clf.fit(Xhat, scores)

            self.W[:, n-1] = clf.coef_ # user profile: (n_movies, user-n)
            self.b[0, n-1] = clf.intercept_

        # Predict the whole users' rating for each film which is Utility matrix
        self.Yhat = abs(self.tfidf.dot(self.W) + self.b)

    def predict(self, user_id:int, rate_test:iter, Yhat) -> tuple:
        np.set_printoptions(precision=2) # 2 digits after .
        movie_ids, scores = self.get_items_rated_by_user(rate_test, user_id)
        print('Rated movies ids :', movie_ids.to_list())
        print('True ratings     :', scores.to_list())
        print('Predicted ratings:', self.regression_func(movie_ids, user_id))
        # print('Predicted ratings:', Yhat[movie_ids-1, user_id-1])
        # Plotting linear regression of rate_test and predicts
        sns.regplot(x=Yhat[movie_ids-1, user_id-1], y=scores)
        plt.xlabel('Predicted scores')
        plt.ylabel('Actual scores')

        return movie_ids.to_list(), Yhat[movie_ids-1, user_id-1]

    def predict_all(self, rates:iter, n_users:int):
        users, ids, reals, preds = [], [], [], []
        for n in range(1, n_users+1):
            movie_ids, scores_truth = self.get_items_rated_by_user(rates, n)
            scores_pred = self.regression_func(movie_ids, n)
            # scores_pred = self.Yhat[movie_ids-1, n-1]
            # Aggerate the performance info
            ids += movie_ids.to_list()
            users += [n for i in range(len(movie_ids.to_list()))]
            reals += scores_truth.to_list()
            preds += scores_pred.tolist()

        df = pd.DataFrame({'movie_id':ids, 'user_id':users,
                           'real':reals, 'predict':preds})
        return df

    def RMSE(self, rates:iter) -> float:
        se, cnt = 0.0, 0
        for n in range(1, n_users+1):
            ids, scores_truth = self.get_items_rated_by_user(rates, n)
            scores_pred = self.regression_func(ids, n)
            # scores_pred = self.Yhat[ids-1, n-1]
            # Calculate sigma of the squared differences
            e = scores_truth - scores_pred
            se += (e*e).sum(axis = 0)
            cnt += e.size

        return np.sqrt(se/cnt)

    def r2_score(self, rates:iter, n_users:int) -> float:
        users, ids, reals, preds = [], [], [], []
        for n in range(1, n_users+1):
            movie_ids, scores_truth = self.get_items_rated_by_user(rates, n)
            scores_pred =self.regression_func(movie_ids, n)
            # scores_pred = self.Yhat[movie_ids-1, n-1]
            # Aggerate the performance info
            try:
                ids += movie_ids.to_list()
                users += [n for i in range(len(movie_ids.to_list()))]
                reals += scores_truth.to_list()
                preds += scores_pred.tolist()
            except Exception as ex:
                pass

        r2 = r2_score(reals, preds)
        return r2

def recommend(items, user_id, Yhat) -> tuple:
    return items['movie id'], Yhat[items['movie id']-1, user_id-1]

def search_regularization(model, train, test, users, parameters, legend):
    mse = []
    for learning_rate in parameters:
        model.fit_transform(train, users, lamda=learning_rate)
        mse.append(model.RMSE(test))

    print(mse)
    sns.lineplot(x=parameters, y=mse)

recommender = ContentBasedWithRating(tfidf=tfidf, n_users=n_users)
recommender_pca = ContentBasedWithRating(tfidf=tfidf_pca, n_users=n_users)
recommender_corr = ContentBasedWithRating(tfidf=tfidf_corr, n_users=n_users)

parameters = [5, 7, 9, 11, 13, 15, 17]

search_regularization(recommender, rate_train, rate_test, n_users, parameters, 'origin')
search_regularization(recommender_pca, rate_train, rate_test, n_users, parameters, 'pca')
search_regularization(recommender_corr, rate_train, rate_test, n_users, parameters, 'corr')

recommender.fit_transform(rate_train, n_users, lamda=7)
recommender_pca.fit_transform(rate_train, n_users, lamda=7)
recommender_corr.fit_transform(rate_train, n_users, lamda=7)

"""Testing prediction on one user"""

_, y_pred_test = recommender.predict(1, rate_test, recommender.Yhat)
_, y_true_test = recommender.get_items_rated_by_user(rate_test, 1)

print('R2 for testing', r2_score(y_pred=y_pred_test, y_true=y_true_test))
print('MSE for testing', mean_squared_error(y_pred=y_pred_test, y_true=y_true_test))

"""#### Evaluate the model"""

print('After remaining origin dimension: ')
print('-> RMSE for testing', recommender.RMSE(rate_test))
print('-> R2-score for testing', recommender.r2_score(rate_test, n_users))
print('\n')
print('After applying PCA for reducing dimensions (n_component=10): ')
print('-> RMSE for testing', recommender_pca.RMSE(rate_test))
print('-> R2-score for testing', recommender_pca.r2_score(rate_test, n_users))
print('\n')
print('After removing high-correlated columns for reducing dimensions: ')
print('-> RMSE for testing', recommender_corr.RMSE(rate_test))
print('-> R2-score for testing', recommender_corr.r2_score(rate_test, n_users))

result = recommender.predict_all(rate_test, n_users)
result

result[['real','predict']].boxplot()

ax1 = sns.distplot(result['real'], hist=False, color='red', label='Actual value')
ax2 = sns.distplot(result['predict'], hist=False, color='blue', label='Predicted value', ax=ax1)
plt.legend()

"""### **Dump Yhat to csv file**"""

# np.savetxt('./artificats/recommends_pca.csv', recommender.Yhat, delimiter=',')

Yhat = np.loadtxt('./artificats/recommends.csv', delimiter=',')

print('Ma trận Utility giữa movie và user:', Yhat.shape)
Yhat

def recommend(items, user_id, Yhat):
    return Yhat[items['movie id']-1, user_id-1]

# sorted(recommend(items, 1, Yhat).tolist(), reverse=True)
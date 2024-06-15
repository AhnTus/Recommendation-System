from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

class ContentBasedWithRating():
    def __init__(self, tfidf:TfidfTransformer, n_users:int) -> None:
        self.tfidf = tfidf
        self.n_users = n_users
        self.d = tfidf.shape[1]
        self.W = np.zeros((self.d, n_users))
        self.b = np.zeros((1, n_users))
        self.Yhat = None
    
    def get_items_rated_by_user(self, rate_matrix, user_id) -> tuple:
        y = rate_matrix.iloc[:,0] # all users
        
        ids = np.where(y == user_id)[0] # users = user_id
        item_ids = rate_matrix.iloc[ids, 1] # movie_ids of user_id
        scores = rate_matrix.iloc[ids, 2] # rates of those movie_ids
        return item_ids, scores

    def fit_transform(self, rate_train:iter) -> None:
        for n in range(1, self.n_users+1):    
            ids, scores = self.get_items_rated_by_user(rate_train, n)
            clf = Ridge(alpha=0.01, fit_intercept = True)
            Xhat = self.tfidf[ids-1, :]
            
            clf.fit(Xhat, scores) 
            self.W[:, n-1] = clf.coef_
            self.b[0, n-1] = clf.intercept_
            
        self.Yhat = self.tfidf.dot(self.W) + self.b
            
    def predict(self, user_id:int, rate_test:iter, Yhat) -> tuple:
        np.set_printoptions(precision=2) # 2 digits after . 
        movie_ids, scores = self.get_items_rated_by_user(rate_test, user_id)
        print('Rated movies ids :', movie_ids.to_list())
        print('True ratings     :', scores.to_list())
        print('Predicted ratings:', Yhat[movie_ids-1, user_id-1])
        # Plotting linear regression of rate_test and predicts
        sns.regplot(x=Yhat[movie_ids-1, user_id-1], y=scores)
        plt.xlabel('Predicted scores')
        plt.ylabel('Actual scores')
        
        return movie_ids.to_list(), Yhat[movie_ids-1, user_id-1]
    
    def MSE(self, rates:iter) -> float:
        for n in range(1, self.n_users+1):
            ids, scores_truth = self.get_items_rated_by_user(rates, n)
            scores_pred = self.Yhat[ids-1, n-1]
            mse = mean_squared_error(y_pred=scores_pred, y_true=scores_truth)
            # e = scores_truth - scores_pred 
            # se += (e*e).sum(axis = 0)
            # cnt += e.size 
        return mse
    
def recommend(items, user_id, Yhat):
    return items['movie id'], Yhat[items['movie id']-1, user_id-1]
    
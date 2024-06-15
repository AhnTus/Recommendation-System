# General Information
In this project, I built a recommendation system based on Content-based. Besides, I divided into two types which the first one will recommend without ratings data and the second one will based on ratings data

In the first model, I have used [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). It contains two dataset that are credits and movies, credits will contain data about cast, crew and title while the movies contain the information of movies such as genre, budget, overview or homepage. This model base only on features of a movie and find top N similar movies

In the recommender based on ratings, I have used [MovieLens](https://grouplens.org/datasets/movielens/) dataset which is stable and contain 100,000 ratings from 1000 users on 1700 movies. Released 4/1998. I built a recommendation system based on Content-Based method with ratings data. It will recommend movies for a user based on their ratings data of other movies

# Problem Solving

### Content-based without ratings
This model just only uses the TF-IDF and Cosine Similarity. 
* Firstly, I apply TF-IDF technique for measuring the importance probability of each word in `tags` (tags is a overall information of a movie)
* Then, I consider each row of TF-IDF matrix like a vector about features of a movie, called features vector
* I apply Cosine Similarity to figure out the angle between target movie and each of recommended movies, the lower angle is, the recommended movies are more similar to target one

### Content-based with ratings

With this type, I use movies data and combine ratings one for predicting the ratings of unrated movies for each user.
 * I also apply `TF-IDF matrix` for create feature vectors of each movie
 * I use Ridge algorithm for learning ratings of each user on their rated movies on the purpose which return the coefficient `W` and intercept `b`
 * I use those parameters to predict the user ratings on the equation:

$$
Yhat = tfidf * W + b
$$

# Run Project

> Note: 
> * Python version 3.8+
> * Install all packages in *requirements.txt*

#### Run Streamlit webpage
```python
streamlit run app.py
```



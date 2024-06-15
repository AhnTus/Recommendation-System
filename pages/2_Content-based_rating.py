import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import requests
sys.path.append('model')
from ContentBasedWithRating import recommend

st.set_page_config(layout="wide")
st.header("Content-based Recommender System With Ratings")
dataframe_tab, chart_tab, recd_tab = st.tabs(['Data table', 'Chart', 'Recommender'])

#define dataframes
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure', 
            'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols, encoding='latin-1')
p_cols = ['movie_id', 'poster']
item_posters = pd.read_csv('./data/ml-100k/movie_poster.csv', sep=',', names=p_cols)

u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

Yhat = np.loadtxt('./artificats/recommends.csv', delimiter=',')


def get_genres(item):
    genres = [genre for genre in items.columns[6:25] if item[genre] == 1]
    return genres

def fetch_poster(movie_id) -> str:
    url = "https://api.themoviedb.org/3/movie/{}?api_key=3fed181afbe284769c6a495334dc66ea&language=en-US".format(movie_id)
    resp = requests.get(url)
    poster_path = resp.json().get('poster_path')
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

with dataframe_tab:
    st.write('Movie table', items)
    st.write('User table',users)
    
with recd_tab:
    user_id = st.sidebar.selectbox(
        'Choose a user id',
        users.iloc[:,0]
    )
    age = users.at[user_id-1, 'age']
    sex = users.at[user_id-1, 'sex']
    occupation = users.at[user_id-1, 'occupation']
    
    st.sidebar.text_input(label='Age',value='{}'.format(age))
    st.sidebar.text_input(label='Sex',value='{}'.format(sex))
    st.sidebar.text_input(label='Occupation',value='{}'.format(occupation))
    
    if st.sidebar.button('Recommend'):
        movie_ids, scores = recommend(items, user_id, Yhat)
        recommends = pd.DataFrame({'movie_id':movie_ids, 'scores':scores, 
                                   'title':items['movie title'].to_list()})
        
        recommends = recommends.merge(item_posters, how='left', on='movie_id')
        recommends = recommends.sort_values(by=['scores'], ascending=False).reset_index()
        st.write(recommends)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col6, col7, col8, col9, col10 = st.columns(5)
        
        for col, idx in zip([col1, col2, col3, col4, col5], range(5)):
            # for idx in range(5):
            with col:
                st.text(f"{recommends.loc[idx, 'title']} - ID: {idx}")
                st.text(get_genres(items.loc[idx, :]))
                try:
                    st.image(recommends.loc[idx, 'poster'])
                    # st.image(fetch_poster(recommends.loc[idx, 'movie_id']))
                except Exception as exc:
                    pass
                    
        for col, idx in zip([col6, col7, col8, col9, col10], range(5, 10)):
            # for idx in range(5, 10):
            with col:
                st.text(f"{recommends.loc[idx, 'title']} - ID: {idx}")
                st.text(get_genres(items.loc[idx, :]))
                try:
                    st.image(recommends.loc[idx, 'poster'])
                    # st.image(fetch_poster(recommends.loc[idx, 'movie_id']))
                except Exception as exc:
                    pass
        
with chart_tab:
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax = sns.histplot(data=recommends, x='scores', kde=True)
        ax.grid()
        st.pyplot(fig)
    except Exception as exc:
        pass

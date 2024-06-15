import requests
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(layout="wide")
st.header("Content-based Recommender System Without Ratings")
movies = pickle.load(open("./artificats/movies.pkl", "rb"))
similarities = pickle.load(open("./artificats/similarities.pkl", "rb"))
movie_titles = movies['title'].values

selected_movie = st.selectbox (
    "Choose your movie",
    movie_titles
)

def fetch_poster(movie_id) -> str:
    url = "https://api.themoviedb.org/3/movie/{}?api_key=3fed181afbe284769c6a495334dc66ea&language=en-US".format(movie_id)
    resp = requests.get(url)
    poster_path = resp.json().get('poster_path')
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(title) -> tuple:
    index = movies[movies['title'] == title].index[0]
    distances = sorted(list(enumerate(similarities[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for distance in distances[0:10]:
        movie_id = movies.loc[distance[0], 'movie_id']
        recommended_movie_names.append(movies.loc[distance[0], 'title'])
        recommended_movie_posters.append(fetch_poster(movie_id))
        
    return recommended_movie_names, recommended_movie_posters

if st.button("Show recommendations"):
    name, poster = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)
    col6, col7, col8, col9, col10 = st.columns(5)
    
    for col, idx in zip([col1, col2, col3, col4, col5], range(5)):
        with col:
            st.text(f"{name[idx]} - ID: {idx}")
            # st.text(get_genres(items.loc[idx, :]))
            try:
                st.image(poster[idx])
            except Exception as exc:
                pass
                    
    for col, idx in zip([col6, col7, col8, col9, col10], range(5, 10)):
        with col:
            st.text(f"{name[idx]} - ID: {idx}")
            # st.text(get_genres(items.loc[idx, :]))
            try:
                st.image(poster[idx])
            except Exception as exc:
                pass
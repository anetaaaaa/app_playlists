import base64
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import io
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SCOPE, URI, SECRET_KEY
import requests
import time

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load models
ResNet50V2_Model = tf.keras.models.load_model('ResNet50V2_Model.h5')
keras_model = load_model('mood_model.h5')

# Define the Flask app
app = Flask(__name__, static_url_path='/static')
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_NAME'] = 'flask_session'

# Class names for the prediction
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

reverse_mood_encoding = {
    0: 'calm',
    1: 'energetic',
    2: 'happy',
    3: 'sad'
}

# Function to search for songs on Spotify based on client credentials token
def search_spotify(song_name, token):
    search_url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': f'Bearer {token}',
    }
    params = {
        'q': song_name,
        'type': 'track',
        'limit': 1,
    }
    response = requests.get(search_url, headers=headers, params=params)
    response_data = response.json()
    if response_data['tracks']['items']:
        return response_data['tracks']['items'][0]['external_urls']['spotify']
    else:
        return None

# Function to get track data for mood recognition
def get_track_info(track_name):
    token_info = session.get("token_info")
    if not token_info:
        return redirect("/login")
    else:
        sp = spotipy.Spotify(auth=token_info.get('access_token'))
        results = sp.search(q=track_name, limit=1)
        track_info = []

        for item in results['tracks']['items']:
            track_id = item['id']
            name = item['name']

            audio_features = sp.audio_features(track_id)[0]

            track_info.append({
                'name': name,
                'acousticness': audio_features['acousticness'],
                'danceability': audio_features['danceability'],
                'energy': audio_features['energy'],
                'instrumentalness': audio_features['instrumentalness'],
                'liveness': audio_features['liveness'],
                'loudness': audio_features['loudness'],
                'speechiness': audio_features['speechiness'],
                'tempo': audio_features['tempo'],
                'valence': audio_features['valence'],
                'popularity': item['popularity']
            })

        return track_info

# Function to create playlist dataframe
def create_playlist_dataframe(playlist_name):
    token_info = session.get("token_info")
    if not token_info:
        return redirect("/login")
    else:
        sp = spotipy.Spotify(auth=token_info.get('access_token'))
        #print(token_info["access_token"])
        playlists = sp.search(q=playlist_name, type='playlist')

        if playlists['playlists']['items']:
            playlist = playlists['playlists']['items'][0]
            results = sp.playlist_tracks(playlist['id'])
            all_tracks_info = []

            for track in results['items']:
                track_name = track['track']['name']
                track_info = get_track_info(track_name)
                all_tracks_info.extend(track_info)

            df = pd.DataFrame(all_tracks_info)
            df.set_index('name', inplace=True)
            if (len(df)>0):
                return df
            else:
                return("Something went wrong")
            
# Function to predict mood for tracks
def predict_mood_for_tracks(df):
    scaler = MinMaxScaler()
    X = df.drop(columns=['popularity'])
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    mood_predictions = np.argmax(keras_model.predict(X), axis=1)
    df['mood'] = mood_predictions
    df['mood'] = df['mood'].map(reverse_mood_encoding)
    return df

# Function to recommend songs to boost the mood
def Recommend_Songs(pred_class, Music_Player):
    if pred_class in ['Neutral', 'Sad']:
        Play = Music_Player[Music_Player['mood'].isin(['happy', 'energetic'])]
    elif pred_class in ['Fear', 'Angry', 'Disgust', 'Surprise']:
        Play = Music_Player[Music_Player['mood'] == 'calm']
    elif pred_class in ['Happy']:
        Play = Music_Player[Music_Player['mood'] == 'energetic']
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index()
    return Play['name'].tolist()

# Function to recommend songs same as mood
def Recommend_Songs_moods(pred_class, Music_Player):
    if pred_class == 'Happy':
        Play = Music_Player[Music_Player['mood'].isin(['happy'])]
    elif pred_class in ['Angry', 'Surprise']:
        Play = Music_Player[Music_Player['mood'] == 'energetic']
    elif pred_class in ['Sad', 'Fear']:
        Play = Music_Player[Music_Player['mood'] == 'sad']
    elif pred_class =='Neutral':
        Play = Music_Player[Music_Player['mood'] == 'calm']
    elif pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'].isin(['sad', 'calm'])]
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index()
    return Play['name'].tolist()

# Function to load and prepare the image
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_prep_image(image_stream, img_shape=224):
    image_bytes = image_stream.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        print("Failed to decode image")
        return None, None
    
    GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(GrayImg, 1.1, 4)

    if len(faces) == 0:
        print("No faces detected")
        return None, None

    for x, y, w, h in faces:
        roi_GrayImg = GrayImg[y: y + h, x: x + w]
        roi_Img = img[y: y + h, x: x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        if len(faces) == 1:
            RGBImg = cv2.cvtColor(roi_Img, cv2.COLOR_BGR2RGB)
            RGBImg = cv2.resize(RGBImg, (img_shape, img_shape))
            RGBImg = RGBImg / 255.
            return RGBImg, img_base64

    print("Multiple faces detected")
    return None, None

# Function to predict and recommend songs
def pred_and_recommend(image_file, class_names, playlist_name):
    img, img_base64 = load_and_prep_image(image_file)
    if img is None:
        print("Image preprocessing failed")
        return None, None, None, None
    pred = ResNet50V2_Model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    df = create_playlist_dataframe(playlist_name)
    df = predict_mood_for_tracks(df)
    songs = Recommend_Songs(pred_class, df)
    moody_songs = Recommend_Songs_moods(pred_class, df)
    token_info = session.get("token_info", None)
    token = token_info["access_token"]
    song_links = []
    moody_song_links = []
    for song in songs:
        link = search_spotify(song, token)
        song_links.append({'title': song, 'link': link})

    for moody_song in moody_songs:
        moody_link = search_spotify(moody_song, token)
        moody_song_links.append({'title': moody_song, 'link': moody_link})
    return pred_class, song_links, moody_song_links, img_base64

# Checks to see if token is valid and gets a new token if not
def get_token(session):
    token_valid = False
    token_info = session.get("token_info", {})

    # Checking if the session already has a token stored
    if not (session.get('token_info', False)):
        token_valid = False
        return token_info, token_valid

    # Checking if token has expired
    now = int(time.time())
    is_token_expired = token_info.get('expires_at') - now < 60

    # Refreshing token if it has expired
    if (is_token_expired):
        # Don't reuse a SpotifyOAuth object because they store token info and you could leak user tokens if you reuse a SpotifyOAuth object
        sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=URI, scope=SCOPE)
        token_info = sp_oauth.refresh_access_token(session.get('token_info').get('refresh_token'))

    token_valid = True
    return token_info, token_valid

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/check_auth', methods=['GET'])
def check_auth():
    token_info = session.get("token_info", {})
    if not token_info:
        return jsonify({'authenticated': False})
    session['token_info'] = token_info
    return jsonify({'authenticated': True})

@app.route('/login')
def login():
    sp_oauth = SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=URI, scope=SCOPE)
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
    # Don't reuse a SpotifyOAuth object because they store token info and you could leak user tokens if you reuse a SpotifyOAuth object
    sp_oauth = spotipy.oauth2.SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=URI, scope=SCOPE)
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)

    # Saving the access token along with all other token related info
    session["token_info"] = token_info
    return redirect(url_for('index'))

@app.route('/user_playlists', methods=['GET'])
def user_playlists():
    session['token_info'], authorized = get_token(session)
    #print(session.get('token_info').get('access_token'))
    session.modified = True
    if not authorized:
        return redirect('/login')
    sp = spotipy.Spotify(auth=session.get('token_info').get('access_token'))
    playlists = sp.current_user_playlists(limit=50)
    playlist_names = [playlist['name'] for playlist in playlists['items']]
    return jsonify(playlist_names if playlist_names else [])

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    playlist_name = request.form.get('playlist')
    pred_class, songs, songs_mood, img_base64 = pred_and_recommend(image_file.stream, class_names, playlist_name)
    if pred_class is None:
        return jsonify({'class': None, 'songs': [], 'songs_mood': [], 'image': None})

    #Returning predicted class and recomennded songs
    return jsonify({'class': pred_class, 'songs': songs, 'songs_mood': songs_mood, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)

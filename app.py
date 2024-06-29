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
import spotipy.util as util
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SCOPE, URI, SECRET_KEY
import requests

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Use a non-interactive backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

script_dir = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_dir)
# Load the ResNet50V2 model for emotion recognition
ResNet50V2_Model = tf.keras.models.load_model('ResNet50V2_Model.h5')

# Loading Keras model for song mood recognition
keras_model = load_model('pls_work.h5')
 

# Load the dataset
#Music_Player = pd.read_csv('data_moods2.csv')

# Define the Flask app
app = Flask(__name__, static_url_path='/static')
app.secret_key = SECRET_KEY

# Class names for the prediction
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

reverse_mood_encoding = {
    0: 'calm',
    1: 'energetic',
    2: 'happy',
    3: 'sad'
}

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID,
                                                    client_secret=SPOTIFY_CLIENT_SECRET,
                                                    redirect_uri=URI,
                                                    scope=SCOPE))


# Get Spotify access token
def get_spotify_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = {
        'Authorization': 'Basic ' + base64.b64encode((SPOTIFY_CLIENT_ID + ':' + SPOTIFY_CLIENT_SECRET).encode()).decode('utf-8'),
    }
    auth_data = {
        'grant_type': 'client_credentials',
    }
    response = requests.post(auth_url, headers=auth_header, data=auth_data)
    response_data = response.json()
    return response_data['access_token']

# Function to search for songs on Spotify
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

# Function to get user's playlists
def get_user_playlists(username):
    token = util.prompt_for_user_token(username, scope=SCOPE, client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=URI)
    if token:
        sp = spotipy.Spotify(auth=token)
        playlists = sp.user_playlists(username)
        playlist_names = [playlist['name'] for playlist in playlists['items']]
        return playlist_names
    else:
        return None


#Function to get track data for mood recognition
def get_track_info(track_name):
    # Search for the track by name
    results = sp.search(q=track_name, limit=1)
    track_info = []

    for item in results['tracks']['items']:
        track_id = item['id']
        name = item['name']

        # Get audio features for the track
        audio_features = sp.audio_features(track_id)[0]

        # Collect the required information
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

def create_playlist_dataframe(playlist_name):
    # Search for playlists by name
    playlists = sp.search(q=playlist_name, type='playlist')

    # Check if any playlists were found
    if playlists['playlists']['items']:
        # Get the first playlist found (you can enhance this logic if needed)
        playlist = playlists['playlists']['items'][0]

        # Get tracks from the playlist
        results = sp.playlist_tracks(playlist['id'])

        # Initialize an empty list to collect track information
        all_tracks_info = []

        # Iterate over tracks in the playlist
        for track in results['items']:
            track_name = track['track']['name']
            track_info = get_track_info(track_name)
            all_tracks_info.extend(track_info)

        # Create a DataFrame from the collected track information
        df = pd.DataFrame(all_tracks_info)
        df.set_index('name', inplace=True)  # Set index on track name
        return df
    else:
        return None

#Function to predict mood for track
def predict_mood_for_tracks(df):
    scaler = MinMaxScaler()
    X = df.drop(columns=['popularity'])  # Exclude 'popularity' column
    # Example: Replace 'mood_predictions' with your model's prediction function
    # Replace with your model's prediction function
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    mood_predictions = np.argmax(keras_model.predict(X), axis=1)
    # Add mood predictions to DataFrame
    df['mood'] = mood_predictions
    df['mood'] = df['mood'].map(reverse_mood_encoding)
    return df

# Function to recommend songs to boost the mood
def Recommend_Songs(pred_class, Music_Player):
    if pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'].isin(['Happy', 'Energetic'])]
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    elif pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    return Play['name'].tolist()

#Function to recommend songs same as mood
def Recommend_Songs_moods(pred_class, Music_Player):
    if pred_class in ['Happy']:
        Play = Music_Player[Music_Player['mood'].isin(['Happy'])]
    elif pred_class == 'Angry':
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    elif pred_class in ['Sad', 'Fear']:
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Surprise', 'Neutral']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    return Play['name'].tolist()

# Function to load and prepare the image
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_prep_image(image_stream, img_shape=224):
    image_bytes = image_stream.read()
    
    # Convert the image bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image array
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
        
        # Plot the image with matplotlib and convert it to base64
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        # Ensure there's only one detected face to return
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
        return None, None, None
    pred = ResNet50V2_Model.predict(np.expand_dims(img, axis=0))
    pred_class = class_names[pred.argmax()]
    df = create_playlist_dataframe(playlist_name)
    df = predict_mood_for_tracks(df)
    songs = Recommend_Songs(pred_class, df)
    # Get Spotify access token
    token = get_spotify_token()
    song_links = []
    for song in songs:
        link = search_spotify(song, token)
        song_links.append({'title': song, 'link': link})
    return pred_class, song_links, img_base64 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Check authentication route
@app.route('/check_auth', methods=['GET'])
def check_auth():
    if 'spotify_token' in session:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})
    
@app.route('/user_playlists', methods=['GET'])
def user_playlists():
    if 'spotify_token' in session:
        sp = spotipy.Spotify(auth=session['spotify_token'])
        playlists = sp.current_user_playlists(limit=50)
        playlist_names = [playlist['name'] for playlist in playlists['items']]
        return jsonify(playlist_names)
    else:
        return jsonify([])
    
# Callback route for Spotify OAuth
@app.route('/callback')
def callback():
    # Handle Spotify OAuth callback
    token_info = sp.get_access_token(request.args['code'])
    session['spotify_token'] = token_info['access_token']
    return redirect('/')

# Login route to handle Spotify OAuth and store token in session
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    scope = request.form.get('scope')
    redirect_uri = request.form.get('redirect_uri')
    
    token = util.prompt_for_user_token(username, scope=scope,
                                       client_id=SPOTIFY_CLIENT_ID,
                                       client_secret=SPOTIFY_CLIENT_SECRET,
                                       redirect_uri=redirect_uri)
    
    if token:
        session['spotify_token'] = token
        session['spotify_username'] = 'anetgirl'
        return redirect('/')
    else:
        return 'Failed to authenticate with Spotify.'

@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['file']
    playlist_name = request.form.get('playlist')
    pred_class, songs, img_base64 = pred_and_recommend(image_file.stream, class_names, playlist_name)
    if pred_class is None:
        return jsonify({'class': None, 'songs': [], 'image': None})

    return jsonify({'class': pred_class, 'songs': songs, 'image': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
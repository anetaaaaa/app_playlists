# Emotion based music recommendation 

Model for emotion recognition from kaggle: (https://www.kaggle.com/code/abduulrahmankhalid/emotion-based-music-recommender-resnet50v2/notebook).

Model for mood recognition for songs from: (https://mikemoschitto.medium.com/deep-learning-and-music-mood-classification-of-spotify-songs-b2dda2bf455)

This is a web interface implemented on [Python](https://www.python.org) that uses a model to recognize emotion and recommend songs from Spotify.

## Install

* Go to the root of cloned repository
* Install dependencies by running `pip3 install -r requirements.txt`

## Run

Execute:

```
python3 app.py
```

It will start a webserver on http://127.0.0.1:5000. Use any web browser to open the web interface.

Using the interface you can capture the screenshot from your webcam and see predicted emotion. You can choose your playlist from the dropdown menu. Two lists are displayed to either improve your emotional state or match your emotional state. Each list displays up to 5 songs. 

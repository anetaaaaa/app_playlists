# Emotion based music recommendation 

Model for emotion recognition from kaggle: (https://www.kaggle.com/code/abduulrahmankhalid/emotion-based-music-recommender-resnet50v2/notebook).

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

Using the interface you can capture the screenshot from your webcam and see predicted emotion. Next you can see up to 5 songs recommended for predicted emotion.

import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ----------------------------
# LOAD DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spotify_songs.csv")

df = pd.read_csv(csv_path)

# remove duplicates early (IMPORTANT)
df = df.drop_duplicates(subset=["track_name", "track_artist"])

# ----------------------------
# AUDIO FEATURES
# ----------------------------
audio_features = [
    'danceability','energy','key','loudness','mode',
    'speechiness','acousticness','instrumentalness',
    'liveness','valence','tempo'
]

df[audio_features] = df[audio_features].fillna(0)

scaler = StandardScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

X = df[audio_features]

# ----------------------------
# HOME
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# AUTOCOMPLETE (clean)
# ----------------------------
@app.route("/api/suggest")
def suggest():
    q = request.args.get("q", "").strip().lower()

    if not q:
        return jsonify([])

    matches = df[
        df["track_name"].str.lower().str.contains(q, na=False)
    ]

    return jsonify(
        matches["track_name"].head(10).tolist()
    )

# ----------------------------
# RECOMMENDATION ENGINE
# ----------------------------
@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    track = data.get("track", "").strip().lower()

    # ----------------------------
    # BETTER MATCHING
    # ----------------------------
    match = df[df["track_name"].str.lower() == track]

    if match.empty:
        match = df[df["track_name"].str.lower().str.contains(track, na=False)]

    if match.empty:
        return jsonify({"found": False})

    # choose most popular match if multiple exist
    match = match.sort_values("track_popularity", ascending=False)

    idx = match.index[0]

    # ----------------------------
    # COSINE SIMILARITY
    # ----------------------------
    scores = cosine_similarity(X.iloc[idx:idx+1], X)[0]

    # ----------------------------
    # DIVERSITY FIX (reduce same artist bias)
    # ----------------------------
    seed_artist = df.iloc[idx]["track_artist"]

    for i in range(len(scores)):
        if df.iloc[i]["track_artist"] == seed_artist:
            scores[i] *= 0.7  # penalize same artist

    # ----------------------------
    # TOP RESULTS
    # ----------------------------
    top_idx = scores.argsort()[::-1][1:30]

    results_df = df.iloc[top_idx][['track_name', 'track_artist']]

    # remove duplicates
    results_df = results_df.drop_duplicates(subset=["track_name"])
    results_df = results_df.drop_duplicates(subset=["track_artist"], keep="first")

    results = results_df.head(5).to_dict(orient="records")

    return jsonify({
        "found": True,
        "results": results
    })

# ----------------------------
# RUN (Render-safe)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

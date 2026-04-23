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

# 🔥 FIX 1: remove duplicate songs at source
df = df.drop_duplicates(subset=["track_name", "track_artist"])

# ----------------------------
# AUDIO FEATURES
# ----------------------------
audio_features = [
    'danceability','energy','key','loudness','mode',
    'speechiness','acousticness','instrumentalness',
    'liveness','valence','tempo'
]

# Clean missing values
df[audio_features] = df[audio_features].fillna(0)

# Normalize features
scaler = StandardScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# Feature matrix
X = df[audio_features]

# ----------------------------
# ROUTES
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# AUTOCOMPLETE
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

    # 🔥 FIX 2: better matching (exact first, fallback second)
    match = df[df["track_name"].str.lower() == track]

    if match.empty:
        match = df[df["track_name"].str.lower().str.contains(track, na=False)]

    if match.empty:
        return jsonify({"found": False})

    idx = match.index[0]

    # cosine similarity
    scores = cosine_similarity(X.iloc[idx:idx+1], X)[0]

    # take more candidates first
    top_idx = scores.argsort()[::-1][1:20]

    results_df = df.iloc[top_idx][['track_name', 'track_artist']]

    # 🔥 FIX 3: remove duplicates properly
    results_df = results_df.drop_duplicates(subset=["track_name", "track_artist"]).head(5)

    results = results_df.to_dict(orient='records')

    return jsonify({
        "found": True,
        "results": results
    })

# ----------------------------
# RUN SERVER (Render-safe)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

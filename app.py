import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "spotify_songs.csv")

df = pd.read_csv(csv_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    track = data.get("track")

    match = df[df["track_name"] == track]

    if match.empty:
        return jsonify({"found": False})

    return jsonify({
        "found": True,
        "message": "Backend is working 🎉"
    })

@app.route("/api/suggest")
def suggest():
    q = request.args.get("q", "")
    matches = df[df["track_name"].str.lower().str.contains(q.lower(), na=False)]
    return jsonify(matches["track_name"].head(10).tolist())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
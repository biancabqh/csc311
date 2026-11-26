"""
Final prediction script for CSC311 project.
This script is self-contained and adheres strictly to MarkUs autograder rules:
  - Imports ONLY standard Python libraries, numpy, and pandas.
  - Implements predict_all(filename) taking an unseen test CSV path
    and returning a list of sport label predictions.
  - Utilizes a pre-trained Regularized Softmax Regression model stored
    in model_bundle.json (23 KB, well below 10MB limit).

Implementation note:
Multinomial Logistic Regression is utilized for standalone inference to comply
with MarkUs restrictions prohibiting external machine learning frameworks (sklearn).
Precomputed model parameters and preprocessing statistics are loaded from
model_bundle.json to enable deterministic, lightweight NumPy vectorization.
"""

import sys
import os
import json
import re
import numpy as np
import pandas as pd

# Load model bundle containing learned weights, biases, vocabs, and scalers
BUNDLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_bundle.json")

RENAME_MAP = {
    "How easy do you think it would be for someone with no prior experience to start playing this sport recreationally? Rate it on a scale from 0 to 10, where 0 means extremely difficult and 10 means extremely easy.": "difficulty_rating",
    "Success in this sport depends heavily on strategy and planning.": "strategy_importance",
    "Players must be comfortable with a high risk of injury.": "injury_risk_comfort",
    "Individual brilliance can outweigh teamwork in this sport.": "individual_brilliance",
    "Where might you play this sport?": "play_location",
    "During which seasons would you be most likely to play this sport recreationally in Toronto?": "play_season",
    "What types of music do you most associate with this sport?": "associated_music",
    "What foods and drinks do you associate with watching this sport in the place or culture you know best?": "associated_foods_drinks",
    "What colour do you associate with this sport?": "associated_color",
    "Which song do you associate with this sport?": "associated_song",
    "How do you feel when you watch this sport?": "emotional_response",
    "How much would you be willing to pay for a decent seat at a professional competition of this sport?": "acceptable_price",
    "label": "target_sport",
}

LIKERT_MAP = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

PLAY_SEASONS = ["Spring", "Summer", "Fall", "Winter", "None"]
PLAY_LOCATIONS = [
    "Outdoor public space",
    "School or university facility",
    "Private sports facility",
    "Community recreation facility",
    "Indoor facility",
    "At home or nearby",
    "Others",
    "None"
]
ASSOCIATED_MUSICS = [
    "Hip-hop or rap",
    "Pop",
    "Rock",
    "Electronic or dance",
    "No particular type",
    "Latin",
    "Country",
    "Folk or traditional",
    "Classical or jazz",
    "Other",
    "None"
]


def parse_ticket_price(val):
    """
    Parses unstructured price entries into clean floating-point values.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ["", "none", "n/a", "na", "null", "-", "no"]:
        return np.nan
    if "free" in s or s == "0":
        return 0.0
    s = re.sub(r"(\d+)\s*(?:thousand|k)\b", r"\g<1>000", s)
    range_match = re.findall(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", s)
    if range_match:
        return (float(range_match[0][0]) + float(range_match[0][1])) / 2.0
    num_match = re.findall(r"(\d+(?:,\d{3})*(?:\.\d+)?)", s)
    if num_match:
        return float(num_match[0].replace(",", ""))
    return np.nan


def load_model_bundle():
    """Loads the precomputed model weights and preprocessing parameters."""
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def transform_features(df_raw, bundle):
    """
    Preprocesses test data and extracts the exact 133 features required by the model.
    """
    df = df_raw.copy()
    for old_col, new_col in RENAME_MAP.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})

    features = {}

    # 1. Likert scale questions (1-5, missing filled with neutral 3)
    likert_cols = ["strategy_importance", "injury_risk_comfort", "individual_brilliance"]
    for col in likert_cols:
        if col in df.columns:
            features[f"{col}_num"] = df[col].map(LIKERT_MAP).fillna(3.0).values
        else:
            features[f"{col}_num"] = np.full(len(df), 3.0)

    # 2. Multi-select questions: binary indicator features
    def extract_dummies(series, categories):
        s_clean = series.fillna("None").astype(str)
        return {cat: s_clean.str.contains(re.escape(cat), regex=True).astype(int).values for cat in categories}

    season_dummies = extract_dummies(df.get("play_season", pd.Series(index=df.index)), PLAY_SEASONS)
    for cat, vals in season_dummies.items():
        features[f"play_season_{cat}"] = vals

    location_dummies = extract_dummies(df.get("play_location", pd.Series(index=df.index)), PLAY_LOCATIONS)
    for cat, vals in location_dummies.items():
        features[f"play_location_{cat}"] = vals

    music_dummies = extract_dummies(df.get("associated_music", pd.Series(index=df.index)), ASSOCIATED_MUSICS)
    for cat, vals in music_dummies.items():
        features[f"associated_music_{cat}"] = vals

    # 3. Numeric price standardization
    scaler = bundle["scaler"]
    if "acceptable_price" in df.columns:
        parsed_price = df["acceptable_price"].apply(parse_ticket_price).fillna(scaler["price_median"])
    else:
        parsed_price = pd.Series(scaler["price_median"], index=df.index)
    log_price = np.log1p(parsed_price)
    features["log_price_std"] = ((log_price - scaler["log_price_mean"]) / (scaler["log_price_std"] + 1e-8)).values

    # 4. Difficulty rating standardization
    if "difficulty_rating" in df.columns:
        parsed_diff = pd.to_numeric(df["difficulty_rating"], errors="coerce").fillna(scaler["difficulty_mean"])
    else:
        parsed_diff = pd.Series(scaler["difficulty_mean"], index=df.index)
    features["difficulty_zscore"] = ((parsed_diff - scaler["difficulty_mean"]) / (scaler["difficulty_std"] + 1e-8)).values

    # 5. Free-text TF-IDF vectorization
    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    tfidf_bundle = bundle["tfidf"]

    for col, info in tfidf_bundle.items():
        raw_text = df[col] if col in df.columns else pd.Series([""] * len(df), index=df.index)
        features[f"{col}_is_missing"] = raw_text.isna().astype(int).values

        clean_text = raw_text.fillna("missing").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()
        vocab = info["vocab"]
        idf = np.array(info["idf"])
        sorted_words = [w for w, idx in sorted(vocab.items(), key=lambda x: x[1])]

        col_vectors = []
        for doc in clean_text:
            tokens = token_pattern.findall(doc)
            tf = np.array([tokens.count(w) for w in sorted_words], dtype=float)
            tfidf = tf * idf
            norm = np.linalg.norm(tfidf)
            if norm > 0:
                tfidf = tfidf / norm
            col_vectors.append(tfidf)

        col_matrix = np.array(col_vectors)
        for i, w in enumerate(sorted_words):
            features[f"{col}_tfidf_{w}"] = col_matrix[:, i]

    # Assemble and align with required feature columns
    feat_df = pd.DataFrame(features, index=df.index)
    feature_cols = bundle["feature_cols"]
    X = feat_df.reindex(columns=feature_cols).values

    # Impute missing entries with training medians
    imputer_medians = np.array(bundle["imputer_medians"])
    nan_mask = np.isnan(X)
    if np.any(nan_mask):
        for j in range(X.shape[1]):
            col_mask = nan_mask[:, j]
            if np.any(col_mask):
                X[col_mask, j] = imputer_medians[j]

    return X

def predict_tree(tree, X, n_classes):
    """
    Predicts class labels for input features X using a single decision tree.
    """
    n_samples = X.shape[0]
    predictions = np.zeros((n_samples, n_classes), dtype=int)

    for i in range(n_samples):
        node = 0
        while True:
            left_child = tree["children_left"][node]
            right_child = tree["children_right"][node]
            if left_child == -1 and right_child == -1:
                # Leaf node
                value = np.array(tree["value"][node][0], dtype=float)
                predictions[i] = value / value.sum()
                break
            else:
                feature_idx = tree["feature"][node]
                threshold = tree["threshold"][node]
                if X[i, feature_idx] <= threshold:
                    node = left_child
                else:
                    node = right_child

    return predictions


def predict_all(filename):
    """
    Main prediction entry point required by MarkUs.
    Reads an unseen test CSV file and returns predicted sport labels as a list of strings.
    """
    df = pd.read_csv(filename)
    bundle = load_model_bundle()

    X = transform_features(df, bundle)
    forest = np.array(bundle["forest"])      # Shape: (3, 133)
    classes = bundle["classes"]          # ['Soccer', 'Baseball', 'Basketball']

    n_samples = X.shape[0]
    n_classes = len(classes)
    forest_probs = np.zeros((n_samples, n_classes))

    for tree in forest:
        forest_probs += predict_tree(tree, X, n_classes)

    forest_probs /= len(forest)

    # Multinomial Logistic Regression score: z = X W^T + b
    # logits = np.dot(X, W.T) + b
    # pred_indices = np.argmax(logits, axis=1)

    # predictions = [classes[idx] for idx in pred_indices]
    return classes[np.argmax(forest_probs, axis=1)]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        preds = predict_all(test_file)
        print(f"Total predictions generated: {len(preds)}")
        print("First 10 predictions:", preds[:10])
    else:
        print("Usage: python pred.py <path_to_test_csv>")

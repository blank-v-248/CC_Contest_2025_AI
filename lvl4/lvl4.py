import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, f1_score

# Load the dataset
df0 = pd.read_csv("lvl4/src/level_4.in")  # Replace with actual filename
df1 = pd.read_csv("lvl3/src/all_data_from_level_1.in")

is_fahrenheit = (df1['Temperature [°C]'] > 60)
df1.loc[is_fahrenheit, 'Temperature [°C]'] = (df1.loc[is_fahrenheit, 'Temperature [°C]'] - 32) * 5 / 9

def average_temperature(df_paths, df_bop):
    """
    For each path in df_paths, compute the average temperature from df_bop
    and add it as a new column 'Avg_temp'.

    df_paths: DataFrame with columns ['Flock ID', 'BOP Path']
    df_bop: DataFrame with columns ['BOP', 'Temperature [°C]', ...]

    Returns: df_paths with a new column 'Avg_temp'
    """
    # Convert BOP to temperature mapping for fast lookup
    temp_dict = df_bop.set_index('BOP')['Temperature [°C]'].to_dict()

    # Function to compute average temp for a single path
    def compute_avg_temp(path_str):
        path_bops = list(map(int, path_str.split()))
        temps = [temp_dict[bop] for bop in path_bops if bop in temp_dict]
        return sum(temps) / len(temps) if temps else None

    # Add new column to df_paths
    df_paths = df_paths.copy()  # avoid modifying original df
    df_paths['Avg_temp'] = df_paths['BOP Path'].apply(compute_avg_temp)

    return df_paths

df = average_temperature(df0, df1)

##### II. FEATURE ENGINEERING BASED ON TASK
# Convert path string to list of integers
df["Path_List"] = df["BOP Path"].apply(lambda x: list(map(int, x.split())))

# Group by Flock ID
grouped = df.groupby("Flock ID")["Path_List"].apply(list)

# Helper: Flatten list of lists
def flatten(paths):
    return [node for path in paths for node in path]

# Feature 1: Symmetricity score
def symmetricity(path):
    score = sum(1 for i in range(len(path)//2) if path[i] == path[-(i+1)])
    return score / (len(path)//2)

# Feature 2: Path length
def path_length(path):
    return len(path)

# Feature 3: Self-similarity (nodes repeated > 3 times)
def self_similarity(path):
    counts = Counter(path)
    return sum(1 for v in counts.values() if v > 3)

# Feature 4: Similarity to flock (exact position match)
def similarity_to_flock(path, flock_paths):
    if not flock_paths:
        return 0
    avg_path = np.mean([np.array(p[:len(path)]) for p in flock_paths if len(p) >= len(path)], axis=0)
    return np.mean([1 if path[i] == int(avg_path[i]) else 0 for i in range(len(path))])

# Feature 5: Similarity to flock (first half only)
def similarity_to_flock_half(path, flock_paths):
    half = len(path) // 2
    return similarity_to_flock(path[:half], [p[:half] for p in flock_paths if len(p) >= half])

# Apply features
feature_rows = []
for idx, row in df.iterrows():
    flock_id = row["Flock ID"]
    path = row["Path_List"]
    flock_paths = [p for p in grouped[flock_id] if p != path]  # exclude self

    features = {
        "Flock ID": flock_id,
        "Symmetricity": symmetricity(path),
        "Path Length": path_length(path),
        "Self Similarity": self_similarity(path),
        "Similarity to Flock": similarity_to_flock(path, flock_paths),
        "Similarity to Flock Half": similarity_to_flock_half(path, flock_paths),
        "Avg_temp": row["Avg_temp"],
        "Species": row["Species"]
    }
    feature_rows.append(features)

features_df = pd.DataFrame(feature_rows)

##### III. MODEL TRAINING ON FEATURES
# Step 1: Drop rows with missing species
train_df = features_df[features_df["Species"]!="missing"].copy()

# Step 2: Encode species labels
le = LabelEncoder()
train_df["Species_encoded"] = le.fit_transform(train_df["Species"])

# Step 3: Define features and target
feature_cols = [
    "Symmetricity",
    "Path Length",
    "Self Similarity",
    "Similarity to Flock",
    "Similarity to Flock Half",
    "Avg_temp"
]
X = train_df[feature_cols]
y = train_df["Species_encoded"]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf.predict(X_test)
f1_macro = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Step 7: Output results
print("📊 Classification Report:\n")
print(report)
print(f"✅ F1 Macro Score on test set: {f1_macro:.4f}")


# Step 8: Predict missing species
test_df = features_df[features_df["Species"]=="missing"].copy()
X_test = test_df[feature_cols]
preds = clf.predict(X_test)
pred_species = le.inverse_transform(preds)

# Step 9: Prepare submission
submission = test_df[["Flock ID"]].copy()
submission["Species"] = pred_species
submission.to_csv("lvl4_output.csv", index=False)
submission.to_csv("lvl4_output.out", index=False)

print("Submission file created yayy :)")

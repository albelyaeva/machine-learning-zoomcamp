import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

url = '/Users/aleksandra/PycharmProjects/machine-learning-zoomcamp/course_lead_scoring.csv'
df = pd.read_csv(url)

print("Missing values before:")
print(df.isnull().sum())

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if 'converted' in numerical_cols:
    numerical_cols.remove('converted')

print(f"\nCategorical: {categorical_cols}")
print(f"Numerical: {numerical_cols}")

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        print(f"Filling {col} with 'NA'")
        df[col] = df[col].fillna('NA')

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        print(f"Filling {col} with 0.0")
        df[col] = df[col].fillna(0.0)

print("\nMissing values after:")
print(df.isnull().sum())

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train['converted'].values
y_val = df_val['converted'].values
y_test = df_test['converted'].values
y_full_train = df_full_train['converted'].values

features_to_test = ['lead_score', 'number_of_courses_viewed', 'interaction_count', 'annual_income']

auc_results = {}

for feature in features_to_test:
    if feature not in df_train.columns:
        print(f"  {feature}: NOT FOUND IN DATA!")
        continue

    scores = df_train[feature].values
    auc = roc_auc_score(y_train, scores)

    if auc < 0.5:
        auc_inverted = roc_auc_score(y_train, -scores)
        print(f"  {feature}: {auc:.4f} -> {auc_inverted:.4f} (INVERTED)")
        auc_results[feature] = auc_inverted
    else:
        print(f"  {feature}: {auc:.4f}")
        auc_results[feature] = auc

best_feature = max(auc_results, key=auc_results.get)
print(f"\n✓ Q1 ANSWER: {best_feature} (AUC: {auc_results[best_feature]:.4f})")

df_train_X = df_train.drop('converted', axis=1)
df_val_X = df_val.drop('converted', axis=1)

print(f"Features shape: {df_train_X.shape}")
print(f"Sample row:")
print(df_train_X.iloc[0].to_dict())

train_dicts = df_train_X.to_dict(orient='records')
val_dicts = df_val_X.to_dict(orient='records')

print(f"\nConverted to {len(train_dicts)} training dictionaries")

# Apply DictVectorizer
print("\nApplying DictVectorizer...")
dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

print(f"  X_train shape: {X_train.shape}")
print(f"  X_val shape: {X_val.shape}")
print(f"  Number of features after encoding: {len(dv.feature_names_)}")

# Train model
print("\nTraining LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)")
model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
model.fit(X_train, y_train)
print("  Model trained!")

# Predict on validation set
print("\nPredicting on validation set...")
y_pred_proba = model.predict_proba(X_val)[:, 1]
print(f"  Prediction shape: {y_pred_proba.shape}")
print(f"  Sample predictions: {y_pred_proba[:5]}")

# Calculate AUC
auc_score = roc_auc_score(y_val, y_pred_proba)
print(f"\n  AUC on validation: {auc_score:.6f}")
print(f"  Rounded to 3 digits: {auc_score:.3f}")
print(f"  Rounded to 2 digits: {auc_score:.2f}")

# Find closest option
options_q2 = [0.32, 0.52, 0.72, 0.92]
closest_q2 = min(options_q2, key=lambda x: abs(x - auc_score))
print(f"\n✓ Q2 ANSWER: {closest_q2} (actual: {auc_score:.3f})")

print("\n" + "=" * 60)
print("QUESTION 3: PRECISION-RECALL INTERSECTION")
print("=" * 60)

thresholds = np.arange(0.0, 1.01, 0.01)
precisions = []
recalls = []

print("Computing precision and recall for all thresholds...")
for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_val == 1)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()
    tn = ((y_pred == 0) & (y_val == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

# Find intersection
differences = [abs(p - r) for p, r in zip(precisions, recalls)]
min_diff_idx = np.argmin(differences)
intersection_threshold = thresholds[min_diff_idx]

print(f"  Intersection at threshold: {intersection_threshold:.3f}")
print(f"  Precision: {precisions[min_diff_idx]:.3f}")
print(f"  Recall: {recalls[min_diff_idx]:.3f}")
print(f"  Difference: {differences[min_diff_idx]:.6f}")

options_q3 = [0.145, 0.345, 0.545, 0.745]
closest_q3 = min(options_q3, key=lambda x: abs(x - intersection_threshold))
print(f"\n✓ Q3 ANSWER: {closest_q3} (actual: {intersection_threshold:.3f})")

print("\n" + "=" * 60)
print("QUESTION 4: MAXIMUM F1 SCORE")
print("=" * 60)

f1_scores = []

print("Computing F1 score for all thresholds...")
for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_val == 1)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    fn = ((y_pred == 0) & (y_val == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_scores.append(f1)

max_f1_idx = np.argmax(f1_scores)
max_f1_threshold = thresholds[max_f1_idx]
max_f1_value = f1_scores[max_f1_idx]

options_q4 = [0.14, 0.34, 0.54, 0.74]
closest_q4 = min(options_q4, key=lambda x: abs(x - max_f1_threshold))
print(f"\n✓ Q4 ANSWER: {closest_q4} (actual: {max_f1_threshold:.2f})")

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = []

for fold_num, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), 1):
    # Get fold data
    df_fold_train = df_full_train.iloc[train_idx]
    df_fold_val = df_full_train.iloc[val_idx]

    # Extract targets
    y_fold_train = df_fold_train['converted'].values
    y_fold_val = df_fold_val['converted'].values

    # Remove target from features
    X_fold_train = df_fold_train.drop('converted', axis=1)
    X_fold_val = df_fold_val.drop('converted', axis=1)

    # Convert to dicts
    train_dicts_fold = X_fold_train.to_dict(orient='records')
    val_dicts_fold = X_fold_val.to_dict(orient='records')

    # Vectorize
    dv_fold = DictVectorizer(sparse=False)
    X_fold_train_enc = dv_fold.fit_transform(train_dicts_fold)
    X_fold_val_enc = dv_fold.transform(val_dicts_fold)

    # Train model
    model_fold = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=1)
    model_fold.fit(X_fold_train_enc, y_fold_train)

    # Predict and score
    y_fold_pred = model_fold.predict_proba(X_fold_val_enc)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
    cv_scores.append(fold_auc)

    print(f"  Fold {fold_num}: AUC = {fold_auc:.6f}")

mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)

options_q5 = [0.0001, 0.006, 0.06, 0.36]
closest_q5 = min(options_q5, key=lambda x: abs(x - std_cv))
print(f"\n✓ Q5 ANSWER: {closest_q5} (actual: {std_cv:.3f})")

C_values = [0.000001, 0.001, 1]
results_q6 = {}

for C in C_values:
    print(f"\nTesting C={C}:")
    cv_scores_c = []

    for fold_num, (train_idx, val_idx) in enumerate(kfold.split(df_full_train), 1):
        df_fold_train = df_full_train.iloc[train_idx]
        df_fold_val = df_full_train.iloc[val_idx]

        y_fold_train = df_fold_train['converted'].values
        y_fold_val = df_fold_val['converted'].values

        X_fold_train = df_fold_train.drop('converted', axis=1)
        X_fold_val = df_fold_val.drop('converted', axis=1)

        train_dicts_fold = X_fold_train.to_dict(orient='records')
        val_dicts_fold = X_fold_val.to_dict(orient='records')

        dv_fold = DictVectorizer(sparse=False)
        X_fold_train_enc = dv_fold.fit_transform(train_dicts_fold)
        X_fold_val_enc = dv_fold.transform(val_dicts_fold)

        model_fold = LogisticRegression(solver='liblinear', C=C, max_iter=1000, random_state=1)
        model_fold.fit(X_fold_train_enc, y_fold_train)

        y_fold_pred = model_fold.predict_proba(X_fold_val_enc)[:, 1]
        fold_auc = roc_auc_score(y_fold_val, y_fold_pred)
        cv_scores_c.append(fold_auc)

    mean_c = np.mean(cv_scores_c)
    std_c = np.std(cv_scores_c)
    results_q6[C] = {'mean': mean_c, 'std': std_c}

    print(f"  Mean: {mean_c:.3f}, Std: {std_c:.3f}")

for C in C_values:
    print(f"  C={C}: {results_q6[C]['mean']:.3f} ± {results_q6[C]['std']:.3f}")

best_C = max(C_values, key=lambda c: (results_q6[c]['mean'], -results_q6[c]['std']))
print(f"\n✓ Q6 ANSWER: {best_C}")

print("\n" + "=" * 60)
print("FINAL ANSWERS SUMMARY")
print("=" * 60)
print(f"Q1: {best_feature}")
print(f"Q2: {closest_q2}")
print(f"Q3: {closest_q3}")
print(f"Q4: {closest_q4}")
print(f"Q5: {closest_q5}")
print(f"Q6: {best_C}")
print("=" * 60)
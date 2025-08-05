import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load and prepare data 
df = pd.read_csv("output_csvs/scut_mean02/cycle_features_01_10.csv")

# Ensure 'label' column exists and contains expected values
if 'label' not in df.columns:
    print("Error: 'label' column not found in the CSV.")
    exit()

# Filter out any unexpected labels if they exist, or handle them
# Assuming labels are 0 (Fake), 1 (Real), 2 (Synthetic)
df = df[df['label'].isin([0, 1, 2])]

class_counts = df["label"].value_counts()

# Get the minimum class size for balancing Real and Fake
# Note: Balancing is applied to the initial Real and Fake dataset (X, y)
# Synthetic samples are handled separately as per the steps.
min_count_real_fake = min(class_counts.get(0, 0), class_counts.get(1, 0))

if min_count_real_fake == 0:
    print("Error: Not enough Real or Fake samples to balance.")
    exit()

# Sample min_count_real_fake rows for Real and Fake classes
balanced_real = df[df["label"] == 1].sample(n=min_count_real_fake, random_state=42)
balanced_fake = df[df["label"] == 0].sample(n=min_count_real_fake, random_state=42)

# Concatenate and shuffle for the initial Real+Fake dataset
df_balanced_real_fake = pd.concat([balanced_real, balanced_fake]).sample(frac=1, random_state=42).reset_index(drop=True)

# Separate synthetic images (all of them, as they are added later)
synthetic_images = df[df["label"] == 2].reset_index(drop=True)

real_feat = df_balanced_real_fake[df_balanced_real_fake["label"] == 1].drop("label", axis=1).values
real_label = df_balanced_real_fake[df_balanced_real_fake["label"] == 1]["label"].values

fake_feat = df_balanced_real_fake[df_balanced_real_fake["label"] == 0].drop("label", axis=1).values
fake_label = df_balanced_real_fake[df_balanced_real_fake["label"] == 0]["label"].values

synthetic_feat = synthetic_images.drop("label", axis=1).values
synthetic_label = synthetic_images["label"].values # Keep original label 2 for now, will re-label to 0 when used


# Combine real and fake features
X = np.vstack((real_feat, fake_feat))
y = np.hstack((real_label, fake_label))

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)

# Apply same scaling to synthetic features
synthetic_feat = scaler.transform(synthetic_feat)
synthetic_feat = selector.transform(synthetic_feat)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Initialize model
model = KNeighborsClassifier(n_neighbors=7)

# Store results
results = {step: [] for step in range(6)}
apcer_results = {step: [] for step in range(6)}
bpcer_results = {step: [] for step in range(6)}
acer_results = {step: [] for step in range(6)}

# Cross-validation loop
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n=== Fold {fold + 1} ===")

    # Test data (always Real and Fake from the original balanced dataset)
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Test samples: {len(X_test)} (Real: {np.sum(y_test==1)}, Fake: {np.sum(y_test==0)})")

    # Helper function to calculate metrics
    def calculate_metrics(y_true, y_pred):
        # Ensure confusion matrix is calculated for binary classes 0 and 1
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        apcer = fp / (fp + tn) if (fp + tn) > 0 else 0
        bpcer = fn / (fn + tp) if (fn + tp) > 0 else 0
        acer = (apcer + bpcer) / 2
        return accuracy_score(y_true, y_pred), apcer, bpcer, acer

    # === Step 0 === Full training on 4/5 real + fake
    X_train_0, y_train_0 = X[train_idx], y[train_idx]
    print(f"Step 0 Train samples: {len(X_train_0)}")
    model.fit(X_train_0, y_train_0)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[0].append(acc)
    apcer_results[0].append(apcer)
    bpcer_results[0].append(bpcer)
    acer_results[0].append(acer)

    # === Step 1 === Train on 2/5 of real + fake
    # Randomly select half of the training indices from Step 0
    reduced_train_idx = np.random.choice(train_idx, size=len(train_idx) // 2, replace=False)
    X_train_1, y_train_1 = X[reduced_train_idx], y[reduced_train_idx]
    print(f"Step 1 Train samples: {len(X_train_1)}")
    model.fit(X_train_1, y_train_1)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[1].append(acc)
    apcer_results[1].append(apcer)
    bpcer_results[1].append(bpcer)
    acer_results[1].append(acer)

    # === Step 2 === Train on 1/5 of real + fake
    # Randomly select half of the training indices from Step 1
    rng = np.random.default_rng(seed=fold)  # or fixed seed per fold
    further_reduced_idx = rng.choice(reduced_train_idx, size=len(reduced_train_idx) // 2, replace=False)
    X_train_2, y_train_2 = X[further_reduced_idx], y[further_reduced_idx]
    print(f"Step 2 Train samples: {len(X_train_2)}")
    model.fit(X_train_2, y_train_2)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[2].append(acc)
    apcer_results[2].append(apcer)
    bpcer_results[2].append(bpcer)
    acer_results[2].append(acer)

    # === Step 3 === Train on 1/5 real+fake + additional real samples + synthetic samples
    # Description: "double the size of the training set: by adding the Real samples that have been removed
    # to arrive at Step 2 (from Step 1) and by adding the Synthetic samples, the corresponding Fake samples
    # of which have been removed to arrive at Step 2 (from Step 1).
    # So the size of the training set is the same as in Step 1, but it is differently composed."

    # Samples removed from Step 1 to Step 2
    removed_from_step1_to_step2_idx = np.setdiff1d(reduced_train_idx, further_reduced_idx)

    # Separate real and fake from these removed samples
    removed_real_idx = [i for i in removed_from_step1_to_step2_idx if y[i] == 1]
    removed_fake_idx = [i for i in removed_from_step1_to_step2_idx if y[i] == 0]

    # Number of synthetic samples to add (equal to the number of fake samples removed)
    num_synthetic_to_add = len(removed_fake_idx)

    # Randomly sample synthetic features and label them as 0 (Fake)
    if len(synthetic_feat) < num_synthetic_to_add:
        print(f"Warning: Not enough synthetic samples for Step 3. Required: {num_synthetic_to_add}, Available: {len(synthetic_feat)}")
        # Handle this case, e.g., by taking all available synthetic samples
        sampled_synthetic_feat = synthetic_feat
        sampled_synthetic_label = np.zeros(len(synthetic_feat))
    else:
        synthetic_indices = rng.choice(len(synthetic_feat), size=num_synthetic_to_add, replace=False)
        sampled_synthetic_feat = synthetic_feat[synthetic_indices]
        sampled_synthetic_label = np.zeros(num_synthetic_to_add) # Label as Fake (0)

    # Combine:
    # 1. X_train_2 (1/5 real+fake)
    # 2. Real samples that were removed from Step 1 to Step 2
    # 3. Synthetic samples (replacing fake samples removed from Step 1 to Step 2)
    X_train_3 = np.vstack((X_train_2, X[removed_real_idx], sampled_synthetic_feat))
    y_train_3 = np.hstack((y_train_2, y[removed_real_idx], sampled_synthetic_label))

    print(f"Step 3 Train samples: {len(X_train_3)} (Target size: {len(X_train_1)})")
    model.fit(X_train_3, y_train_3)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[3].append(acc)
    apcer_results[3].append(apcer)
    bpcer_results[3].append(bpcer)
    acer_results[3].append(acer)

    # === Step 4 === Keep 1/5 real+fake + add remaining real + synthetic
    # Description: "We repeat the same 5 folds to obtain all results, we keep again 1/5 of the real data
    # (Real samples and Fake samples) as the training set for each fold. But now, we have the remaining
    # three folds composed of Real samples and the corresponding Synthetic samples as training data.
    # The size of the training data is the same as in Step 5, but it is differently composed."

    # Remaining samples from the original training fold (train_idx) after taking X_train_2
    remaining_original_train_idx = np.setdiff1d(train_idx, further_reduced_idx)

    # Separate real and fake from these remaining samples
    remaining_real_original_idx = [i for i in remaining_original_train_idx if y[i] == 1]
    remaining_fake_original_idx = [i for i in remaining_original_train_idx if y[i] == 0]

    # Number of synthetic samples to add (equal to the number of remaining fake samples)
    num_synthetic_for_remaining = len(remaining_fake_original_idx)

    # Randomly sample synthetic features for these remaining fake samples
    if len(synthetic_feat) < num_synthetic_for_remaining:
        print(f"Warning: Not enough synthetic samples for Step 4. Required: {num_synthetic_for_remaining}, Available: {len(synthetic_feat)}")
        sampled_synthetic_for_remaining_feat = synthetic_feat
        sampled_synthetic_for_remaining_label = np.zeros(len(synthetic_feat))
    else:
        synthetic_indices_4 = rng.choice(len(synthetic_feat), size=num_synthetic_for_remaining, replace=False)
        sampled_synthetic_for_remaining_feat = synthetic_feat[synthetic_indices_4]
        sampled_synthetic_for_remaining_label = np.zeros(num_synthetic_for_remaining) # Label as Fake (0)

    # Combine:
    # 1. X_train_2 (1/5 real+fake)
    # 2. Remaining real samples from original training fold
    # 3. Synthetic samples (replacing remaining fake samples from original training fold)
    X_train_4 = np.vstack((X_train_2, X[remaining_real_original_idx], sampled_synthetic_for_remaining_feat))
    y_train_4 = np.hstack((y_train_2, y[remaining_real_original_idx], sampled_synthetic_for_remaining_label))

    print(f"Step 4 Train samples: {len(X_train_4)}")
    model.fit(X_train_4, y_train_4)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[4].append(acc)
    apcer_results[4].append(apcer)
    bpcer_results[4].append(bpcer)
    acer_results[4].append(acer)

    # === Step 5 === Train on only real + synthetic (no fake)
    # Description: "training set for each fold consists of all Real samples from the other 4 folds
    # plus the corresponding Synthetic samples from those 4 folds."

    # Get all real samples from the current training fold
    # FIX: Corrected to select real samples (label 1)
    all_real_in_train_idx = [i for i in train_idx if y[i] == 1]
    X_real_step5 = X[all_real_in_train_idx]
    y_real_step5 = y[all_real_in_train_idx] # These are all label 1

    # Get the count of fake samples that *would have been* in this training fold
    # These are the ones we are replacing with synthetic samples
    num_fake_in_train_fold = np.sum(y[train_idx] == 0)

    # Randomly sample synthetic features to match the number of fake samples
    if len(synthetic_feat) < num_fake_in_train_fold:
        print(f"Warning: Not enough synthetic samples for Step 5. Required: {num_fake_in_train_fold}, Available: {len(synthetic_feat)}")
        X_synth_step5 = synthetic_feat
        y_synth_step5 = np.zeros(len(synthetic_feat)) # FIX: Label as Fake (0)
    else:
        synthetic_indices_5 = rng.choice(len(synthetic_feat), size=num_fake_in_train_fold, replace=False)
        X_synth_step5 = synthetic_feat[synthetic_indices_5]
        y_synth_step5 = np.zeros(num_fake_in_train_fold) # FIX: Label as Fake (0)

    # Combine real + synthetic
    X_train_5 = np.vstack((X_real_step5, X_synth_step5))
    y_train_5 = np.hstack((y_real_step5, y_synth_step5))
    print(f"Step 5 Train samples: {len(X_train_5)}")
    model.fit(X_train_5, y_train_5)
    acc, apcer, bpcer, acer = calculate_metrics(y_test, model.predict(X_test))
    results[5].append(acc)
    apcer_results[5].append(apcer)
    bpcer_results[5].append(bpcer)
    acer_results[5].append(acer)

# === Print summary ===
print("\n=== Summary of Results ===")
for step in range(6):
    print(
        f"Step {step} Accuracy: {np.mean(results[step]):.4f} | "
        f"APCER: {np.mean(apcer_results[step]):.4f} "
        f"BPCER: {np.mean(bpcer_results[step]):.4f} "
        f"ACER: {np.mean(acer_results[step]):.4f}"
    )

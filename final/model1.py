import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from skimage.transform import resize
from skimage.feature import hog
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)
import joblib

SEED = 67


# the data folder is expected in the same directory as this script
ROOT_PATH = Path(__file__).resolve().parent
DATA = ROOT_PATH / "data"
#we cache the HOG features in this directory so that they dont get recomputed every time
FEAT = ROOT_PATH / "feat"
MODELE = ROOT_PATH / "models"
RES = ROOT_PATH / "res"
#creating the output folders if they are missing
for folder in (FEAT, MODELE, RES):
    folder.mkdir(exist_ok=True)
print("project root:", ROOT_PATH)


RESIZE = (128, 64)  

def load_gray(path):
    #converting the RGBA image to a single grayscale channel since viridis is a 1-D colormap,
    #so luminance keeps the signal magnitude and reduces 4 channels to 1 (4x fewer input numbers)
    img = Image.open(path).convert("L")
    #scaling the pixels from 0-255 to 0-1 and resizing every image to one fixed shape
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return resize(arr, RESIZE, anti_aliasing=True)

def hog_feature(gray):
    #HOG describes the distribution of gradient orientations
    #it captures the shape/orientation of the energy ridges in the spectrogram
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),block_norm="L2-Hys", feature_vector=True)

def build_feature_matrix(df, split):
    #running HOG on every image of the dataframe, keeping the rows in the same order as the csv
    feats = []
    for file_name in df["id"]:
        gray = load_gray(DATA / split / file_name)
        feats.append(hog_feature(gray))
    return np.asarray(feats, dtype=np.float32)

#train has id+label andtest has only id
df_antrenare = pd.read_csv(DATA / "train.csv")
df_test = pd.read_csv(DATA / "test.csv")

#extracting HOG once and caching to disk
#we save a couple minutes on reloading the script 
if (FEAT / "Xtr_hog.npy").exists():
    imagini = np.load(FEAT / "Xtr_hog.npy")
    img_test = np.load(FEAT / "Xte_hog.npy")
else:
    imagini = build_feature_matrix(df_antrenare, "train")
    img_test = build_feature_matrix(df_test, "test")
    np.save(FEAT / "Xtr_hog.npy", imagini)
    np.save(FEAT / "Xte_hog.npy", img_test)

#the SVM works with the original 1-5 labels directly
etichete = df_antrenare["label"].to_numpy()
print("HOG features:", imagini.shape, "| test:", img_test.shape)



#carving a stratified validation set out of train so we can measure accuracy honestly 
#stratify keeps the class proportions equal between the two splits
img_antrenare, img_validare, et_antrenare, et_validare = train_test_split(imagini, etichete, test_size=0.2, stratify=etichete, random_state=SEED)

#the RBF SVM is distance-based, so the features must be standardized with a 0 mean and 1 variance
#we fit the scaler on train only and apply it to validation to avoid leakage
scaler = StandardScaler().fit(img_antrenare)
Xtr_s, Xval_s = scaler.transform(img_antrenare), scaler.transform(img_validare)

#PCA compresses the HOG dimensions down which maeks a much faster SVM without losing much signal
N_PCA = 150
pca = PCA(n_components=N_PCA, random_state=SEED).fit(Xtr_s)
Xtr_p, Xval_p = pca.transform(Xtr_s), pca.transform(Xval_s)
print(f"PCA: {Xtr_s.shape[1]} -> {Xtr_p.shape[1]} dims ({pca.explained_variance_ratio_.sum():.3f} variance)")

#the RBF SVM is ~O(n^2), so tuning on all 12,400 samples is slow; we tune on a stratified 4000-sample subset
sub_idx, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=4000, random_state=SEED).split(Xtr_p, et_antrenare))
Xtune_p, ettune = Xtr_p[sub_idx], et_antrenare[sub_idx]
print("tuning subset:", Xtune_p.shape)



def build_classifier(C=1, gamma="scale"):
    #we use an RBF SVM in which C controls the regularization strength, gamma the reach of each support vector
    return SVC(kernel="rbf", C=C, gamma=gamma, random_state=SEED)

def write_submission(preds, test_ids, output_path):
    #the SVM predicts the original 1-5 labels directly
    submission = pd.DataFrame({"id": test_ids.to_numpy(), "label": preds})
    submission.to_csv(output_path, index=False)
    print(f"Wrote {output_path.name} | Label range: {preds.min()} - {preds.max()}")

def evaluate_model(model, img_validare, et_validare, results_dir):
    #predicting on the held-out validation set and reporting accuracy + the required confusion matrix
    pred = model.predict(img_validare)
    val_acc = accuracy_score(et_validare, pred)
    print("val accuracy:", round(val_acc, 4))
    print(classification_report(et_validare, pred))

    cm = confusion_matrix(et_validare, pred)
    figure, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=[1, 2, 3, 4, 5]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("SVM Validation Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_svm.png", dpi=300, bbox_inches="tight")
    plt.show()

    pd.DataFrame([{"model": "svm", "val_accuracy": val_acc}]).to_csv(results_dir / "val_scores_model1.csv", index=False)
    return val_acc

#tuning C and gamma with 3-fold cross-validation on the subset 
param_grid = {"C": [1, 10, 100], "gamma": ["scale", 0.001, 0.01]}
svm_grid = GridSearchCV(build_classifier(), param_grid, cv=3, scoring="accuracy", n_jobs=-1)
svm_grid.fit(Xtune_p, ettune)
print("best params:", svm_grid.best_params_)

#saving the full grid of every C and gamma combination
pd.DataFrame(svm_grid.cv_results_)[["param_C", "param_gamma", "mean_test_score", "std_test_score"]] \
    .sort_values("mean_test_score", ascending=False).to_csv(RES / "hp_table_svm.csv", index=False)

#refitting the best configuration on the full training portion
svm_best = build_classifier(**svm_grid.best_params_).fit(Xtr_p, et_antrenare)
joblib.dump(svm_best, MODELE / "svm_best.joblib")

final_accuracy = evaluate_model(svm_best, Xval_p, et_validare, RES)

#refitting the scaler and PCA on ALL the training data so the final model uses every example
scaler_full = StandardScaler().fit(imagini)
Xall_s = scaler_full.transform(imagini)
pca_full = PCA(n_components=N_PCA, random_state=SEED).fit(Xall_s)
Xall_p = pca_full.transform(Xall_s)
svm_final = build_classifier(**svm_grid.best_params_).fit(Xall_p, etichete)

#transforming the test features through the same scaler + PCA then predicting and writing the submission
Xte_p = pca_full.transform(scaler_full.transform(img_test))
test_pred = svm_final.predict(Xte_p)
write_submission(test_pred, df_test["id"], RES / "submission_model1.csv")

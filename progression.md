# Model 1 — Classical baseline: HOG features + SVM (or k-NN)

This is your "feature engineering" model. Extract hand-crafted features from each image, feed them to a classical classifier.

* **Features to try:** Histogram of Oriented Gradients (HOG), raw pixel intensities, image statistics (mean, std, skewness, energy), and a 2D FFT magnitude histogram. For radio signals (often spectrogram-like images), FFT features and HOG tend to work surprisingly well.
* **Classifiers:** SVM with RBF kernel, k-NN, Random Forest. Try at least two so your report has comparisons.
* **Hyperparameters to tune (and report in a table):** SVM `C` and `gamma` via grid search; k-NN `k`; RF `n_estimators` and `max_depth`.
* **Tools:** `scikit-image` for HOG, `scikit-learn` for classifiers and `GridSearchCV`.

## Why HOG

     Objects are captured by the way the edges are and which way they point not by the pixel intensity, two pictures that have the same object in differnt lightings and differnet gradient variations have very diff pixel values but similar edge patterns.
    
     For analyzing the edges we need the way they point and their magnitude, meaning how strong the edge is based on how much the brightness changes horizontally and vertically.


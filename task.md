
# Task

Participants have to train a model for object detection in radio signal images. This is a multi-way classification task in which an image must be classified into one of five classes.

The training data is composed of 15,500 image files. The test is composed of 5,500 image files.

## File Descriptions

* **`train.csv`** - the training metadata file containing the training image file names and the corresponding labels (one example per row)
* **`test.csv`** - the test metadata file containing the test image file names (one sample per row)
* **`sample_submission.txt`** - a sample submission file in the correct format

## Data Format

### Metadata Files

The metadata files are provided in the following format based on comma separated values:

```csv
id,label
000001.png,1
000002.png,2
...

```

Each line represents an example where:

* The first column shows the image file name of the example.
* The second column is the label associated to the example.

### Image Files

The image files are provided in `.png` format.

## RULES

### Strict Rules & Disqualification Warnings

**Using additional data (from other sources), pre-defined models or pre-trained models is not allowed.**
*Failure to comply will result in disqualification.*

Your grade will be given only if you provide the code and a documentation for your proposed approach.

### Report Guidelines

The report should follow these guidelines:

* **Approach Description (Required):** The description of your machine learning approaches including the chosen feature set (color histograms, etc.), the tested models (K-NN, SVM, etc.). Details should also include data preprocessing / augmentation, feature representation and details / proof (via tables / figures) of hyperparameter tuning (learning rate, performance function, regularization, etc.). A minimum of 3 pages (excluding tables and figures) is expected. Documenting all the tried approaches (even unsuccessful ones) is a must.
* **Model Count:** Documenting and using at least two different machine learning models is required.
* **Confusion Matrices (Required):** Report confusion matrices for the provided validation set with various models. Use tables / figures to report these.
* **Hyperparameter Configurations (Required):** Present results with various hyperparameter configurations via tables / figures.

### Code & Submission Requirements

* **Code Quality:** The Python code of your model should include explanatory comments.
* **Reproducibility:** The Python code must be complete and fully functional (*failing to reproduce results will result in disqualification*).
* **File Format:** Competitors must convert `*.ipynb` files to `*.py` (*failure to comply will result in disqualification*).
* **Folder Structure:** The `*.py` files must be placed in the same folder named `'{family_name}_{first_name}_{group}'`. Subfolders with code are not allowed (*failure to comply will result in disqualification*).
* **Report Naming:** The report (in PDF format) must be named `'{family_name}_{first_name}_{group}_doc.PDF'`.
* **Final Archive:** The folder containing the `*.py` files and the documentation must be archived in a single `*.zip` file.

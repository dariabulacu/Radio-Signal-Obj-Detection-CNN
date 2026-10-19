import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from sklearn.model_selection import train_test_split

SEED = 67
keras.utils.set_random_seed(SEED)  

# the data folder is expected in the same directory as this script
ROOT_PATH = Path(__file__).resolve().parent
DATA = ROOT_PATH/"data"
MODELE = ROOT_PATH/"models"
RES = ROOT_PATH/"res"
assert DATA.is_dir(), f"The 'data' directory does not exist in {ROOT_PATH} (actual cwd: {Path.cwd()})"
#further on for saving weights in the folder 
MODELE.mkdir(exist_ok=True)
#further on for saving submissions and plot figures 
RES.mkdir(exist_ok=True)
print("project root:", ROOT_PATH)


#the images are sized (128,55) so the chosen format doesn't modify the frequency axis of the spectogram
#  in trade off to more inputs so slower training time  
IMG_WIDTH = 64
IMG_HEIGHT = 128   

def load_img(path):
    #converting the image to a single grayscale channel from four channels since the images are RGBA using the viridis colormap
    #that way we can reduce the computational power from 4->1 channels so that way the CNN doesnt have to learn from 4x more input numbers
    img = Image.open(path).convert('L').resize((IMG_WIDTH, IMG_HEIGHT))  

    #transforming the image into an array so we can apply mathematical solutions on it 
    # because of using GradientDescent we need to scale the input features from 0-255 to 0-1 
    return np.asarray(img, dtype=np.float32) / 255.0

#id and label
df_antrenare = pd.read_csv(DATA/'train.csv')
#id
df_test  = pd.read_csv(DATA/'test.csv')

#transforming the arrays into 4-D tensors so that the CNN can work with them 
#the images are stacked along a new axis and then add a new one for the chanel even if there s only one (grayscale)
X_list =[load_img(DATA/'train'/file_name) for file_name in df_antrenare['id']]
imagini = np.stack(X_list)[..., np.newaxis]

#same thing for the testing data set
X_test_list = [load_img(os.path.join(DATA, 'test', file_name)) for file_name in df_test['id']]
img_test = np.stack(X_test_list)[..., np.newaxis]

#a big debugging situation because the dense layer with 5 neurons maps the labels to 0-4, so we substract 1 so that 
# label 5 can get mapped correctly 
etichete = df_antrenare['label'].to_numpy() - 1
#we need a validation data set so we can reduce overfitting and validate the accuracy for the training set
#we respect the trade off between the two data sets and keep the proportions equal between classes 
img_antrenare, img_validare, et_antrenare, et_validare = train_test_split(imagini,etichete,test_size=0.15,stratify=etichete,random_state=SEED)


AUTOTUNE = tf.data.AUTOTUNE
BATCH = 64
#using the Data API from TF to convert the imgs into a data object, transform it
#and feed it to the model 

def build_dataset(Xa, ya, training):
    dataset = tf.data.Dataset.from_tensor_slices((Xa, ya))
    #only training imgs are augmented
    if training:
        #forming iid instances 
        dataset = dataset.shuffle(len(Xa), seed=SEED)
    dataset = dataset.batch(BATCH)
    #it prepares the next batch while training the current one on GPU 
    #it dynamically figures out how many batches it keeps on the counter so that GPU never has free time 
    return dataset.prefetch(AUTOTUNE)

set_antrenare = build_dataset(img_antrenare, et_antrenare, training=True)
set_validare = build_dataset(img_validare, et_validare, training=False)

#augmentation becomes the first layer of the model 
def make_augmentation():
#performing data regularization by shifting the values by offsets and expanding the training data set 
#for tolerance for more variations in positions
    return keras.Sequential([
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1),
    ], name='augment')

#building the models architecture using the Functional API so that we can also  have the deep learning path
# but also the short learning path (in the case of a Sequential API the short path would distort the output by
#passing it through all the layers)
#Functional API allows us to have a chain of functions as layers

#Setting implicit hyperparameters as opposed to what the models learns
def build_architecture(filters=(64,128,256), dense=128, dropout_conv=0.25, dropout_dense=0.5,learning_rate=1e-3, optimizer='adam', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),n_classes=5):
    #description of the input 
    inp = keras.Input(shape=input_shape)
    #augmentation becomes a level so kera sets the training flag automatically to true 
    x = make_augmentation()(inp)
    #builds 3 filter blocks for 
    for f in filters:  
    #using two conv layers before pooling for a larger pattern perception by the block
        for _ in range(2):    
            #each call is based on its previous output so that we can keep the data flowing in the nn                         
            #using filters to detect patterns -> each filter makes a channel so the ouput gains f channels
            #enable padding so the input keeps the same height and width
            x = layers.Conv2D(f, 3, padding='same')(x)
            #stabilizing training - mild regularization
            x = layers.BatchNormalization()(x)
            #using nonlinearity so the model can be more complex since the layers dont collapse into a linear function
            x = layers.Activation('relu')(x)
        #for shrinking the spatial size and give more translation tolerance 
        x = layers.MaxPooling2D()(x)
        if dropout_conv:
            x = layers.Dropout(dropout_conv)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(dense)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout_dense:
        x = layers.Dropout(dropout_dense)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    #compresses the whole chain flow into a trainable model
    model = keras.Model(inp, out)
    #adjusting weights so that the loss is not great
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def callbacks_for(ckpt):
    return [
        #stop if the accuracy doesnt improve after 20 epoches and restore the best weights that were found until then
        keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',patience=20, restore_best_weights=True),
        #when val loss stalls for 5 epochs, halve the learning rate. Lets the model take big steps early, fine steps later
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=5, min_lr=1e-6),
        #save the best model to disk to restore later on 
        keras.callbacks.ModelCheckpoint(ckpt, monitor='val_accuracy', mode='max',save_best_only=True),
    ]

def write_submission(probs, test_ids, output_path):
    #converts probabilities to 0-indexed predictions and the shift them back to their original labels
    preds = probs.argmax(axis=1) + 1
    #creating the dataframe using explicitly passed IDs
    submission_dataframe = pd.DataFrame({'id': test_ids.to_numpy(), 'label': preds})
    submission_dataframe.to_csv(output_path, index=False)
    

model = build_architecture(learning_rate=1e-3, optimizer='adam')
save_point = MODELE / 'cnn_best.keras'
istoric_antrenare = model.fit(set_antrenare, validation_data=set_validare, epochs=80,callbacks=callbacks_for(save_point), verbose=2)
print('best accuracy for validation:', max(istoric_antrenare.history['val_accuracy']))
write_submission(model.predict(img_test), df_test['id'], RES / 'submission_cnn_baseline.csv')

def plot_training_history(history_obj, save_dir: Path, filename: str = 'training_curves_cnn.png'):
    metrics = history_obj.history
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    #loss
    ax1.plot(metrics['loss'], label='Training Loss', linewidth=2)
    ax1.plot(metrics['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    #accuracy
    ax2.plot(metrics['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(metrics['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

plot_training_history(istoric_antrenare, RES)

def run_experiments(train_data, val_data, img_test, test_ids: pd.Series, baseline_val_acc: float, models_dir: Path, results_dir: Path, seed: int = 67) -> pd.DataFrame:
    #we build a dictionary of experiments with hyperparameter overrides so we can see how it varries
    variants = {
        'higher_lr':{'learning_rate': 1e-2, 'optimizer': 'adam'},
        'lower_lr':{'learning_rate': 1e-4, 'optimizer': 'adam'},
        'sgd_momentum':{'learning_rate': 1e-2, 'optimizer': 'sgd'},
        'no_dropout':{'learning_rate': 1e-3, 'optimizer': 'adam', 'dropout_conv': 0.0, 'dropout_dense': 0.0},
        'smaller_model':{'learning_rate': 1e-3, 'optimizer': 'adam', 'filters': (16, 32, 64)},
    }
    #we start with the baseline model so we can see how it compares to the rest that will be added sequentially
    results = [{'variant': 'baseline', 'val_accuracy': baseline_val_acc,'learning_rate': 1e-3, 'optimizer': 'adam'}]
    for name, params in variants.items():
        keras.utils.set_random_seed(seed)
        #any parameter not in the dictionary keeps its default in the building of the archi
        model = build_architecture(**params)
        ckpt_path = models_dir / f'cnn_{name}.keras'
        history = model.fit(train_data, validation_data=val_data, epochs=5, callbacks=callbacks_for(ckpt_path), verbose=0)
        best_acc = max(history.history['val_accuracy'])
        results.append({'variant': name, 'val_accuracy': best_acc, **params})
        probs = model.predict(img_test, verbose=0)
        write_submission(probs, test_ids, results_dir / f'submission_cnn_{name}.csv')

    results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)
    results_df.to_csv(results_dir / 'hp_table_cnn.csv', index=False)
    return results_df

baseline_acc = max(istoric_antrenare.history['val_accuracy'])
hp_report = run_experiments(
    train_data=set_antrenare,
    val_data=set_validare,
    img_test=img_test,
    test_ids=df_test['id'],
    baseline_val_acc=baseline_acc,
    models_dir=MODELE,
    results_dir=RES,
    seed=SEED,
)

print(hp_report[['variant', 'val_accuracy', 'learning_rate', 'optimizer']])


def evaluate_model(model_save_point_path: Path | str, img_validare, et_validare, results_dir: Path, class_labels: list = [1, 2, 3, 4, 5]) -> float:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    best_model = keras.models.load_model(model_save_point_path)

    y_pred = best_model.predict(img_validare, verbose=0).argmax(axis=1)

    val_acc = accuracy_score(et_validare, y_pred)
    target_names = [str(label) for label in class_labels]
    print(classification_report(et_validare, y_pred, target_names=target_names))
    #building the confusion matrix 
    cm = confusion_matrix(et_validare, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    figure, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('CNN Validation Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_cnn.png', dpi=300, bbox_inches='tight')
    plt.show()

    pd.DataFrame([{'model': 'cnn', 'val_accuracy': val_acc}]).to_csv(results_dir / 'val_scores_model2.csv', index=False)
    return val_acc

final_accuracy = evaluate_model(model_save_point_path=MODELE / 'cnn_best.keras', img_validare=img_validare, et_validare=et_validare, results_dir=RES,)


#we train the same architecture 3 times with 3 different seeds and the average the predictions using soft voting 

def train_seed_ensemble(train_ds, val_ds, img_validare, et_validare, img_test, test_ids: pd.Series,models_dir: Path, results_dir: Path, n_models: int = 3, base_seed: int = 67, epochs: int = 80) -> float:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    val_probabilities, test_probabilities = [], []

    for i in range(n_models):
        #computing the seed differently for each model
        current_seed = base_seed + i
        print(f"\n Current: {i+1} out of {n_models} | Specified seed: {current_seed}")

        #controls the random weights and shuffling for each model, making them diverse
        keras.utils.set_random_seed(current_seed)
        #building the untrained cnn architecture
        model = build_architecture(learning_rate=1e-3, optimizer='adam')
        ckpt_path = models_dir / f'cnn_seed_{current_seed}.keras'
        model.fit(train_ds, validation_data=val_ds, epochs=epochs,callbacks=callbacks_for(ckpt_path), verbose=0)
        # predict immediately and keep only probabilities rather than holding onto 3 full models in memory
        val_probabilities.append(model.predict(img_validare, verbose=0))
        test_probabilities.append(model.predict(img_test, verbose=0))

    ensemble_val_probs = np.mean(val_probabilities, axis=0)
    ensemble_test_probs = np.mean(test_probabilities, axis=0)

    ensemble_val_acc = accuracy_score(et_validare, ensemble_val_probs.argmax(axis=1))
    print(f"\n Final val accuracy after ensemble: {ensemble_val_acc:.4f}")

    write_submission(ensemble_test_probs, test_ids, results_dir/'submission_model2.csv')
    return ensemble_val_acc

ensemble_score = train_seed_ensemble(train_ds=set_antrenare, val_ds=set_validare,img_validare=img_validare, et_validare=et_validare, img_test=img_test, test_ids=df_test['id'],models_dir=MODELE, results_dir=RES,n_models=3, base_seed=SEED, epochs=80,)


def train_kfold_ensemble(imagini, etichete, img_test, test_ids: pd.Series, models_dir: Path, results_dir: Path, n_splits: int = 5, seed: int = 67, epochs: int = 60) -> float:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    #we divide the data into 5 equal parts and keeps the class proportions accurate to the whole data set so that we dont accidentally
    #train teh model on a fold consisting of for ex 60% class one data and we shuffle them for randomness
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    test_fold_probabilities = []
    #we use the out of fold prediction method so that the model doesnt cheat by looking at the data set and memorizing the answers
    #every single sample is predicted by a model that has never seen it before which assures us 0% data leakage
    oof_predictions = np.zeros(len(etichete), dtype=int)

    #we iterate through the splits and each fold has the set 80% training and 20% validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(imagini, etichete), start=1):
        print(f"\nFold {fold} out of {n_splits}")
        X_tr, y_tr = imagini[train_idx], etichete[train_idx]
        X_va, y_va = imagini[val_idx], etichete[val_idx]

        #each fold uses a different 20% as validation
        train_ds = build_dataset(X_tr, y_tr,training=True)
        val_ds   = build_dataset(X_va, y_va,training=False)

        #we start each model from a different random weight intialization which makes the 5 models diverse 
        keras.utils.set_random_seed(seed + fold)
        model = build_architecture(filters=(64, 128, 256), learning_rate=1e-3, optimizer='adam')
        ckpt_path = models_dir / f'cnn_ensemble_fold_{fold}.keras'

        model.fit(train_ds, validation_data=val_ds, epochs=epochs,callbacks=callbacks_for(ckpt_path), verbose=0)

        val_preds = model.predict(X_va, verbose=0).argmax(axis=1)
        oof_predictions[val_idx] = val_preds
        print(f"Fold {fold} has the val accuracy: {accuracy_score(y_va, val_preds):.4f}")

        test_fold_probabilities.append(model.predict(img_test, verbose=0))
    #we use cross validation to compare each oof predictions to its true label y
    overall_cv_accuracy = accuracy_score(etichete, oof_predictions)
    print(f"\n Final OOF CV accuracy: {overall_cv_accuracy:.4f}")

    #we use soft voting by averaging on the probabilities 
    #by letting the confidence voting, we pull the accuracy to the models with more confidence on a certain class than
    #by selecting the amount of times a model
    #has voted on the same class but with a marginally confidence
    ensemble_test_probs = np.mean(test_fold_probabilities, axis=0)
    write_submission(ensemble_test_probs, test_ids, results_dir / 'submission_model2_kfold.csv')
    return overall_cv_accuracy

cv_score = train_kfold_ensemble( imagini=imagini, etichete=etichete, img_test=img_test, test_ids=df_test['id'], models_dir=MODELE, results_dir=RES,)


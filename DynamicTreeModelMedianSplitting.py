#!/usr/bin/env python3
# filepath: /path/to/your/iris_dtmm.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.datasets import load_iris

#######################################
# Helper Functions for Cascaded Model
#######################################

def one_hot_encode(labels, num_classes):
    """Convert integer labels to one-hot encoded vectors."""
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)

def build_initial_model_multi(input_shape, m, output_dim, dropout_rate=0.3, l2_reg=1e-3):
    """
    Build the initial cascaded model with a shared base that uses dropout and L2 regularization.
    Each of the m output heads has `output_dim` neurons with softmax activation.
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Shared base network with three hidden layers:
    x = tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg))(x)
    # Shared layer with dropout:
    base_out = tf.keras.layers.Dense(32, activation='relu', name='shared_dense',
                                     kernel_regularizer=regularizers.l2(l2_reg))(x)
    base_out = tf.keras.layers.Dropout(dropout_rate)(base_out)
    
    outputs = []
    for i in range(m):
        head = tf.keras.layers.Dense(output_dim, activation='softmax', name=f'head_{i}')(base_out)
        outputs.append(head)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Compile with a loss and metric per output.
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=['categorical_crossentropy'] * m,
                  metrics=['accuracy'] * m)
    return model

def create_new_model_multi(old_model, m, new_output_dim, fine_tune_lr=1e-6):
    """
    Create a new cascaded model by reusing the shared base from old_model (which remains trainable
    but now with a lower learning rate via a new optimizer) and replacing the output heads.
    """
    base_output = old_model.get_layer('shared_dense').output
    outputs = []
    for i in range(m):
        head = tf.keras.layers.Dense(new_output_dim, activation='softmax', name=f'head_{i}')(base_output)
        outputs.append(head)
    
    model = tf.keras.models.Model(inputs=old_model.input, outputs=outputs)
    # Recompile with a lower learning rate overall to allow fine-tuning.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                  loss=['categorical_crossentropy'] * m,
                  metrics=['accuracy'] * m)
    return model

def generate_new_labels(prev_true, prev_pred):
    """
    Generate new integer labels for one category based on previous true labels and predictions.
    If prev_true==0 → new = prev_pred * 2; else new = prev_pred * 2 + 1
    """
    return np.where(prev_true == 0, prev_pred * 2, prev_pred * 2 + 1)

def predict_category_per_sample(model, X_test, n_iters, m):
    """
    For each test sample, calculate the positive probability for each head
    and then output the predicted category per sample.
    """
    # positive indices in a 2^n_iters softmax output: all odd positions
    positive_indices = [i for i in range(2**n_iters) if i % 2 == 1]
    preds = model.predict(X_test)
    num_samples = X_test.shape[0]
    sample_scores = np.zeros((num_samples, m))
    
    for i in range(m):
        head_pred = preds[i]  # shape: (num_samples, 2^n_iters)
        pos_prob = np.sum(head_pred[:, positive_indices], axis=1)
        sample_scores[:, i] = pos_prob
    
    # predicted category = head index with maximum positive-sum
    predicted_categories = np.argmax(sample_scores, axis=1)
    return predicted_categories

#######################################
# Data Preprocessing for Iris Dataset
#######################################

def preprocess_iris_data():
    """
    Load and preprocess the Iris dataset.
    Standardize features and return X, integer labels 0–2.
    """
    data = load_iris()
    X = data.data.astype(np.float32)
    y_int = data.target                   # 0,1,2
    # standardize
    X = StandardScaler().fit_transform(X)
    return X, y_int

#######################################
# Main Training and Comparison
#######################################

def main():
    # -------------------------------
    # Load and Preprocess Iris Data
    # -------------------------------
    X, y_int = preprocess_iris_data()
    
    # shuffle & split
    X, y_int = shuffle(X, y_int, random_state=42)
    X_train, X_test, y_train_int, y_test_int = train_test_split(
        X, y_int, test_size=0.2, stratify=y_int, random_state=42
    )
    
    # prepare binary labels for cascaded heads (one-vs-rest for each of 3 classes)
    m = 3
    y_train_binary = np.zeros((len(y_train_int), m), dtype=int)
    y_test_binary  = np.zeros((len(y_test_int), m), dtype=int)
    for idx, label in enumerate(y_train_int):
        y_train_binary[idx, label] = 1
    for idx, label in enumerate(y_test_int):
        y_test_binary[idx, label] = 1
    
    # iteration‑1 labels: each head a 2‑class problem
    y_train_list = [one_hot_encode(y_train_binary[:, i], 2) for i in range(m)]
    
    # -------------------------------
    # Cascaded Iterative Training
    # -------------------------------
    n_iters = 3  # creates heads of size 2, 4, 8
    
    # Iteration 1: 2-way
    cascaded_model = build_initial_model_multi(input_shape=(X_train.shape[1],),
                                               m=m, output_dim=2)
    cascaded_model.fit(X_train, y_train_list, epochs=2, batch_size=16, verbose=1)
    preds = cascaded_model.predict(X_train)
    prev_true = [y_train_binary[:, i] for i in range(m)]
    new_labels_list = []
    new_pred=np.zeros((len(y_train_int), m), dtype=int)
    for i in range(m):
        pos_indices = np.where(y_train_binary[:, i] == 1)[0]
        neg_indices = np.where(y_train_binary[:, i] == 0)[0]

        pos_probs = preds[i][pos_indices, 1] if len(pos_indices) > 0 else []
        neg_probs = preds[i][neg_indices, 0] if len(neg_indices) > 0 else []

        pos_threshold = np.median(pos_probs) if len(pos_probs) > 0 else 0.5
        neg_threshold = np.median(neg_probs) if len(neg_probs) > 0 else 0.5


        new_labels_category = np.zeros(len(X_train), dtype=int)
        for idx_val in range(len(X_train)):
            if y_train_binary[idx_val, i] == 1:
                if preds[i][idx_val, 1] > pos_threshold:
                    new_labels_category[idx_val] = 1
                else:
                    new_labels_category[idx_val] = 0
            else:
                if preds[i][idx_val, 0] > neg_threshold:
                    new_labels_category[idx_val] = 0
                else:
                    new_labels_category[idx_val] = 1
        

        new_labels_list.append(new_labels_category)
    new_labels_pred=[generate_new_labels(prev_true[i], new_labels_list[i])
                     for i in range(m)]
    new_labels = [one_hot_encode(generate_new_labels(prev_true[i], new_labels_list[i]), 4)
                  for i in range(m)]
    for i in range(m):
        for j in range(len(X_train)):
            new_pred[j,i]=new_labels_pred[i][j]
    
    # Iteration 2: 4-way
    print("\n### Cascaded Model: Iteration 2 (4-class heads)")
    cascaded_model = create_new_model_multi(cascaded_model, m, new_output_dim=4, fine_tune_lr=1e-4)
    cascaded_model.fit(X_train, new_labels, epochs=4, batch_size=16, verbose=1)
    preds = cascaded_model.predict(X_train)

    
    # Iteration 3: 8-way
    new_labels_list = []

    for i in range(m):
        index0 = np.where(new_pred[:, i] == 0)[0]
        index1 = np.where(new_pred[:, i] == 1)[0]
        index2 = np.where(new_pred[:, i] == 2)[0]
        index3 = np.where(new_pred[:, i] == 3)[0]

        probs0 = preds[i][index0, 0] if len(index0) > 0 else []
        probs1 = preds[i][index1, 1] if len(index1) > 0 else []
        probs2 = preds[i][index2, 2] if len(index2) > 0 else []
        probs3 = preds[i][index3, 3] if len(index3) > 0 else []

        threshold0 = np.median(probs0) if len(probs0) > 0 else 0.5
        threshold1 = np.median(probs1) if len(probs1) > 0 else 0.5
        threshold2 = np.median(probs2) if len(probs2) > 0 else 0.5
        threshold3 = np.median(probs3) if len(probs3) > 0 else 0.5

       
        new_labels_category = np.zeros(len(X_train), dtype=int)
        for idx_val in range(len(X_train)):
            if new_pred[idx_val, i] == 0:
                if preds[i][idx_val, 0] > threshold0:
                    new_labels_category[idx_val] = 0
                else:
                    new_labels_category[idx_val] = 1
            if new_pred[idx_val, i] == 1:
                if preds[i][idx_val, 1] > threshold1:
                    new_labels_category[idx_val] = 1
                else:
                    new_labels_category[idx_val] = 0
            if new_pred[idx_val, i] == 2:
                if preds[i][idx_val, 2] > threshold2:
                    new_labels_category[idx_val] = 2
                else:
                    new_labels_category[idx_val] = 3
            if new_pred[idx_val, i] == 3:
                if preds[i][idx_val, 3] > threshold3:
                    new_labels_category[idx_val] = 3
                else:
                    new_labels_category[idx_val] = 2


        new_labels_list.append(new_labels_category)
    new_labels = [one_hot_encode(generate_new_labels(prev_true[i], new_labels_list[i]), 8)
                  for i in range(m)]
    print("\n### Cascaded Model: Iteration 3 (8-class heads)")
    
    cascaded_model = create_new_model_multi(cascaded_model, m, new_output_dim=8, fine_tune_lr=1e-4)
    cascaded_model.fit(X_train, new_labels, epochs=40, batch_size=16, verbose=1)
    
    # -------------------------------
    # Final Prediction & Accuracy
    # -------------------------------
    predicted_categories = predict_category_per_sample(cascaded_model, X_test, n_iters, m)
    true_categories = np.argmax(y_test_binary, axis=1)
    accuracy = np.mean(predicted_categories == true_categories) * 100
    print(f"\nCascaded Model Test Accuracy on Iris: {accuracy:.2f}%")
    
if __name__ == '__main__':
    main()

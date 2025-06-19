#!/usr/bin/env python3
# filepath: /path/to/your/titanic_dtmm.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    If the previous prediction matches the true label, label becomes (true * 2 + 1); if not, (true * 2).
    """
    return np.where(prev_true % 2 == 0, prev_pred * 2, prev_pred * 2 + 1)

def predict_category_per_sample(model, X_test, n_iters, m):
    """
    For each test sample, calculate the positive probability for each head
    and then output the predicted category per sample. In the cascaded scheme,
    for each output head (with 2^n_iters classes), sum the softmax probabilities
    over 'positive' indices.
    
    Returns an array of predicted categories (one per sample).
    """
    positive_indices = [1, 3, 5, 7]  # Indices considered as YES.
    preds = model.predict(X_test)
    # list of m arrays; each has shape (num_samples, 2^n_iters)
    num_samples = X_test.shape[0]
    # Initialize a matrix to hold positive probabilities for each sample and each category.
    sample_scores = np.zeros((num_samples, m))
    
    for i in range(m):
        head_pred = preds[i]  # shape: (num_samples, 2^n_iters)
        # For each sample in head i, sum the probabilities at the positive indices.
        pos_prob = np.sum(head_pred[:, positive_indices], axis=1)
        sample_scores[:, i] = pos_prob
    # For each sample, the predicted category is the one with maximum positive probability.
    predicted_categories = np.argmax(sample_scores, axis=1)
    return predicted_categories

#######################################
# Baseline Model (Standard Network)
#######################################

def build_baseline_model(input_shape, num_classes, dropout_rate=0.3, l2_reg=1e-3):
    """A conventional feed-forward network for binary classification on Titanic."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

#######################################
# Data Preprocessing for Titanic Dataset
#######################################

def preprocess_titanic_data(csv_path):
    """
    Load and preprocess the Titanic dataset.
    Select relevant features, fill missing values, and encode categorical variables.
    """
    df = pd.read_csv(csv_path)
    
    # Select features and target
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[features + ['Survived']]
    
    # Fill missing Age with median and Embarked with mode
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Convert categorical features:
    # Encode Sex: female -> 0, male -> 1
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    # One-hot encode Embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df.drop('Embarked', axis=1), embarked_dummies], axis=1)
    
    # Separate features and target
    X = df.drop('Survived', axis=1).values
    y = df['Survived'].values  # 0 or 1
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

#######################################
# Main Training and Comparison
#######################################

def main():
    # -------------------------------
    # Load and Preprocess Titanic Data
    # -------------------------------
    # Using Kaggle's Titanic training dataset path
    csv_path = '/kaggle/input/titanic/train.csv'
    X, y = preprocess_titanic_data(csv_path)
    
    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # -------------------------------
    # Prepare Binary Labels for Cascaded Model
    # -------------------------------
    # For the cascaded scheme, we treat each Titanic class (0 = did not survive, 1 = survived) as separate categories.
    # Hence, m is set to 2.
    m = 2
    y_train_binary = np.zeros((len(y_train), m), dtype=int)
    y_test_binary  = np.zeros((len(y_test), m), dtype=int)
    for idx, label in enumerate(y_train):
        y_train_binary[idx, label] = 1
    for idx, label in enumerate(y_test):
        y_test_binary[idx, label] = 1
    
    # For iteration 1, one-hot encode each category's binary label (2 classes: 0 or 1)
    y_train_list = [one_hot_encode(y_train_binary[:, i], 2) for i in range(m)]
    
    # -------------------------------
    # Cascaded Iterative Training
    # -------------------------------
    # We'll run 3 iterations:
    # Iteration 1: 2 outputs per category.
    # Iteration 2: 4 outputs per category.
    # Iteration 3: 8 outputs per category.
    n_iters = 3
    
    # Retrain iteration 1 with 3 epochs.
    cascaded_model = build_initial_model_multi(input_shape=(X_train.shape[1],), m=m, output_dim=2)
    cascaded_model.fit(X_train, y_train_list, epochs=2, batch_size=16, verbose=1)
    preds = cascaded_model.predict(X_train)
    prev_pred_list = [np.argmax(preds[i], axis=1) for i in range(m)]
    prev_true_list = [y_train_binary[:, i] for i in range(m)]
    
    # ---- Iteration 2: 4-Class Classification ----
    print("\n### Cascaded Model: Iteration 2 (4-class classification)")
    new_labels_list = []
    for i in range(m):
        new_int_labels = generate_new_labels(prev_true_list[i], prev_pred_list[i])
        new_labels = one_hot_encode(new_int_labels, num_classes=4)
        new_labels_list.append(new_labels)
    
    # Create new model; now we allow fine-tuning of shared layers at a lower LR.
    cascaded_model = create_new_model_multi(cascaded_model, m, new_output_dim=4, fine_tune_lr=1e-4)
    cascaded_model.fit(X_train, new_labels_list, epochs=4, batch_size=16, verbose=1)
    preds = cascaded_model.predict(X_train)
    prev_pred_list = [np.argmax(preds[i], axis=1) for i in range(m)]
    # For next iteration, use the new labels as true labels.
    
    # ---- Iteration 3: 8-Class Classification ----
    print("\n### Cascaded Model: Iteration 3 (8-class classification)")
    new_labels_list = []
    for i in range(m):
        new_int_labels = generate_new_labels(prev_true_list[i], prev_pred_list[i])
        new_labels = one_hot_encode(new_int_labels, num_classes=8)
        new_labels_list.append(new_labels)
    
    cascaded_model = create_new_model_multi(cascaded_model, m, new_output_dim=8, fine_tune_lr=1e-4)
    cascaded_model.fit(X_train, new_labels_list, epochs=20, batch_size=16, verbose=1)
    
    # -------------------------------
    # Final Prediction from Cascaded Model using the New Rule
    # -------------------------------
    predicted_categories = predict_category_per_sample(cascaded_model, X_test, n_iters, m)
    true_categories = np.argmax(y_test_binary, axis=1)
    accuracy = np.mean(predicted_categories == true_categories) * 100
    print("\nCascaded Model - Per-sample Test Accuracy: {:.2f}%".format(accuracy))
    
if __name__ == '__main__':
    main()

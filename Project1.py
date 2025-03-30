#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import regularizers

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

def create_new_model_multi(old_model, m, new_output_dim, fine_tune_lr=1e-4):
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
    If the previous prediction matches the true label, label becomes (true * 2); if not, (true * 2 + 1).
    """
    return np.where(prev_true == prev_pred, prev_true * 2, prev_true * 2 + 1)

def predict_category(model, X_test, n_iters, m):
    """
    For each of the m output heads (each with 2^n_iters classes, where n_iters=3 gives 8 classes),
    sum the softmax probabilities for outputs corresponding to classes that are considered “YES”
    according to the new rule:
      - Classes 0, 3, 5, and 6 (when interpreted as a 3-bit number, they have an even number of set bits)
        contribute to positive (YES).
      - Classes 1, 2, 4, and 7 contribute to negative (NO).
    The final decision is the category (0 to m-1) that has the highest aggregated YES probability.
    """
    # In our cascaded scheme, n_iters=3 implies final dimension is 8.
    positive_indices = [0, 3, 5, 6]  # according to the new scheme
    preds = model.predict(X_test)  # list of m arrays, each with shape (num_samples, 8)
    total_scores = []
    for i in range(m):
        head_pred = preds[i]  # shape (num_samples, 8)
        # Sum probabilities over the indices considered positive:
        pos_prob = np.sum(head_pred[:, positive_indices], axis=1)  # per-sample positive probability
        total_scores.append(np.sum(pos_prob))  # aggregate over samples for this head
    total_scores = np.array(total_scores)
    best_category = np.argmax(total_scores)
    return best_category, total_scores

#######################################
# Baseline Model (Standard Network)
#######################################

def build_baseline_model(input_shape, num_classes, dropout_rate=0.3, l2_reg=1e-3):
    """A conventional feed-forward network for multi-class classification on Iris."""
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
# Main Training and Comparison
#######################################

def main():
    # -------------------------------
    # Load and Preprocess Iris Data
    # -------------------------------
    iris = load_iris()
    X = iris.data  # shape (150, 4)
    y = iris.target  # labels: 0, 1, or 2

    # Standardize features.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # For baseline model, create one-hot encoded labels for 3 classes.
    y_train_baseline = one_hot_encode(y_train, 3)
    y_test_baseline  = one_hot_encode(y_test, 3)

    # -------------------------------
    # Train Baseline Model
    # -------------------------------
    print("\nTraining Baseline Model (Conventional Network)")
    baseline_model = build_baseline_model(input_shape=(X_train.shape[1],), num_classes=3)
    baseline_model.summary()
    baseline_model.fit(X_train, y_train_baseline, epochs=50, batch_size=16, verbose=1)
    baseline_eval = baseline_model.evaluate(X_test, y_test_baseline, verbose=0)
    print("\nBaseline Test Accuracy: {:.2f}%".format(baseline_eval[1]*100))

    # -------------------------------
    # Prepare Binary Labels for Cascaded Model
    # -------------------------------
    # For the cascaded scheme, we treat each Iris class as a separate category.
    # For each sample, we create an m-dimensional binary label vector (m=3).
    m = 3
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

    # ---- Iteration 1: Determine half-convergence via early stopping ----
    print("\n### Cascaded Model: Iteration 1 (Preliminary run to determine full convergence)")
    prelim_model = build_initial_model_multi(input_shape=(X_train.shape[1],), m=m, output_dim=2)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, min_delta=0.001, verbose=1)
    prelim_model.fit(X_train, y_train_list, epochs=50, batch_size=16,
                     callbacks=[early_stop], verbose=1)
    full_conv_epochs = early_stop.stopped_epoch if early_stop.stopped_epoch > 0 else 50
    half_conv_epochs = max(1, int(np.ceil(full_conv_epochs / 2)))
    print("Full convergence at epoch {}. Training iteration 1 for {} epochs.".format(full_conv_epochs, half_conv_epochs))
    
    # Retrain iteration 1 with half-convergence epochs.
    cascaded_model = build_initial_model_multi(input_shape=(X_train.shape[1],), m=m, output_dim=2)
    cascaded_model.fit(X_train, y_train_list, epochs=half_conv_epochs, batch_size=16, verbose=1)
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
    cascaded_model.fit(X_train, new_labels_list, epochs=3, batch_size=16, verbose=1)
    preds = cascaded_model.predict(X_train)
    prev_pred_list = [np.argmax(preds[i], axis=1) for i in range(m)]
    # For next iteration, use the new labels as true labels.
    prev_true_list = [np.argmax(new_labels_list[i], axis=1) for i in range(m)]
    
    # ---- Iteration 3: 8-Class Classification ----
    print("\n### Cascaded Model: Iteration 3 (8-class classification)")
    new_labels_list = []
    for i in range(m):
        new_int_labels = generate_new_labels(prev_true_list[i], prev_pred_list[i])
        new_labels = one_hot_encode(new_int_labels, num_classes=8)
        new_labels_list.append(new_labels)
    
    cascaded_model = create_new_model_multi(cascaded_model, m, new_output_dim=8, fine_tune_lr=1e-4)
    cascaded_model.fit(X_train, new_labels_list, epochs=3, batch_size=16, verbose=1)
    
    # -------------------------------
    # Final Prediction from Cascaded Model using the New Rule
    # -------------------------------
    # Here, we use the new aggregation rule where classes with indices
    # [0, 3, 5, 6] (even number of set bits) contribute to YES.
    best_category, total_scores = predict_category(cascaded_model, X_test, n_iters, m)
    print("\nCascaded Model - Aggregated YES scores for each category on test data:", total_scores)
    print("Cascaded Model - Final predicted category (index):", best_category)
    
    # For reference, compare with true categories (argmax of binary test labels)
    true_categories = np.argmax(y_test_binary, axis=1)
    cascaded_accuracy = np.mean(np.array([best_category == tc for tc in true_categories])) * 100
    print("Cascaded Model - (Simplistic aggregated) Test Accuracy: {:.2f}%".format(cascaded_accuracy))
    
if __name__ == '__main__':
    main()

### Custom Neural Network Model Documentation

This document provides a detailed explanation of the custom neural network model implemented in custom_model.py and used within the training pipeline train_models.py.

---

### 1. Overview

The `CustomNeuralNetwork` is a deep learning model for multi-class classification, built from the ground up using only the NumPy library. It is designed to be a fully-featured, modern neural network, incorporating standard techniques for training, regularization, and optimization, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.

It functions as a feedforward neural network, also known as a Multi-Layer Perceptron (MLP), and is integrated into a scikit-learn compatible pipeline, allowing it to be trained and evaluated alongside other standard machine learning models.

### 2. Model Type

*   **Type**: Feedforward Neural Network / Multi-Layer Perceptron (MLP).
*   **Implementation**: Built from scratch using NumPy. It manually implements forward propagation, backpropagation, and optimization algorithms.

### 3. Architecture

The network has a sequential architecture consisting of an input layer, multiple hidden layers, and an output layer. Each hidden layer follows a `Dense -> BatchNorm -> ReLU -> Dropout` pattern.

**Flow Diagram:**

```
Input Layer (Features)
      |
      V
[Dense Layer 1 (128 neurons)] -> [Batch Normalization] -> [ReLU Activation] -> [Dropout (rate=0.3)]
      |
      V
[Dense Layer 2 (64 neurons)]  -> [Batch Normalization] -> [ReLU Activation] -> [Dropout (rate=0.2)]
      |
      V
[Dense Layer 3 (32 neurons)]  -> [Batch Normalization] -> [ReLU Activation]
      |
      V
[Output Layer (3 neurons)]    -> [Softmax Activation]
      |
      V
Output (Class Probabilities)
```

**Component Breakdown:**

*   **Input Layer**: The number of neurons in this layer is equal to the number of features in the input data from combined_dataset.csv after preprocessing.
*   **Dense Layers**: Fully connected layers that perform a linear transformation (`output = input @ Weights + bias`).
    *   **Initialization**: Weights are initialized using "He initialization" (`np.sqrt(2.0 / input_dim)`), which is standard practice for layers followed by a ReLU activation function to prevent vanishing/exploding gradients.
*   **Batch Normalization (`BatchNorm`)**: Applied after each dense layer (before activation). It normalizes the output of the previous layer by re-centering and re-scaling. This stabilizes and accelerates the training process. It operates differently during training (using batch statistics) and inference (using running averages).
*   **Activation Functions**:
    *   **ReLU (Rectified Linear Unit)**: `relu(x) = max(0, x)`. Used in all hidden layers. It introduces non-linearity, allowing the model to learn complex patterns.
    *   **Softmax**: Used in the output layer. It converts the raw output scores (logits) into a probability distribution over the classes, where the sum of probabilities is 1.
*   **Dropout**: A regularization technique applied after the activation function in the first two hidden layers. During training, it randomly sets a fraction of input units to 0 at each update, which helps prevent co-adaptation of neurons and reduces overfitting.

### 4. Inputs

The model takes a 2D NumPy array `X` as input, where:
*   Rows represent individual samples (e.g., a single sensor reading at a point in time).
*   Columns represent the features.

The features are derived from the combined_dataset.csv file. Based on the `get_feature_matrix` function call in train_models.py, the model uses a curated set of features from the CSV, excluding identifiers and the other target label.

**Example Input Features:**
`body_temp`, `ambient_temp`, `humidity`, `heart_rate`, `skin_resistance`, `resp_rate`, `movement`, `avg_sensor_temp`, `sensor_spread`, `temp_humidity_index`, `heat_index`, `hr_temp_product`, `skin_resistance_normalized`, `body_amb_diff`.

**Preprocessing**: Before being fed into the network, the input data `X` is standardized using a built-in scaler within the `CustomNeuralNetwork` class. This scaler computes the mean and standard deviation from the training data and applies the transformation `(X - mean) / std` to all data, ensuring all features have a mean of 0 and a standard deviation of 1.

### 5. Output

The model produces two types of outputs:

1.  **Class Probabilities (`predict_proba`)**: A 2D NumPy array where each row corresponds to an input sample and each column corresponds to a class probability. For a 3-class problem, the shape would be `(n_samples, 3)`. Each row sums to 1.
    *   Example: `[[0.8, 0.15, 0.05], [0.1, 0.7, 0.2]]`
2.  **Predicted Class (`predict`)**: A 1D NumPy array containing the final predicted class index for each sample. This is determined by finding the index of the highest probability in the `predict_proba` output.
    *   Example: `[0, 1]`

The classes correspond to the target labels: `0: 'Normal'`, `1: 'Moderate'`, `2: 'High'`.

### 6. Training and Optimization Flow

The training process is managed by the `fit` method and uses mini-batch stochastic gradient descent.

1.  **Data Splitting**: The training data is internally split into a training set (90%) and a validation set (10%).
2.  **Epochs & Batches**: The model iterates through the training data for a set number of `epochs`. In each epoch, the data is shuffled and processed in mini-batches.
3.  **Forward Pass**: For each mini-batch, the data flows through the network as described in the architecture section to compute the final class probabilities.
4.  **Loss Calculation**: The cross-entropy loss is calculated, which measures the difference between the predicted probabilities and the true labels. An L2 regularization penalty is added to the loss to discourage large weights.
5.  **Backward Pass (Backpropagation)**: The gradient of the loss is calculated with respect to every model parameter (weights and biases of Dense and BatchNorm layers) by propagating the error backward from the output layer to the input layer.
6.  **Parameter Update**: The **Adam optimizer** is used to update the model's parameters. Adam is an adaptive learning rate optimization algorithm that computes individual learning rates for different parameters from estimates of first and second moments of the gradients.
7.  **Learning Rate Decay**: The learning rate is gradually decreased during training (`lr = initial_lr * (decay_rate ** (epoch / 10))`) to allow for finer adjustments as the model converges.
8.  **Early Stopping**: After each epoch, the model evaluates its loss on the validation set. If the validation loss does not improve for a specified number of epochs (`patience`), training is stopped to prevent overfitting. The model then reverts to the weights that yielded the best validation loss.

### 7. Pros and Cons

**Pros:**

*   **Transparency & Control**: Since it's built from scratch, every step of the computation is explicit and customizable, offering deep insight into the model's inner workings.
*   **Modern Architecture**: It is not a simplistic MLP; it incorporates essential modern techniques like Batch Normalization, Dropout, Adam optimization, He initialization, and Early Stopping, making it a robust and effective model.
*   **No Dependencies**: It relies only on NumPy, making it lightweight and portable without requiring large deep learning libraries.
*   **Good Performance**: As shown in the training script, this custom implementation is competitive with established librar    y implementations like `RandomForestClassifier` and `GradientBoostingClassifier`.

**Cons:**

*   **Computational Efficiency**: NumPy-based implementations are significantly slower than optimized C++ or CUDA backends used by frameworks like TensorFlow/PyTorch, especially for large datasets and models. Training can be time-consuming.
*   **Complexity & Maintenance**: The code is more complex and prone to bugs than using a high-level API. Any new feature (e.g., a different layer type) must be manually implemented, including its forward and backward passes.
*   **Scalability**: It is not designed to run on GPUs, which severely limits its scalability for very large-scale deep learning tasks.
*   **Limited Feature Set**: It lacks the vast ecosystem of layers, optimizers, and utilities available in major deep learning frameworks.
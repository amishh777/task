# Task 1: Neural Network with Manual Backpropagation

## Objective

This project builds a small feedforward neural network and implements the forward pass, loss calculation, and backpropagation manually using NumPy. Automatic differentiation is not used for training.

The model classifies handwritten digits from 0 to 9 using the reduced dataset provided by scikit-learn.

## Dataset and preprocessing

Each image is 8 by 8 grayscale pixels. It is flattened into 64 values and scaled by dividing by 16.

The dataset contains 1,797 images. The rows are shuffled using a fixed random seed and split into:

- 1,437 training images;
- 360 testing images.

The weights are updated using only the training data. The test data is held back until after training.

## Network architecture

The architecture is:

```text
64 input pixels → 32 hidden ReLU neurons → 10 output classes
```

The first layer calculates:

```text
Z1 = pixel_digit_train @ W1 + b1
A1 = max(0, Z1)
```

The second layer produces ten raw scores:

```text
scores = A1 @ W2 + b2
```

Softmax converts these scores into probabilities. The implementation subtracts the largest score before applying the exponential function for numerical stability.

## Loss function

The model uses softmax followed by cross-entropy loss. For each image:

```text
image_loss = -log(correct_probability)
```

The losses are averaged over all training images. A small lower bound is used before taking the logarithm to avoid calculating `log(0)`.

Mean squared error was considered, but it creates more complicated derivatives when combined with softmax. Cross-entropy with softmax has the simpler derivative:

```text
d_scores = (P - target_values) / number_of_training_images
```

## Manual backpropagation

The output-layer gradients are:

```python
dW2 = A1.T @ d_scores
db2 = np.sum(d_scores, axis=0)
```

The gradient is then moved backward through the second layer and ReLU:

```python
dA1 = d_scores @ W2.T
dZ1 = dA1 * (Z1 > 0)
```

The first-layer gradients are:

```python
dW1 = pixel_digit_train.T @ dZ1
db1 = np.sum(dZ1, axis=0)
```

The weights and biases are updated using gradient descent:

```python
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2
```

The forward pass, loss, gradients, and updates are repeated for 1,000 iterations. Repeating this process is necessary because each update changes the parameters used by the next iteration.

## Gradient checking

The file `test_gradients.py` checks whether the manually calculated gradient is correct. It uses 20 images and checks the weight `W2[0][0]`.

The manual gradient is compared with a numerical gradient calculated by slightly increasing and decreasing the weight:

```text
numerical_gradient =
(loss_plus - loss_minus) / (2 * small_change)
```

The final check produced:

```text
Manual gradient:    0.020583038123039573
Numerical gradient: 0.020583038140564724
Gradient check passed
```

The close values provide evidence that the manually calculated output-layer gradient is correct.

## Results

The final test accuracy was:

```text
Test accuracy: 0.9444444444444444
```

This is approximately 94.44% accuracy on the 360 held-out test images. Training accuracy also increased during the iterations, showing that the parameter updates allowed the network to learn.

## Debugging and lessons learned

A major design decision was changing from mean squared error to cross-entropy. MSE with softmax is valid, but its derivative through softmax is more complicated because all softmax probabilities are connected. Cross-entropy gives the simpler `P - target_values` gradient.

It was also important to distinguish the different gradients. `d_scores` is the gradient with respect to the raw output scores. It is not the same as `dW2` or `dW1`. The weight gradients are obtained by continuing the chain rule through the matrix operations and ReLU.

The numerical gradient check was used to compare the manual derivative with an independent finite-difference estimate. This helped identify whether the backpropagation implementation was mathematically correct.

## Conclusion

The final implementation is a working feedforward neural network trained with manually calculated gradients. It performs handwritten-digit classification using NumPy, ReLU, softmax, cross-entropy loss, and gradient descent. The gradient check passed, and the model achieved approximately 94.44% accuracy on the held-out test set.


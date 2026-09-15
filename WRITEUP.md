# Task 1: Manual Backpropagation Neural Network

## 1. Project Objective
I worked on a neural network which reads handwritten images of digits from 0-9 and predicts the most probable event

## 2.  Dataset and Model Design

It works on reduced `sklearn` digits dataset, it had 1797 total 8x8 pixel images, which were flattened to a 64 pixel row
I used 80% of these images (shuffled on a specific seed) training  and 20% for testing.
The network works on
                    64 inputs --> 32 hidden neurons --> 10 output neurons 
                    
 PS: I started v late on this project and am still not very strong w understanding arrays and so many variables needed for gradient calculation
     I wanted to add another layer but that would be just copy-pasting the code which I wanted to avoid 

## 3. Forward Pass

The input images are passed through the network in stages.
First, the input pixels are multiplied by the first layer weights, and biases are added, which is defined as 'Z1', which is an array and stores data associated w each neuron in hidden layer

```python
Z1 = pixel_digit_train @ W1 + b1
```
I used activation function ReLU, it turns negative values to 0 and positive values remain as they are, it makes calculating derivative associated w it easier, 0 if z1 was neg. , 1 if z1 was +ve

scores is the raw score associated with each neuron in the output layer

```python
scores = A1 @ W2 + b2
```
This score is now passed into softmax function which converts scores to probabilty

## 4. Loss Function
I used cross-entropy loss, loss depends on probabilty associated with each digit as:
```python
image_loss = -np.log(correct_digit_probability)
```
The losses are averaged over all training images.
I tried switching to mean square error for loss function, but it made understanding code very difficult for me as it involved creating one-hot y array which seemed like a lot of un necessary work

Softmax along with cross entropy made the derivative relatively simpler to understand
d_scores = ∂Loss/∂scores which is equal to:
```python
d_scores = (P - target_values) / correct_digits_train.shape[0]
```

## 5. Manual Backpropagation

After calculating the loss, I calculated the gradients manually using the chain rule.

The first gradient is with respect to the raw output scores:

```python
d_scores = (P - target_values) / correct_digits_train.shape[0]
```
The output-layer weight gradient is calculated using:
```python
dW2 = A1.T @ d_scores
```
dW2 tells us how much each weight connecting the hidden layer to the output layer contributed to the loss.
The output-layer bias gradient is:
```python
db2 = np.sum(d_scores, axis=0)
```
The gradient is then moved backward into the hidden layer:
```python
dA1 = d_scores @ W2.T
```
Since the hidden layer uses ReLU, its derivative is applied using:
```python
dZ1 = dA1 * (Z1 > 0)
```
this makes dz1 very easy to calculate and gives output which I explained in the beginning.

Finally, the gradients for the first layer are calculated:
```python
dW1 = pixel_digit_train.T @ dZ1
db1 = np.sum(dZ1, axis=0)
```
The weights and biases are then updated using gradient descent, these calculations are repeated inside the training loop. The updated parameters are used during the next iteration.

## 6:Training and Testing
The model trains using the 1,437 training images. The weights are updated each iteration across 80% of dataset and 
the 20% testing images are not used during training. They are used afterward to measure how well the model works on unseen data.
The final test accuracy was: 99.4%

## 7. Gradient Check
I created test_gradients.py to compare a manually calculated gradient with a numerical gradient.
The numerical gradient is calculated by slightly increasing and decreasing one weight:
```python
numerical gradient = (loss_plus - loss_minus) / (2 × small_change)
```
Manual gradient: 0.020583038123039573

Numerical gradient: 0.020583038140564724

Gradient check passed
The values are very close, which indicates that the manual gradient calculation is correct.

## 8: Problems encountered
At first, I found it difficult to understand the difference between `P`, the loss, and the gradients. I initially thought that `d_scores` was the same as a weight gradient. I later understood that `d_scores` is the gradient with respect to the raw output scores, which is then used to calculate `dW2`, `db2`, `dW1`, and `db1`.

I also tried using mean squared error, but its derivative became difficult to follow when combined with softmax. I changed to cross-entropy loss because it is commonly used for classification and gives the simpler gradient:

```python
d_scores = (P - target_values) / correct_digits_train.shape[0]
```
Another mistake was putting the weight updates outside the training loop. In that case, the weights would only change once. I fixed this by putting the forward pass, loss calculation, gradient calculation, and parameter updates inside the loop.

I also had a hard time understanding backpropagation because the derivative from one layer has to be passed back to the previous layer. Each layer receives the gradient from the layer after it, modifies it according to its own operation, and passes the result backward. The process initially looked like many unrelated variables, but I eventually understood that they represent the gradient at different stages of the same network.

The numerical gradient check helped me confirm that my manual gradient calculations were correct.

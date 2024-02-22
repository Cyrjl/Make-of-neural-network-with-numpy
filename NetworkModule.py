import numpy as np
from numpy import ndarray as Tensor

from typing import (Dict, Tuple, Callable, 
                    Sequence, Iterator, NamedTuple)
Func = Callable[[Tensor], Tensor]

# Class constructiong a generic loss function

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """ 
        Compute the loss between predictions and actual labels 
        """
        raise NotImplementedError

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        """ 
        Compute the gradient for the backward pass 
        """
        raise NotImplementedError

# Mean square error loss

class MeanSquareError(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2) / len(actual)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return (2 * (predicted - actual)) / len(actual)
    
# Binary cross entropy loss

class BinCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return - ((np.sum(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))) / len(actual))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return - (((actual / predicted) - ((1 - actual) / (1 - predicted))) / len(actual))
    
# Class constructing a generic layer

class Layer:
    def __init__(self) -> None:

        # Store the parameters values and gradients in dictionnaries
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


# Class constructing a linear layer of neurons

class Linear(Layer):
    """
    Inputs are of size (batch_size, input_size)
    Outputs are of size (batch_size, output_size)
    """
    def __init__(self, input_size: int, output_size: int, Seed: int = 0) -> None:
    
        # Inherit from base class Layer
        super().__init__()
        
        # Initialize the weights and bias with random values
        if Seed != 0:
            np.random.seed(Seed)

        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        inputs shape is (batch_size, input_size)
        """
        self.inputs = inputs

        # Compute the feed forward pass
        # (b,i) @ (i,o) + (1,o) = (b,o)
        return inputs @ self.params["w"] + self.params["b"]
        
         
    def backward(self, grad: Tensor) -> Tensor:
        """
        grad shape is (batch_size, output_size)
        """
        # Compute the gradient parameters for the layer
        # (i,b) @ (b,o) = (i,o)
        self.grads["w"] =  np.transpose(self.inputs) @ grad
        self.grads["b"] = grad #(b,o)
    
        # Compute the feed backward pass
        # (b,o) @ (o,i) = (b,i)
        return grad @ np.transpose(self.params["w"])


#Defining possible activation functions

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    return 1 - (np.tanh(x))**2

def sigmoid(x: Tensor) -> Tensor:
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x)*(1 - sigmoid(x))


# Class constructing an activation layer

class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """
    def __init__(self, f: Func, f_prime: Func) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad

# Class defining how to organize the data in series of batch

Batch = NamedTuple("Batch", [("inputs", Tensor), ("targets", Tensor)])
        
class BatchIterator:
    """ 
    Organize the data in batch that are shuffled at each epoch
    """
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs: Tensor, targets: Tensor) -> Iterator[Batch]:
        """ 
        Create batch iteratively and yields them one after the other
        """
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start:end]
            batch_targets = targets[start:end]
            yield Batch(batch_inputs, batch_targets)

# Class constructing the network, doing the full forward and backward pass and optimizing the parameters

class NeuralNet:
    def __init__(self, layers: Sequence[Layer], lr: float = 0.01) -> None:
        self.layers = layers
        # Learning rate
        self.lr = lr 
        
    def forward(self, inputs: Tensor) -> Tensor:
        """
        The forward pass takes the layers in order
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        The backward pass is the other way around
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def optimize(self) -> None:
        """
        Optimize the paramaters value at each step
        """
        for layer in self.layers:
            for name in layer.params.keys():
                layer.params[name] = layer.params[name] - self.lr * layer.grads[name]


    def validate(self, inputs: Tensor, targets: Tensor,
                 loss: Loss = BinCrossEntropy(),
                 iterator = BatchIterator(),
                 cut: float = 0.5) -> Tuple:
        """
        Compute the accuracy and loss of the network 
        on another dataset not seen in train
        """

        # Lists to store the input variables,
        # predicted and actual labels 
        # in the right order   
        Predicted_list: Sequence =[] 
        Actual_list: Sequence = []
        Input_list:Sequence = [] 
        
        for batch in iterator(inputs, targets):

            Batch_loss : Sequence = [] 

            predicted = self.forward(batch[0])
            for p in predicted:
                Predicted_list.append(p)
            for a in batch[1]:
                Actual_list.append(a)
            for i in batch[0]:
                Input_list.append(i)

            Batch_loss.append(loss.loss(predicted, batch[1]))   
        
        Predicted_array = np.array(Predicted_list)
        Actual_array = np.array(Actual_list)
        Input_array = np.array(Input_list)

        # Compute the loss as the mean of batch loss     
        val_loss = np.mean(Batch_loss)
        # Decide the label resulting from prediction
        Round_predicted = np.where(Predicted_array >= cut, 1, 0)
        # Compare all the labels
        val_acc = np.mean(Round_predicted==Actual_array) * 100

        return (val_loss, val_acc, Actual_array, Predicted_array, Input_array)

    def train(self, inputs: Tensor, targets: Tensor,
              val_inputs: Tensor, val_targets: Tensor,
              loss: Loss = BinCrossEntropy(),
              iterator =  BatchIterator(),
              num_epochs: int = 1000,
              cut: float = 0.5,
              Print: bool = True) -> Tuple:
        """
        Train the network in series of batch
        and for a number of epochs
        Compute the evolution of the loss and accuracy
        """
        Loss_list : Sequence = []
        Acc_list : Sequence = []

        Val_Loss_list : Sequence = []
        Val_Acc_list : Sequence = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Lists to store the predicted and actual labels 
            # in the right order, at each epoch
            Predicted_list: Sequence = []
            Actual_list: Sequence = [] 

            for batch in iterator(inputs, targets):
                
                Batch_loss : Sequence = []
                Batch_grad : Sequence = []

                predicted = self.forward(batch[0])
                for p in predicted:
                    Predicted_list.append(p)
                for a in batch[1]:
                    Actual_list.append(a)
                    
                Batch_loss.append(loss.loss(predicted, batch[1]))
                grad = loss.grad(predicted, batch[1])
                Batch_grad.append(grad) 
                self.backward(grad)
                self.optimize()


            Predicted_array = np.array(Predicted_list)
            Actual_array = np.array(Actual_list)   

            # Compute the loss as the mean of batch loss    
            epoch_loss = np.mean(Batch_loss)
            # Decide the label resulting from prediction
            Round_predicted = np.where(Predicted_array >= cut, 1, 0)
            # Compare all the labels for the epoch
            epoch_acc = np.mean(Round_predicted==Actual_array) * 100

            Loss_list.append(epoch_loss)
            Acc_list.append(epoch_acc)
                        
            # Compute the validation accuracy every 100 epochs
            if epoch % 100 == 0:
                val_loss = self.validate(val_inputs, val_targets, iterator=iterator, cut=cut)[0]
                val_acc = self.validate(val_inputs, val_targets, iterator=iterator, cut=cut)[1]

                Val_Loss_list.append(val_loss)
                Val_Acc_list.append(val_acc)

                if Print == True:
                    print("# Training" )
                    print(f'Epoch = {epoch:4d} Loss = {epoch_loss:.3f} Acc = {epoch_acc:.3f}')
                    print("# Validation")
                    print(f'Epoch = {epoch:4d} Loss = {val_loss:.3f} Acc = {val_acc:.3f}')
                    print("--------------------------------------")

        return (Loss_list, Acc_list, Actual_array, Predicted_array, Val_Loss_list, Val_Acc_list)
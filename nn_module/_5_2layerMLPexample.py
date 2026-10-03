import torch


# Module base class provides a lot of functionality making it easier to build and trained models.
class TwoLayerMLP(torch.nn.Module):
    # num_inputFeatures: Number of input features the sample data record will have.
    # num_outputClasses: Number of output classes.
    def __init__(self, num_inputFeatures, num_outputClasses):
        super().__init__()

        # Define the layers of the model.
        self.layers = torch.nn.Sequential(  # Sequential container to hold layers.
            # Input layer.
            # Input layer technically does not have neurons. It is just a placeholder for the input data (no computation done here).
            # In this case, the input shape is (num_inputFeatures) input units.
            # All input units are connected to all neurons in the next layer i.e. 1st hidden layer.
            # 1st hidden layer.
            # Consists of 30 neurons. Each neuron
            #    has its own set of weights and bias. From training, these parameters (weight, bias) will be updated to minimize the loss function.
            #    will take num_inputFeatures inputs (i.e. input features) and produce a single output (scalar).
            # Output (scalar) for these neurons will be combined into a vector (size 30) and passed to each neuron in the next layer i.e. 2nd hidden layer.
            # Total parameters this layer = (num_inputFeatures X num_outputClasses) weights + num_outputClasses biases. For example, if num_inputFeatures=50 and num_outputClasses=30, then total parameters = 50*30 + 30 = 1530.
            # nn.Linear has an optimized weight initialization scheme contributing to more stable and effective model training than nn.Parameter.
            torch.nn.Linear(num_inputFeatures, 30),
            torch.nn.ReLU(),  # ReLU activation function. Non-linear activation funcitons are placed between the hidden layers.
            # 2nd hidden layer.
            # Consists of 20 neurons. Each neuron
            #    has its own set of weights and bias. From training, these parameters (weight, bias) will be updated to minimize the loss function.
            #    will take 30 inputs (i.e. output of the previous layer) and produce a single output (scalar).
            # Output (scalar) for these neurons will be combined into a vector (size 20) and passed to each neuron in the next layer i.e. output layer.
            # Total parameters this layer = (30 X 20) weights + 20 biases. For example, if num_inputFeatures=50 and num_outputClasses=30, then total parameters = 30*20 + 20 = 620.
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # Output layer.
            # Consists of num_outputClasses neurons. Each neuron
            #    has its own set of weights and bias. From training, these parameters (weight, bias) will be updated to minimize the loss function.
            #    will take 20 inputs (i.e. output of the previous layer) and produce a single output (scalar).
            # Output (scalar) for these neurons will be combined into a vector (size num_outputClasses) and passed to the loss function.
            torch.nn.Linear(
                20, num_outputClasses
            ),  # Linear layer with 20 inputs and num_outputClasses outputs.
        )

    def forward(self, x):
        logits = self.layers(
            x
        )  # Use built-in Forward pass through the network instead of calling each layer individually.
        return logits  # Return the output of the network. The output here are the logits (raw scores) for each class since we have not applied any activation function.


def run_demo() -> None:
    torch.manual_seed(123)
    model = TwoLayerMLP(50, 3)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Number of trainable model parameters: {num_params}")
    print(model.layers[0].weight)
    print(model.layers[0].bias)
    x = torch.randn(1, 50)
    print(x)
    model.eval()
    with torch.no_grad():
        print(f"scores: {model(x)}")


if __name__ == "__main__":
    run_demo()

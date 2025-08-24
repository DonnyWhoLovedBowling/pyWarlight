import torch

class PopArt:
    def __init__(self, output_layer, beta=0.0003, epsilon=1e-5):
        self.mean = 0.0
        self.sq_mean = 0.0
        self.std = 1.0
        self.beta = beta
        self.epsilon = epsilon
        self.n = 0
        self.output_layer = output_layer  # nn.Linear

    def update(self, targets: torch.Tensor):
        # Update running mean and std
        batch_mean = targets.mean().item()
        batch_sq_mean = (targets ** 2).mean().item()
        self.n += 1
        self.mean = (1 - self.beta) * self.mean + self.beta * batch_mean
        self.sq_mean = (1 - self.beta) * self.sq_mean + self.beta * batch_sq_mean
        self.std = ((self.sq_mean - self.mean ** 2) + self.epsilon) ** 0.5

    def normalize(self, targets: torch.Tensor):
        return (targets - self.mean) / (self.std + self.epsilon)

    def denormalize(self, values: torch.Tensor):
        return values * (self.std + self.epsilon) + self.mean

    def adjust_weights(self, old_mean, old_std):
        # Adjust output layer weights and bias to preserve unnormalized output
        if hasattr(self.output_layer, 'weight') and hasattr(self.output_layer, 'bias'):
            w = self.output_layer.weight.data
            b = self.output_layer.bias.data
            new_std = self.std + self.epsilon
            old_std = old_std + self.epsilon
            w.mul_(old_std / new_std)
            b.sub_(old_mean).mul_(old_std / new_std).add_(self.mean)


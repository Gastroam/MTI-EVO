"""
Dense tissue primitives for matrix-style adaptation.
"""

import numpy as np

from mti_evo.core.config import MTIConfig


class MTIDenseTissue:
    """
    A layer of neuronal tissue densely connected.
    Handles multiple neurons as a single matrix entity (Tensor).
    Optimized for 'War Economy': Maximum compute density, minimum overhead.
    """

    def __init__(self, input_size, output_size, config: MTIConfig = None):
        if config is None:
            config = MTIConfig()

        self.input_size = input_size
        self.output_size = output_size
        self.gravity = config.gravity
        self.momentum = config.momentum
        self.initial_lr = config.initial_lr
        self.decay_rate = config.decay_rate
        seed = config.random_seed if getattr(config, "deterministic", True) else None
        self.rng = np.random.default_rng(seed)

        # SYNAPTIC INITIALIZATION
        self.weights = self.rng.normal(0.0, 0.01, size=(input_size, output_size))
        self.bias = np.zeros((1, output_size))

        # KINETIC MEMORY
        self.v_weights = np.zeros_like(self.weights)
        self.v_bias = np.zeros_like(self.bias)

        # Internal State Storage
        self.inputs = None
        self.logits = None
        self.output = None

    def forward(self, inputs):
        """
        Collective Perception.
        """
        self.inputs = inputs
        self.logits = np.dot(inputs, self.weights) + self.bias
        self.output = 1 / (1 + np.exp(-self.logits))
        return self.output

    def adapt(self, y_true, learning_rate=0.5, momentum=0.9):
        """
        Matrix Neuroplasticity.
        """
        if self.output is None:
            raise ValueError("Must call forward() before adapt()")

        # 1. PAIN CALCULATION
        error = self.output - y_true
        weighted_error = error * np.where(y_true == 1, self.gravity, 1.0)

        # 2. GRADIENTS
        d_weights = np.dot(self.inputs.T, weighted_error) / len(self.inputs)
        d_bias = np.mean(weighted_error, axis=0, keepdims=True)

        # 3. MOMENTUM UPDATE
        self.v_weights = (momentum * self.v_weights) - (learning_rate * d_weights)
        self.v_bias = (momentum * self.v_bias) - (learning_rate * d_bias)

        # Apply Plasticity
        self.weights += self.v_weights
        self.bias += self.v_bias

        # Return Average Voltage
        loss = -np.mean(
            self.gravity * y_true * np.log(np.clip(self.output, 1e-15, 1 - 1e-15))
            + (1 - y_true) * np.log(1 - np.clip(self.output, 1e-15, 1 - 1e-15))
        )
        return loss

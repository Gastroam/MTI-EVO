"""
Core neuron primitives for MTI-EVO.
"""
import math
from dataclasses import dataclass, field
from typing import List, Any
import time
import numpy as np
from mti_evo.core.config import MTIConfig


class MTINeuron:
    """
    MTI-EVO Artificial Neuron
    Implements the 'Resonant Homeostasis' protocol:
    1. Sigmoid Activation (Perception)
    2. Bio-Voltage Loss (Pain/Gravity)
    3. Kinetic Adaptation (Momentum + Decay)
    """

    def __init__(self, input_size, config: MTIConfig = None, trainable_bias=True, rng=None, time_fn=None):
        """
        Initialize the Resonant Neuron.
        trainable_bias: If False, bias remains fixed (useful for pure pattern resonance).
        """
        if config is None:
            config = MTIConfig()  # Default DNA

        if rng is not None:
            self.rng = rng
        else:
            seed = config.random_seed if getattr(config, "deterministic", True) else None
            self.rng = np.random.default_rng(seed)

        # Time Source (Dependency Injection for Determinism)
        self.time_fn = time_fn if time_fn else time.time

        # Synaptic Weights
        # [PHASE 7.1] Curiosity Logic: Initialize to Silence.
        # Small random weights + Strong Negative Bias = Low Resonance (Curiosity)
        # Prevents "False Recognition" of random vectors.
        # Ensure we use the instance rng for this.
        self.weights = self.rng.normal(0.0, 0.05, size=(input_size,))
        self.bias = -2.0  # Default Inhibition (Sigmoid(-2) ~= 0.11)

        # Kinetic Memory (Velocity) for Momentum
        self.velocity = np.zeros(input_size)

        # DNA / Hyperparameters from Config
        self.gravity = config.gravity
        self.momentum = config.momentum
        self.initial_lr = config.initial_lr
        self.decay_rate = config.decay_rate
        self.trainable_bias = trainable_bias

        # Resonance Control
        self.weight_cap = getattr(config, "weight_cap", 80.0)
        self.diminishing_returns = getattr(config, "diminishing_returns", True)

        # [PHASE 6] Coherence Engine Data
        self.semantic_vector = None  # Cached Embedding
        self.health_index = 1.0  # Epistemic Health (0.0 - 1.0)
        self.critic_history = []  # History of feedback scores
        self.label = None  # [PHASE 27] Semantic Label (e.g. "auth.py")

        # State tracking
        self.age = 0  # Epoch/Cycle count
        now = self.time_fn()
        self.last_accessed = now
        self.created_at = now

    def sigmoid(self, x):
        """Activation function: squashes output to (0, 1)."""
        return 1 / (1 + np.exp(-x))

    def _validate_inputs(self, inputs) -> np.ndarray:
        """
        Single source of truth for input normalization used by perceive/adapt.
        """
        x = np.asarray(inputs)
        x = np.atleast_2d(x)
        expected_dim = self.weights.shape[0]

        if x.shape[1] != expected_dim:
            if x.size == expected_dim:
                x = x.reshape(-1, expected_dim)
            elif x.shape[1] > expected_dim:
                current_dim = expected_dim
                new_dim = x.shape[1]

                extra_weights = self.rng.normal(0.0, 0.01, size=(new_dim - current_dim,))
                self.weights = np.concatenate([self.weights, extra_weights])

                extra_vel = np.zeros(new_dim - current_dim)
                self.velocity = np.concatenate([self.velocity, extra_vel])
            else:
                raise ValueError(
                    f"Shape Mismatch: Neuron expects {expected_dim} inputs, got {x.shape}"
                )

        return x

    def perceive(self, inputs):
        """
        Forward Pass (Perception).
        Returns scalar for single input, array for batch input.
        """
        self.last_accessed = self.time_fn()

        # Track if input was originally 1D (single sample)
        was_1d = np.ndim(inputs) == 1

        # Input Validation
        inputs = self._validate_inputs(inputs)

        # Z = W . X + b
        logits = np.dot(inputs, self.weights) + self.bias
        result = self.sigmoid(logits)

        # Return scalar if single input was provided
        if was_1d:
            return float(result[0])
        return result

    def adapt(self, inputs, y_true="auto"):
        """
        Homeostatic Adaptation (Backward Pass / Learning).
        """
        self.age += 1
        self.last_accessed = self.time_fn()

        # 1. Perception (single-source validated shape)
        x = self._validate_inputs(inputs)
        y_pred = np.atleast_1d(self.perceive(x)).astype(float)

        # Logic Refinement: Signal Reinforcer
        # If input_size == 1 and y_true is "auto", we assume Self-Reinforcement (Signal Detection)
        if isinstance(y_true, str) and y_true == "auto":
            if self.weights.shape[0] == 1:
                y_true = 1.0  # "I exist" - Reinforce presence
            else:
                raise ValueError("y_true must be provided for multi-dimensional pattern matching.")

        # 2. Voltage (Loss) Calculation
        # [PHASE 62] Long-Term Potentiation (LTP)
        # Repeated stimulation increases structural mass (Gravity)
        self.gravity = min(self.gravity + 0.05, 50.0)  # Cap at 50

        if np.isscalar(y_true):
            y_true_arr = np.full(y_pred.shape, float(y_true))
        else:
            y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
            if y_true_arr.size == 1 and y_pred.size > 1:
                y_true_arr = np.full(y_pred.shape, float(y_true_arr[0]))
            elif y_true_arr.size != y_pred.size:
                raise ValueError(
                    f"y_true shape mismatch. Expected {y_pred.size} value(s), got {y_true_arr.size}."
                )

        epsilon = 1e-15
        y_pred_clamped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Weighted Binary Cross Entropy
        # [Opt-2] Optimized weight factor (avoid np.where if constant)
        if isinstance(y_true, float) and y_true == 1.0:
             weight_factor = self.gravity
        else:
             weight_factor = np.where(y_true_arr == 1.0, self.gravity, 1.0)

        loss_vector = -1 * (
            weight_factor * y_true_arr * np.log(y_pred_clamped)
            + (1 - y_true_arr) * np.log(1 - y_pred_clamped)
        )
        voltage = np.mean(loss_vector)

        # 3. Gradient Calculation
        error = y_pred - y_true_arr
        
        # [Opt-2] Avoid allocating new array for weighted_error if scalar weight count
        if np.isscalar(weight_factor):
             weighted_error = error * weight_factor
        else:
             weighted_error = error * weight_factor

        if x.shape[0] > 1:
            # Batch Mode
            gradient = np.dot(x.T, weighted_error) / x.shape[0]
            bias_gradient = np.mean(weighted_error)
        else:
            # Single Instance Mode
            # x[0] is (Dim,)
            # weighted_error[0] is scalar
            gradient = x[0] * weighted_error[0]
            bias_gradient = weighted_error[0]

        # 4. Neuroplasticity Adjustment (Cooling)
        current_lr = self.initial_lr / (1 + self.decay_rate * self.age)

        # [PHYSICS UPDATE] Diminishing Returns (Logarithmic Dampening)
        # Prevents runaway saturation by making stronger neurons harder to impress.
        if self.diminishing_returns:
            current_magnitude = np.mean(np.abs(self.weights))
            # Damping factor: 1 / (1 + W/10)
            reinforcement_damping = 1.0 / (1.0 + (current_magnitude / 10.0))
            current_lr *= reinforcement_damping

        # 5. Momentum Update
        # [Opt-2] In-place update to reduce memory churn
        # self.velocity = (self.momentum * self.velocity) - (current_lr * gradient)
        
        # v *= momentum
        self.velocity *= self.momentum
        # v -= lr * grad
        self.velocity -= (gradient * current_lr)

        # Update Synapses
        self.weights += self.velocity

        if self.trainable_bias:
            self.bias -= current_lr * bias_gradient

        # [PHYSICS UPDATE] Weight Normalization (Stability Ceiling)
        # Hard cap on absolute mass to prevent black holes.
        max_w = self.weight_cap
        current_norm = np.linalg.norm(self.weights)

        if current_norm > max_w:
            scale_factor = max_w / current_norm
            self.weights *= scale_factor
            # Drain Kinetic Energy on impact (Simulating collision with ceiling)
            self.velocity *= 0.5

        return {
            "voltage": voltage,
            "prediction": float(y_pred[0]) if y_pred.size == 1 else y_pred,
            "lr": current_lr,
            "weights_mean": np.mean(self.weights),
        }

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

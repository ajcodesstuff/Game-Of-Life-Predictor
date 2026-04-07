# Game of Life Behavior Predictor

A project exploring whether the long-term behavior of a configuration in Conway's Game of Life can be predicted directly from its initial state using machine learning


# Problem

The Game of Life is a cellular automaton governed by simple local rules, yet it exhibits highly complex global behavior.
A known challenge is:
Determining whether a given configuration will stabilize, oscillate, or die out typically requires explicit simulation.
This project explores whether a model (CNN) can learn to predict this behavior directly from the initial grid.

# Approach
Grid size: 20 × 20
Data generation:
random initial configurations
simulation run to determine eventual behavior
Dataset:
~121,000 training samples
~10,000 test samples
Labels: Stable (eventually reaches fixed configuration), Oscillating, Dead
Model: Convolutional Neural Network (CNN)
multiple architectures tested (increasing complexity)


# Results
The dataset was imbalanced, with a large proportion of configurations falling into the Stable class.
Baseline:
Always predicting “Stable” → ~50.5% accuracy
Model performance:
CNN → ~50–52% accuracy
The model failed to significantly outperform the baseline.


# Conclusion
This project did not achieve meaningful predictive performance, but it provided insight into:
The limitations of naive ML approaches on dynamic systems
The importance of problem framing
The complexity that can arise from simple rules

Even though the model failed to outperform the baseline, the exploration highlights why this problem is non-trivial and worth deeper investigation.

# Micrograd From Scratch

A tiny scalar automatic differentiation engine and neural network library
built from scratch in Python.

The goal of this project is to understand how automatic differentiation,
backpropagation, and neural networks work internally rather than treating
them as black boxes.


## Project Structure

```text
micrograd-from-scratch/
│
├── src/
│   └── micrograd/
│       ├── engine.py
│       ├── nn/
│       │   └── modules.py
│       └── visualization/
│           └── graph.py
│
├── tests/
│   ├── test_engine.py
│   └── test_nn.py
│
├── examples/
│   ├── basic_autograd.py
│   ├── graph_visualization.py
│   └── train_mlp.py
│
├── README.md
├── pyproject.toml
└── requirements.txt

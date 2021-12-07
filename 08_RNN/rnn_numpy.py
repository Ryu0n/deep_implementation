"""
ht = tanh(Wx*x(t) + Wh*h(t-1) + b)

inputs (x) : (time_steps, input_dim)

Wx : (hidden_size, input_dim)
x(t) : (input_dim, 1)

Wh : (hidden_size, hidden_size)
h(t-1) : (hidden_size, 1)

b : (hidden_size, 1)
"""

import numpy as np

# sequence data : N x T x D
time_steps = 10  # T
input_dim = 4    # D (d)

hidden_size = 8  # D_{h}

inputs = np.random.random((time_steps, input_dim))
hidden_state_t = np.zeros((hidden_size,))

Wx = np.random.random((hidden_size, input_dim))
Wh = np.random.random((hidden_size, hidden_size))
b = np.random.random((hidden_size,))

total_hidden_states = []

for input_t in inputs:
    output_t = np.tanh(np.dot(Wx, input_t) + np.dot(Wh, hidden_state_t) + b)
    total_hidden_states.append(list(output_t))
    print(np.shape(total_hidden_states))
    hidden_state_t = output_t

total_hidden_states = np.stack(total_hidden_states, axis=0)
print('\n', total_hidden_states.shape)
print(total_hidden_states)

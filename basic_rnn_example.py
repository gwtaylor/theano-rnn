import numpy
import theano
import theano.tensor as TT

# number of hidden units
n = 50
# number of input units
nin = 5
# number of output units
nout = 5

# input (where first dimension is time)
u = TT.matrix()
# target (where first dimension is time)
t = TT.matrix()
# initial hidden state of the RNN
h0 = TT.vector()
# learning rate
lr = TT.scalar()
# recurrent weights as a shared variable
W = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01))
# input to hidden layer weights
W_in = theano.shared(numpy.random.uniform(size=(nin, n), low=-.01, high=.01))
# hidden to output layer weights
W_out = theano.shared(numpy.random.uniform(size=(n, nout), low=-.01, high=.01))


# recurrent function (using tanh activation function) and linear output
# activation function
def step(u_t, h_tm1, W, W_in, W_out):
    h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
    y_t = TT.dot(h_t, W_out)
    return h_t, y_t

# the hidden state `h` for the entire sequence, and the output for the
# entrie sequence `y` (first dimension is always time)
[h, y], _ = theano.scan(step,
                        sequences=u,
                        outputs_info=[h0, None],
                        non_sequences=[W, W_in, W_out])
# error between output and target
error = ((y - t) ** 2).sum()
# gradients on the weights using BPTT
gW, gW_in, gW_out = TT.grad(error, [W, W_in, W_out])
# training function, that computes the error and updates the weights using
# SGD.
fn = theano.function([h0, u, t, lr],
                     error,
                     updates={W: W - lr * gW,
                             W_in: W_in - lr * gW_in,
                             W_out: W_out - lr * gW_out})

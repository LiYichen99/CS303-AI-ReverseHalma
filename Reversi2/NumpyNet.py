import numpy as np


def relu(x):
    return np.maximum(0, x)


def log_softmax(x):
    y = np.exp(x - np.max(x))
    y /= np.sum(y)
    return np.log(y)


def tanh(x):
    return np.tanh(x)


def get_im2col_indices(x_shape, field_height,
                       field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height,
                                 field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv2d(X, W, b, stride=1, padding=0):
    n_x, c_x, h_x, w_x = X.shape
    n_w, c_w, h_w, w_w = W.shape
    n_y = n_x
    c_y = n_w
    h_y = (h_x - h_w + 2 * padding) / stride + 1
    w_y = (w_x - w_w + 2 * padding) / stride + 1
    h_y, w_y = int(h_y), int(w_y)
    x_col = im2col_indices(X, h_w, w_w, padding=padding, stride=stride)
    w_col = W.reshape(n_w, -1)
    Y = (np.dot(w_col, x_col).T + b).T
    Y = Y.reshape(c_y, h_y, w_y, n_y)
    Y = Y.transpose(3, 0, 1, 2)
    return Y


def linear(X, W, b):
    Y = np.dot(X, W.T) + b
    return Y


def batchNorm1d(X, mean, var, gamma, beta, eps=1e-5):
    X = ((X - mean) / (np.sqrt(var + eps))) * gamma + beta
    return X


def batchNorm2d(X, mean, var, gamma, beta, eps=1e-5):
    for i in range(len(mean)):
        X[:, i, :, :] = ((X[:, i, :, :] - mean[i]) / (np.sqrt(var[i] + eps))) * gamma[i] + beta[i]
    return X


# with open("net_params.txt", 'r', encoding="utf8") as file:
#     net_dict = eval(file.read())
net_dict = {}

net_dict = {k: np.array(v) for k, v in net_dict.items()}


class PolicyValueNet(object):
    def forward(self, s):
        s = s.reshape((-1, 1, 8, 8))
        s = relu(batchNorm2d(conv2d(s, net_dict['conv1.weight'], net_dict['conv1.bias'], stride=1, padding=1),
                             mean=net_dict['bn1.running_mean'], var=net_dict['bn1.running_var'],
                             gamma=net_dict['bn1.weight'], beta=net_dict['bn1.bias']))
        s = relu(batchNorm2d(conv2d(s, net_dict['conv2.weight'], net_dict['conv2.bias'], stride=1, padding=0),
                             mean=net_dict['bn2.running_mean'], var=net_dict['bn2.running_var'],
                             gamma=net_dict['bn2.weight'], beta=net_dict['bn2.bias']))
        s = relu(batchNorm2d(conv2d(s, net_dict['conv3.weight'], net_dict['conv3.bias'], stride=1, padding=0),
                             mean=net_dict['bn3.running_mean'], var=net_dict['bn3.running_var'],
                             gamma=net_dict['bn3.weight'], beta=net_dict['bn3.bias']))
        s = s.reshape(-1, 512)
        s = relu(batchNorm1d(linear(s, net_dict['fc1.weight'], net_dict['fc1.bias']),
                             mean=net_dict['fc_bn1.running_mean'], var=net_dict['fc_bn1.running_var'],
                             gamma=net_dict['fc_bn1.weight'], beta=net_dict['fc_bn1.bias']))
        s = relu(batchNorm1d(linear(s, net_dict['fc2.weight'], net_dict['fc2.bias']),
                             mean=net_dict['fc_bn2.running_mean'], var=net_dict['fc_bn2.running_var'],
                             gamma=net_dict['fc_bn2.weight'], beta=net_dict['fc_bn2.bias']))
        pi = linear(s, net_dict['fc3.weight'], net_dict['fc3.bias'])
        v = linear(s, net_dict['fc4.weight'], net_dict['fc4.bias'])
        return log_softmax(pi), tanh(v)

    def predict(self, board):
        b = np.copy(board).reshape((1, 8, 8))
        log_pi, v = self.forward(b)
        return np.exp(log_pi)[0], v[0]



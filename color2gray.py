import os
import sys
import argparse
from numpy import linalg as LA
import numpy as np
import scipy
from scipy import io, misc, sparse
from skimage import color

PIC_DIR = 'pics'
EXTENSION = '.png'

ytop = 8
theta = np.pi / 4
v_theta = np.asarray((np.cos(theta), np.sin(theta)))

def find_neighbor(position_matrix, r):
    d = 1
    l1, h1 = max(r[0]-d, 0), min(r[0]+d, position_matrix.shape[0])
    l2, h2 = max(r[1]-d, 0), min(r[1]+d, position_matrix.shape[1])
    return position_matrix[l1:h1 + 1, l2:h2 + 1]

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def crunch(c):
    return ytop * np.tanh(c / ytop)

def calc_delta(color_lab, r, S):
    values = []
    for s in S:
        delta_l = color_lab[r][0] - color_lab[s][0]
        delta_c = color_lab[r][1:] - color_lab[s][1:]
        delta_c_norm = LA.norm(delta_c)
        crunch_value = crunch(delta_c_norm)
        product = np.dot(v_theta, delta_c)
        if abs(delta_l) > crunch_value:
            values.append(delta_l)
        elif product >= 0:
            values.append(crunch_value)
        else:
            values.append(crunch(-delta_c_norm))
    return values

def generate_delta_matrix(color_lab, neighbor_list, xy2idx, height, width):
    size = height * width
    delta =  sparse.lil_matrix((size, size))
    for i in xrange(height):
        for j in xrange(width):
            current_index = xy2idx[i, j]
            neighbors = neighbor_list[current_index]
            neighbor_indexes = [ xy2idx[pos] for pos in neighbors ]
            current_values = calc_delta(color_lab, (i, j), neighbors)
            delta[current_index, neighbor_indexes] = np.asmatrix(current_values)
    return delta

def main(arguments):
    # ---------------- Data Preparation ------------------
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', help='Specify the input picture.', type=str, default='Sunrise')
    parser.add_argument('--output', help='Specify the output name you want.', type=str, default='my_output')

    args = parser.parse_args(arguments)
    color_name = args.input
    bw_name = args.output

    in_file_path = os.path.join(PIC_DIR, color_name + EXTENSION)
    out_file_path = os.path.join(PIC_DIR, color_name + EXTENSION)

    color_rgb = misc.imread(in_file_path)
    color_lab = color.rgb2lab(color_rgb)

    (height, width) = color_rgb.shape[0:2]
    size = height * width
    # gray_matrix = np.ones((height, width), dtype='float32') * 128.0
    gray_matrix = np.average(color_rgb, axis=2)

    # ---------------- Generate delta matrix ------------------
    cart = cartesian([xrange(height), xrange(width)])
    cart_r = cart.reshape(height, width, 2) # cart_r[i, j] is [i, j]
    xy2idx = np.arange(size).reshape(height, width)
    neighbor_list = [] # list of list each element is a pos
    for i in xrange(height):
        for j in xrange(width):
            current_index = xy2idx[i, j]
            current_neighbor = find_neighbor(cart_r, [i, j]).reshape(-1, 2)
            current_neighbor = [ tuple(item) for item in current_neighbor ]
            current_neighbor.remove((i, j))
            neighbor_list.append(current_neighbor)
    delta_matrix = generate_delta_matrix(color_lab, neighbor_list, xy2idx, height, width)
    # ---------------- Optimization ------------------
    for i in xrange(height):
        for j in xrange(width):
            current_index = xy2idx[i, j]
            current_neighbor = neighbor_list[current_index]
            neighbor_indexes = [ xy2idx[pos] for pos in current_neighbor ]
            for elem in neighbor_indexes:
                print(delta_matrix[current_index, elem])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
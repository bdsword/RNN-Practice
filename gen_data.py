#!/usr/bin/env python3

import numpy as np
import random

def to_bin(x):
    return bin(x)[2:]

def main():
    num_sample = 100
    train_data = np.ndarray(shape=(8, num_sample, 2), dtype=float)
    label = np.ndarray(shape=(8, num_sample, 1), dtype=float)
    used = set()
    with open('data.npz', 'wb') as f:
        for i in range(num_sample):
            while True:
                a = random.randint(0, 255)
                b = random.randint(0, 255)
                if a + b <= 255 and (a, b) not in used and (b, a) not in used:
                    break

            a_bin = '{:0>8}'.format(to_bin(a))
            b_bin = '{:0>8}'.format(to_bin(b))
            a_b_bin = '{:0>8}'.format(to_bin(a + b))
            for j in range(8):
                train_data[j, i, :] = [int(a_bin[j]), int(b_bin[j])]
                label[j, i, 0] = int(a_b_bin[j])
        np.savez(f, samples=train_data, labels=label)

if __name__ == '__main__':
    main()
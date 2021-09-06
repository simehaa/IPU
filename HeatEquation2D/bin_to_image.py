#! /usr/bin/env python
# Copyright 2020 Graphcore Ltd.

import numpy as np
import skimage.io

import image_pb2


def main(in_file):
    out_file = in_file.replace('.bin', '')

    image_pb = image_pb2.Image()
    with open(in_file, 'rb') as in_file:
        image_pb.ParseFromString(in_file.read())

    pixels = np.fromstring(image_pb.values, dtype=np.uint8)
    pixels = pixels.reshape(image_pb.shape)
    pixels = np.transpose(pixels, axes=[1, 2, 0]) # CHW to HWC
    pixels = np.squeeze(pixels)

    skimage.io.imsave(out_file, pixels)

    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-file', type=str, default='denoised.jpg.bin',
        help='Protobuf to turn into image.')

    args = parser.parse_args()
    exit(main(in_file=args.in_file))

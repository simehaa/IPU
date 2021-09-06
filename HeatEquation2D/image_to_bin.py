#! /usr/bin/env python
# Copyright 2020 Graphcore Ltd.

import numpy as np
import skimage.io

import image_pb2


def main(in_file):
    image = skimage.io.imread(in_file)

    if image.ndim < 3:
        image = np.expand_dims(image, axis=-1)
    image = np.transpose(image, axes=[2, 0, 1]) # HWC to CHW
    image = image.mean(axis=0, keepdims=True).astype(image.dtype)

    image_pb = image_pb2.Image(
        shape=list(image.shape),
        values=image.tostring()
    )

    with open('{}.bin'.format(in_file), 'wb') as out_file:
       out_file.write(image_pb.SerializeToString())

    return 0


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-file', type=str, default='mona_lisa_noisy.jpg',
        help='Image to turn into Protobuf.')

    args = parser.parse_args()
    exit(main(in_file=args.in_file))

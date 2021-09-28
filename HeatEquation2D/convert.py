import argparse
import numpy as np
import skimage.io
import image_pb2


def bin_to_jpg(in_file):
    out_file = in_file.replace(".bin", "")
    image_pb = image_pb2.Image()
    with open(in_file, "rb") as in_file:
        image_pb.ParseFromString(in_file.read())
    pixels = np.fromstring(image_pb.values, dtype=np.uint8)
    pixels = pixels.reshape(image_pb.shape)
    pixels = np.transpose(pixels, axes=[1, 2, 0])  # CHW to HWC
    pixels = np.squeeze(pixels)
    skimage.io.imsave(out_file, pixels)
    return None


def jpg_to_bin(in_file):
    out_file = in_file.replace(".jpg", "")
    image = skimage.io.imread(in_file)
    if image.ndim < 3:
        image = np.expand_dims(image, axis=-1)
    image = np.transpose(image, axes=[2, 0, 1])  # HWC to CHW
    image = image.mean(axis=0, keepdims=True).astype(image.dtype)
    image_pb = image_pb2.Image(shape=list(image.shape), values=image.tostring())
    with open("{}.bin".format(in_file), "wb") as out_file:
        out_file.write(image_pb.SerializeToString())
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file",
        type=str,
        default="denoised.jpg.bin",
        help="File to convert: either jpg->bin or bin->jpg.",
    )
    args = parser.parse_args()
    in_file = args.in_file
    file_type = in_file.split(".")[-1]
    if file_type == "bin":
        print(f"Converting {in_file} to a jpg image.")
        bin_to_jpg(in_file)
    elif file_type == "jpg":
        print(f"Converting {in_file} to a binary file.")
        jpg_to_bin(in_file)
    else:
        print(
            "Error: provide --in-file <yyyy.xxx>, where xxx is either 'jpg' or 'bin'. "
            f"The given file type '.{file_type}' is not supported by this program."
        )
import numpy as np
import scipy
import pydicom
import torch
import os

from matplotlib import pylab as plt


def find_all_files(path: str):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                yield os.path.join(dirname, filename)
    else:
        yield path


def read_series(path: str):
    series = dict()
    for p in find_all_files(path):
        try:
            dc = pydicom.dcmread(p)
            sop = str(dc[0x8, 0x16].value)
            imtype = dc[0x8, 0x8].value
        except Exception as e:
            print(e)
        else:
            if (
                sop == "1.2.840.10008.5.1.4.1.1.2"
                and imtype[0] == "ORIGINAL"
                and imtype[1] == "PRIMARY"
                and imtype[2] == "AXIAL"
            ):
                suid = dc[0x20, 0xE].value
                if suid not in series:
                    series[suid] = list()
                series[suid].append(dc)
    for dcs in series.values():
        dcs.sort(key=lambda x: x[0x20, 0x32].value[2])
    return series


def get_interference_batches(series):
    arr = np.array(
        [
            pydicom.pixel_data_handlers.apply_modality_lut(dc.pixel_array, dc)
            for dc in series
        ],
        dtype=np.float32,
    )

    spacing_xy = list(series[0][0x28, 0x30].value)
    spacing_z = np.sqrt(
        np.sum(
            (
                np.array(series[0][0x20, 0x32].value)
                - np.array(series[1][0x20, 0x32].value)
            )
            ** 2
        )
    )
    spacing_in = np.array(
        [
            spacing_z,
        ]
        + spacing_xy
    )

    spacing_out = np.ones(3) * 1.5
    mat = np.zeros((3, 3))
    mat[0, 0] = spacing_out[0] / spacing_in[0]
    mat[2, 1] = spacing_out[1] / spacing_in[1]
    mat[1, 2] = spacing_out[2] / spacing_in[2]

    shape_in = np.array(arr.shape)

    shape_out = np.ceil(shape_in * spacing_in / spacing_out).astype(int)
    shape_out = (shape_out[0], 384, 384)

    arr_out = scipy.ndimage.affine_transform(
        arr, mat, cval=-1024, output_shape=shape_out
    )[:, :, ::-1]

    # plt.imshow(arr_out[0,:,:])
    # plt.show()
    for i in range(0, shape_out[0], 32):
        ten_in = torch.full((32, 1, 384, 384), -1024, dtype=torch.float32)
        next = min(i + 32, shape_out[0])
        lim = next - i
        ten_in.numpy()[:lim, 0, :, :] = arr_out[i:next, :, :]
        ten_in.add_(1024)
        ten_in.div_(2048)
        ten_in.clip_(0, 1)
        yield ten_in


def inference(series: list, model_path: str):
    models_path = list(
        [os.path.join(model_path, "freezed_model{}.pt".format(i)) for i in range(1, 5)]
    )
    for ten_in in get_interference_batches(series):
        labels = np.zeros_like(ten_in)
        with torch.no_grad():
            for mind, mpath in enumerate(models_path):
                model = torch.jit.load(mpath).eval()
                ten_out = model(ten_in).cpu().numpy()
                for i in range(1, ten_out.shape[1]):
                    labels[:, 0, :, :] = np.where(
                        ten_out[:, i, :, :] > 0.90, i + mind * 15, labels[:, 0, :, :]
                    )
                    # labels[:, 0, :, :] += (ten_out[:, i, :, :] > 0.90) * i

            for i in range(ten_in.shape[0]):
                plt.subplot(1, 2, 1)
                plt.imshow(ten_in[i, 0, :, :], cmap="bone")
                plt.subplot(1, 2, 2)
                plt.imshow(labels[i, 0, :, :])
                plt.show()


if __name__ == "__main__":
    torch.set_num_threads(12)
    path = r"C:\Users\ander\OneDrive\image_sim\thorax"

    model_path = r"C:\Users\ander\OneDrive\ctseg\models"
    for key, series in read_series(path).items():
        inference(series, model_path)

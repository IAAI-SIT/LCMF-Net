import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import nibabel as nib

flair_name = '-t2f.nii'
t1_name = '-t1n.nii'
t1ce_name = '-t1c.nii'
t2_name = '-t2w.nii'
mask_name = '-seg.nii'

brats_path = '/Dataset/visualize'
outputImg_path = r'/Dataset/visualize2'
outputMask_path = r'/Dataset/visualize2/Mask'

if not os.path.exists(outputImg_path):
    os.mkdir(outputImg_path)
if not os.path.exists(outputMask_path):
    os.mkdir(outputMask_path)

output_flair_path = os.path.join(outputImg_path, "flair")
output_t1_path = os.path.join(outputImg_path, "t1")
output_t1ce_path = os.path.join(outputImg_path, "t1ce")
output_t2_path = os.path.join(outputImg_path, "t2")

for path in [output_flair_path, output_t1_path, output_t1ce_path, output_t2_path]:
    if not os.path.exists(path):
        os.mkdir(path)



def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return: dir or file
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp


from scipy.ndimage import zoom
# def crop_ceter(img, croph, cropw):
#     # for n_slice in range(img.shape[0]):
#     height, width = img[0].shape
#     starth = height // 2 - (croph // 2)
#     startw = width // 2 - (cropw // 2)
#     return img[:, starth:starth + croph, startw:startw + cropw]

def crop_ceter(array, croph, cropw):

    if array.shape != (155, 240, 240):
        raise ValueError("输入数组的形状必须为 (155, 240, 240)")

    zoom_factors = (1, croph / 240, cropw / 240)

    resized_array = zoom(array, zoom_factors, order=1)
    print(resized_array.shape)
    return resized_array
def crop_ceter_mask(array, croph, cropw):

    if array.shape != (155, 240, 240):
        raise ValueError("输入数组的形状必须为 (155, 240, 240)")

    zoom_factors = (1, croph / 240, cropw / 240)

    resized_array = zoom(array, zoom_factors, order=0)
    print(resized_array.shape)
    return resized_array


def get_data(data_flies):
    # for subsetindex in range(len(data_flies)):
    for subsetindex in tqdm(range(len(data_flies)), total=len(data_flies)):
        brats_subset_path = brats_path + "/" + str(data_flies[subsetindex]) + "/"
        flair_image = brats_subset_path + str(data_flies[subsetindex]) + flair_name
        t1_image = brats_subset_path + str(data_flies[subsetindex]) + t1_name
        t1ce_image = brats_subset_path + str(data_flies[subsetindex]) + t1ce_name
        t2_image = brats_subset_path + str(data_flies[subsetindex]) + t2_name
        mask_image = brats_subset_path + str(data_flies[subsetindex]) + mask_name
        flair_src = sitk.ReadImage(flair_image, sitk.sitkInt16)
        t1_src = sitk.ReadImage(t1_image, sitk.sitkInt16)
        t1ce_src = sitk.ReadImage(t1ce_image, sitk.sitkInt16)
        t2_src = sitk.ReadImage(t2_image, sitk.sitkInt16)
        mask = sitk.ReadImage(mask_image, sitk.sitkUInt8)
        flair_array = sitk.GetArrayFromImage(flair_src)
        t1_array = sitk.GetArrayFromImage(t1_src)
        t1ce_array = sitk.GetArrayFromImage(t1ce_src)
        t2_array = sitk.GetArrayFromImage(t2_src)
        mask_array = sitk.GetArrayFromImage(mask)
        # flair_array_nor = normalize(flair_array)
        # t1_array_nor = normalize(t1_array)
        # t1ce_array_nor = normalize(t1ce_array)
        # t2_array_nor = normalize(t2_array)
        flair_crop = crop_ceter(flair_array, 160, 160)
        t1_crop = crop_ceter(t1_array, 160, 160)
        t1ce_crop = crop_ceter(t1ce_array, 160, 160)
        t2_crop = crop_ceter(t2_array, 160, 160)
        mask_crop = crop_ceter_mask(mask_array, 160, 160)

        for n_slice in range(flair_crop.shape[0]):
            maskImg = mask_crop[n_slice, :, :]
            flairImg = flair_crop[n_slice, :, :]
            flairImg = flairImg.astype(np.float32)
            t1Img = t1_crop[n_slice, :, :]
            t1Img = t1Img.astype(np.float32)
            t1ceImg = t1ce_crop[n_slice, :, :]
            t1ceImg = t1ceImg.astype(np.float32)
            t2Img = t2_crop[n_slice, :, :]
            t2Img = t2Img.astype(np.float32)
            flair_imagepath = os.path.join(output_flair_path, f"{data_flies[subsetindex]}_{n_slice}.npy")
            t1_imagepath = os.path.join(output_t1_path, f"{data_flies[subsetindex]}_{n_slice}.npy")
            t1ce_imagepath = os.path.join(output_t1ce_path, f"{data_flies[subsetindex]}_{n_slice}.npy")
            t2_imagepath = os.path.join(output_t2_path, f"{data_flies[subsetindex]}_{n_slice}.npy")
            maskpath = os.path.join(outputMask_path, f"{data_flies[subsetindex]}_{n_slice}.npy")

            # flair_imagepath = outputImg_path + "\\" + "flair" + "\\" + str(data_flies[subsetindex]) + "_" + str(
            #     n_slice) + ".npy"
            # t1_imagepath = outputImg_path + "\\" + "t1" + "\\" + str(data_flies[subsetindex]) + "_" + str(
            #     n_slice) + ".npy"
            # t1ce_imagepath = outputImg_path + "\\" + "t1ce" + "\\" + str(data_flies[subsetindex]) + "_" + str(
            #     n_slice) + ".npy"
            # t2_imagepath = outputImg_path + "\\" + "t2" + "\\" + str(data_flies[subsetindex]) + "_" + str(
            #     n_slice) + ".npy"
            # maskpath = outputMask_path + "\\" + str(data_flies[subsetindex]) + "_" + str(n_slice) + ".npy"
            np.save(flair_imagepath, flairImg)  # (170,170,4) np.float dtype('float32')
            np.save(t1_imagepath, t1Img)  # (170,170,4) np.float dtype('float32')
            np.save(t1ce_imagepath, t1ceImg)  # (170,170,4) np.float dtype('float32')
            np.save(t2_imagepath, t2Img)  # (170,170,4) np.float dtype('float32')
            np.save(maskpath, maskImg)  # (170, 170) dtype('uint8') 值为0 1 2 4

    print("Done！")


if __name__ == '__main__':
    data_list = file_name_path('/Dataset/visualize')
    get_data(data_list)

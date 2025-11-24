import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage.interpolation import rotate, shift
from glob import glob

def extract_and_increase_number(file_name,slice_num):
    parts = file_name.rsplit("_", 1)
    slice_name = parts[0].rsplit("/", 1)[1]
    end = parts[1].rsplit(".", 1)
    result = []

    number = int(end[0])
    for i in range(slice_num):
        target_number = number - (i + 1)
        if target_number < 0:
            target_number = 0
        elif target_number > 154:
            target_number = 154
        target_end = str(target_number) + "." + end[1]
        target_name = slice_name + "_" + target_end
        result.append(target_name)

    for i in range(slice_num):
        target_number = number + (i + 1)
        if target_number < 0:
            target_number = 0
        elif target_number > 154:
            target_number = 154
        target_end = str(target_number) + "." + end[1]
        target_name = slice_name + "_" + target_end
        result.append(target_name)

    return result


def channel(patient_label):
    et = patient_label == 3
    tc = torch.logical_or(patient_label == 1, patient_label == 3)
    wt = torch.logical_or(tc, patient_label == 2)
    patient_label = torch.stack([wt, tc, et])
    return patient_label


def random_rotate(flair, t1, t1ce, t2, nplabel, max_angle=30):

    angle = random.uniform(-max_angle, max_angle)
    rotated_flair_image = rotate(flair, angle, axes=(1, 2), reshape=False, order=3)
    rotated_t1_image = rotate(t1, angle, axes=(1, 2), reshape=False, order=3)
    rotated_t1ce_image = rotate(t1ce, angle, axes=(1, 2), reshape=False, order=3)
    rotated_t2_image = rotate(t2, angle, axes=(1, 2), reshape=False, order=3)
    rotated_segmentation = rotate(nplabel, angle, axes=(1, 2), reshape=False, order=0)
    return rotated_flair_image, rotated_t1_image, rotated_t1ce_image, rotated_t2_image, rotated_segmentation


def add_noise(image, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + noise
    return noisy_image

def safe_load_modality(path):
    try:
        return np.load(path).astype("float32")
    except Exception:
        return None


class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, t1='', t1ce='', t2='', slice_num=2, aug=False,FLAIR='FLAIR'):
        self.args = args
        self.flairimg_paths = sorted(img_paths)
        self.t1img_paths = t1
        self.t1ceimg_paths = t1ce
        self.t2img_paths = t2
        self.mask_paths = sorted(mask_paths)
        self.aug = aug
        self.split = None
        self.slice_num = slice_num
        self.FLAIRMissing = FLAIR

    def __len__(self):
        return len(self.flairimg_paths)

    def __getitem__(self, idx):
        img_path = self.flairimg_paths[idx]
        mask_path = self.mask_paths[idx]
        npmask = np.load(mask_path)
        tensor_mask = torch.from_numpy(npmask)
        getlabel = channel(tensor_mask)
        nplabel = getlabel.numpy()
        nplabel = nplabel.astype("float32")

        slice_name = img_path.rsplit("/", 1)[1]
        flair_base = img_path.rsplit("/", 1)[0]
        result = extract_and_increase_number(img_path, self.slice_num)

        flair = np.empty(((self.slice_num * 2) + 1, 160, 160))
        t1 = np.empty(((self.slice_num * 2) + 1, 160, 160))
        t1ce = np.empty(((self.slice_num * 2) + 1, 160, 160))
        t2 = np.empty(((self.slice_num * 2) + 1, 160, 160))
        if self.FLAIRMissing =="FLAIR":
            modalities = {
            "flair": safe_load_modality(self.flairimg_paths[idx]),
            "t1": safe_load_modality(self.t1img_paths + slice_name),
            "t1ce": safe_load_modality(self.t1ceimg_paths + slice_name),
            "t2": safe_load_modality(self.t2img_paths + slice_name)
            }
        else:
            modalities = {
            't1': safe_load_modality(self.flairimg_paths[idx]),
            "flair": None,
            "t1ce": safe_load_modality(self.t1ceimg_paths + slice_name),
            "t2": safe_load_modality(self.t2img_paths + slice_name)
            }
        valid_modalities = [v for v in modalities.values() if v is not None]
        mean_image = np.mean(valid_modalities, axis=0)
        for name in modalities:
            image = modalities[name]
            if image is None:
                modalities[name] = mean_image
            eval(name)[2, :, :] = modalities[name]

        point = 0
        for i in range(self.slice_num * 2):
            if point == 2:
                point += 1
            if self.FLAIRMissing =="FLAIR":
                slice_modalities = {
                "flair": safe_load_modality(flair_base + "/" + result[i]),
                "t1": safe_load_modality(self.t1img_paths + result[i]),
                "t1ce": safe_load_modality(self.t1ceimg_paths + result[i]),
                "t2": safe_load_modality(self.t2img_paths + result[i])
                }
            else:
                slice_modalities = {
                't1': safe_load_modality(flair_base + "/" + result[i]),
                "flair": None,
                "t1ce": safe_load_modality(self.t1ceimg_paths + result[i]),
                "t2": safe_load_modality(self.t2img_paths + result[i])
                }
            valid_slices = [v for v in slice_modalities.values() if v is not None]
            if not valid_slices:
                raise RuntimeError(f"All optional modalities missing for {result[i]}")

            mean_slice = np.mean(valid_slices, axis=0)

            for name in slice_modalities:
                image = slice_modalities[name]
                if image is None:
                    slice_modalities[name] = mean_slice
                eval(name)[point, :, :] = slice_modalities[name]
            point += 1

        if self.aug:
            flair, t1, t1ce, t2, nplabel = random_rotate(flair, t1, t1ce, t2, nplabel)
            flair = add_noise(flair)
            t1 = add_noise(t1)
            t1ce = add_noise(t1ce)
            t2 = add_noise(t2)

        nplabel = nplabel.astype("float32")
        flair = flair.astype("float32")
        t1 = t1.astype("float32")
        t1ce = t1ce.astype("float32")
        t2 = t2.astype("float32")
        tensor_label = torch.from_numpy(nplabel)
        tensor_flair_image = torch.from_numpy(flair)
        tensor_t1_image = torch.from_numpy(t1)
        tensor_t1ce_image = torch.from_numpy(t1ce)
        tensor_t2_image = torch.from_numpy(t2)

        sample = {'flair_image': tensor_flair_image, 't1_image': tensor_t1_image, 't1ce_image': tensor_t1ce_image,
                  't2_image': tensor_t2_image, 'label': tensor_label}

        return sample


if __name__ == '__main__':
    img_paths = glob(r'/Dataset/Brats2023MEN/flair/*')
    t1img_paths = '/Dataset/Brats2023MEN/t1/'
    t1ceimg_paths = '/Dataset/Brats2023MEN/t1ce/'
    t2img_paths = '/Dataset/Brats2023MEN/t2/'
    mask_paths = glob(r'/Dataset/Brats2023MEN/Mask/*')

    img_paths = sorted(img_paths)
    mask_paths = sorted(mask_paths)

    train_dataset = Dataset(None, img_paths=img_paths, mask_paths=mask_paths,
                            t1=t1img_paths, t1ce=t1ceimg_paths, t2=t2img_paths,
                            slice_num=2, aug=False)
    print(len(train_dataset))


    for idx in range(min(10, len(train_dataset))):
        sample = train_dataset[idx]

import numpy as np
import torch.utils.data
import scipy.ndimage

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = sorted(img_paths)
        self.mask_paths = sorted(mask_paths)
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    # 根据npmask数组的不同取值，生成三个不同的标签数组，分别代表整个肿瘤区域（WT）、肿瘤核心区域（TC）和增强区域（ET）。同时，将这些标签数组合并到一个形状为(3, 160, 160)的数组中
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        # 读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 1.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 0.
        nplabel = np.empty((176, 176, 3))

        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))
        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")
        # 使用 scipy.ndimage.zoom 进行缩放
        # zoom_factor = [1, 256 / 240, 256 / 240]
        # nplabel = scipy.ndimage.zoom(nplabel, zoom_factor, order=3)  # order=3 表示三次插值
        # npimage = scipy.ndimage.zoom(npimage, zoom_factor, order=3)  # order=3 表示三次插值
        tensor_label = torch.from_numpy(nplabel)
        tensor_image = torch.from_numpy(npimage)
        return tensor_image, tensor_label

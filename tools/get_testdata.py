import os
from glob import glob

import torch
from sklearn.model_selection import train_test_split


def main():
    torch.manual_seed(21)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(21)  # 为所有的GPU设置种子，以使得结果是确定的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    img_paths = glob(r'E:\zJuny\my_dataset\trainImage\*')
    mask_paths = glob(r'E:\zJuny\my_dataset\trainMask\*')
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print("train_num:%s" % str(len(train_img_paths)))
    print("val_num:%s" % str(len(val_img_paths)))


if __name__ == '__main__':
    main()
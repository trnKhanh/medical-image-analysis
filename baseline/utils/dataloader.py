import os.path
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class DatasetSegmentation(Dataset):
    """
    We didn't use semi-supervised learning, this Dataset Class just load labelled datas.
    If you want to try semi-supervised learning to make full use of unlabelled datas, please design your own Dataset Class!
    """
    def __init__(self, dir, ls, transform=None):
        self.transform = transform  # using transform in torch!
        self.dir = dir

        labels = []
        images = []
        files = os.listdir(os.path.join(dir, "images"))
        for file in files:
            if file in ls:
                image_path = os.path.join(dir, "images", file)
                label_path = os.path.join(dir, "labels", file)
                image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(image).transpose(2,0,1)
                label = sitk.ReadImage(label_path)
                label = sitk.GetArrayFromImage(label)

                images.append(np.array([image]))
                labels.append(np.array([label]))

        images = np.concatenate(images, axis=0)
        labels = np.concatenate(labels, axis=0)
        self.images = images
        self.labels = labels
        print(f"Image:{self.images.shape}\tLabel:{self.labels.shape}")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = correct_dims(self.images[idx])
        label = np.array([self.labels[idx]])
        sample = {}
        if self.transform:
            image, label = self.transform(image, label)

        sample['image'] = image
        sample['label'] = label
        return sample


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from augmentation import JointTransform2D

    tf = JointTransform2D(p_flip=0, crop=None,long_mask=True)
    ls = os.listdir("../dataset/train/images")
    dataset = DatasetSegmentation("../dataset/train", ls, tf)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for idx, sample in enumerate(dataloader):
        if idx == 1:
            image = sample['image']
            label = sample['label']
            print(image.shape, label.shape)
            print(image.dtype,label.dtype)
            """
            plt.imshow(image[0][0],cmap="gray")
            plt.show()
            plt.imshow(label[0], cmap="gray")
            plt.show()
            """
            break
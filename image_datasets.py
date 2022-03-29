import os
import sys
import torch
import json
import numpy as np

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

rgb = [2, 1, 0]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }


mask_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=Image.NEAREST),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])


def mask_process(dim):
    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256, interpolation=Image.NEAREST),
        transforms.CenterCrop(224),
        transforms.Resize(dim, interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    return mask_transform


class VisualGenome(Dataset):

    def __init__(self, root_dir=None, transform=None, max_batch_size=64, mask_dim=7):
        self._root_dir = root_dir
        self._transform = transform
        self._samples = self._load_obj()
        self._labels = self._load_labels()
        self._max_batch_size = max_batch_size
        self._mask_dim = mask_dim

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path = 'data/vg/VG_100K/%d.jpg' % self._samples[idx]['image_id']
        path = os.path.join(self._root_dir, path)

        ori_img = Image.open(path).convert('RGB')
        if self._transform is not None:
            img = self._transform(ori_img)
        else:
            img = ori_img

        targets = []
        masks = []
        for obj in self._samples[idx]['objects']:
            target = torch.zeros((len(self._labels),))
            if obj in self._labels:
                label = self._labels[obj]
                target[label] = 1
            box_mask = torch.zeros(ori_img.size)
            for box_anno in self._samples[idx]['objects'][obj]:
                xmin = box_anno['x']
                xmax = box_anno['x'] + box_anno['w']
                ymin = box_anno['y']
                ymax = box_anno['y'] + box_anno['h']
                box_mask[ymin:ymax, xmin:xmax] = 1
            targets.append(target)
            mask_transform = mask_process(self._mask_dim)
            box_mask = mask_transform(box_mask)
            masks.append(box_mask)

        return img, torch.stack(targets), torch.stack(masks)

    def _load_obj(self):
        dataFile = os.path.join(self._root_dir, 'vg/vg_objects.json')
        with open(dataFile) as f:
            data = json.load(f)
        return data

    def _load_labels(self):
        dataFile = os.path.join(self._root_dir, 'vg/vg_labels.pkl')
        with open(dataFile, 'rb') as f:
            data = pickle.load(f)
        return data


class MyCocoDetection(CocoDetection):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        target_list = []
        for t in target:
            mask = coco.annToMask(t)
            im = Image.fromarray(mask)
            mask = np.array(im.resize((224, 224)))
            obj = list(label_embedding['stoi'].keys())[t['category_id'] - 1]
            idxs = list(label_embedding['stoi'].values())[t['category_id'] - 1]
            mask_dict = {'mask': mask, 'object': obj, 'idx': idxs}
            target_list.append(mask_dict)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target_list


class MyCocoSegmentation(CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.anns.keys()))
        self.dim = 7

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        ann_id = self.ids[index]
        ann = self.coco.anns[ann_id]
        img_id = ann['image_id']

        mask = coco.annToMask(ann)
        label_embedding_file = "./data/coco/coco_label_embedding.pth"
        label_embedding = torch.load(label_embedding_file)
        target_onehot = torch.zeros((len(label_embedding['itos']),))
        idxs = list(label_embedding['stoi'].values())[ann['category_id'] - 1]
        for idx in idxs:
            i = list(label_embedding['itos'].keys()).index(idx)
            target_onehot[i] = 1

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        mask_transform = mask_process(self.dim)
        mask = mask_transform(mask)

        return img, target_onehot, mask



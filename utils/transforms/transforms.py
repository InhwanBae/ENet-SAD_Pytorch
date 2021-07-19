import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th


class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  #(W, H)

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_CUBIC)
        if segLabel is not None:
            segLabel = cv2.resize(segLabel, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size


class RandomResize(Resize):
    """
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH)
    """
    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW+1)
        h = np.random.randint(self.minH, self.maxH+1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        degree = (u-0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if segLabel is not None:
            segLabel = cv2.warpAffine(segLabel, R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta


class HorizontalFlip(CustomTransform):
    def __init__(self, probability, relabel=True):
        self.probability = probability
        self.relabel = relabel

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        if u < self.probability:
            img = cv2.flip(img, 1)
            if segLabel is not None:
                segLabel = cv2.flip(segLabel, 1)
                if self.relabel:
                    segLabel = 5 - segLabel
                    segLabel[segLabel == 5] = 0  # bg

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_probability(self, probability):
        self.probability = probability


class RandomCrop(CustomTransform):
    def __init__(self, minWratio, maxWratio, minHratio=None, maxHratio=None):
        self.minWratio = minWratio
        self.maxWratio = maxWratio
        self.minHratio = minHratio if minHratio is not None else minWratio
        self.maxHratio = maxHratio if maxHratio is not None else maxWratio

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img_w = img.shape[1]
        img_h = img.shape[0]

        u = np.random.uniform()
        v = np.random.uniform()

        out_w = int(img_w * (self.minWratio + (self.maxWratio - self.minWratio) * u))
        out_h = int(img_h * (self.minHratio + (self.maxHratio - self.minHratio) * v))

        x = np.random.uniform()
        y = np.random.uniform()

        if out_w > img_w:
            temp = out_w - img_w
            img = cv2.copyMakeBorder(img, 0, 0, int(temp * x), int(temp * (1 - x)), cv2.BORDER_CONSTANT)
            if segLabel is not None:
                segLabel = cv2.copyMakeBorder(segLabel, 0, 0, int(temp * x), int(temp * (1 - x)), cv2.BORDER_CONSTANT)
        else:
            temp = img_w - out_w
            img = img[0:img_h, int(temp * x):int(img_w - temp * (1 - x))]
            if segLabel is not None:
                segLabel = segLabel[0:img_h, int(temp * x):int(img_w - temp * (1 - x))]

        if out_h > img_h:
            temp = out_h - img_h
            img = cv2.copyMakeBorder(img, int(temp * y), int(temp * (1 - y)), 0, 0, cv2.BORDER_CONSTANT)
            if segLabel is not None:
                segLabel = cv2.copyMakeBorder(segLabel, int(temp * y), int(temp * (1 - y)), 0, 0, cv2.BORDER_CONSTANT)
        else:
            temp = img_h - out_h
            img = img[int(temp * y):int(img_h - temp * (1 - y)), 0:out_w]
            if segLabel is not None:
                segLabel = segLabel[int(temp * y):int(img_h - temp * (1 - y)), 0:out_w]

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_ratio(self, minWratio, maxWratio, minHratio=None, maxHratio=None):
        self.minWratio = minWratio
        self.maxWratio = maxWratio
        self.minHratio = minHratio if minHratio is not None else minWratio
        self.maxHratio = maxHratio if maxHratio is not None else maxWratio


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        img = sample.get('img')

        img = self.transform(img)

        _sample = sample.copy()
        _sample['img'] = img
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        exist = sample.get('exist', None)

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype) / 255.
        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)
        if exist is not None:
            exist = torch.from_numpy(exist).type(torch.float32)  # BCEloss requires float tensor

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        _sample['exist'] = exist
        return _sample



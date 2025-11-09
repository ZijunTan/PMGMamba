import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, image_d, image_g, label):
        for t in self.transforms:
            image, image_d, image_g, label = t(image, image_d, image_g, label)
        return image, image_d, image_g, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, img_d, img_g, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(img_d), F.hflip(img_g), F.hflip(label)
        return img, img_d, img_g, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, pic_d, pic_g,label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(pic_d), F.to_tensor(pic_g), F.to_tensor(label)

class PairResize(transforms.Resize):
    def __call__(self, img, img_d, img_g, label):
        return F.resize(img, (self.size, self.size)), F.resize(img_d, (self.size, self.size)), \
               F.resize(img_g, (self.size, self.size)), F.resize(label, (self.size, self.size))


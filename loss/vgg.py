import torch
import torch.nn as nn
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGG19_relu().to(device)
        self.criterion = nn.MSELoss()
        self.weights = [1.0 / 64, 1.0 / 64, 1.0 / 32, 1.0 / 32, 1.0 / 1]
        self.IN = nn.InstanceNorm2d(512, affine=False, track_running_stats=False)

    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = self.weights[0] * self.criterion(self.IN(x_vgg['relu1_1']), self.IN(y_vgg['relu1_1']))
        loss += self.weights[1] * self.criterion(self.IN(x_vgg['relu2_1']), self.IN(y_vgg['relu2_1']))
        loss += self.weights[2] * self.criterion(self.IN(x_vgg['relu3_1']), self.IN(y_vgg['relu3_1']))
        loss += self.weights[3] * self.criterion(self.IN(x_vgg['relu4_1']), self.IN(y_vgg['relu4_1']))
        loss += self.weights[4] * self.criterion(self.IN(x_vgg['relu5_1']), self.IN(y_vgg['relu5_1']))

        return loss


class VGG19_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        cnn = torchvision.models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1')
        cnn = cnn
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {'relu1_1': relu1_1, 'relu1_2': relu1_2, 'relu2_1': relu2_1, 'relu2_2': relu2_2,
               'relu3_1': relu3_1, 'relu3_2': relu3_2, 'relu3_3': relu3_3, 'relu3_4': relu3_4,
               'relu4_1': relu4_1, 'relu4_2': relu4_2, 'relu4_3': relu4_3, 'relu4_4': relu4_4,
               'relu5_1': relu5_1, 'relu5_2': relu5_2, 'relu5_3': relu5_3, 'relu5_4': relu5_4}

        return out



if __name__ == '__main__':
    model = PerceptualLoss()
    x = torch.randn((2, 3, 256, 256))
    y = torch.randn((2, 3, 256, 256))
    out = model(x, y)
    # for i in out:
    #     print(i.shape)


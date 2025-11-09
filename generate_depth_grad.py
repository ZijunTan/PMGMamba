import numpy as np
from scipy.ndimage import median_filter, minimum_filter
import os
import cv2
import math



def atmLight(I, DepthMap):
    height, width, _ = I.shape
    imsize = width * height
    JDark = DepthMap
    numpx = max(math.floor(imsize / 1000), 1)  # 至少取1个像素
    JDarkVec = JDark.reshape(-1)
    ImVec = I.reshape(-1, 3)

    # 取深度最大的前0.1%像素
    indices = np.argsort(JDarkVec)[-numpx:]
    A = np.mean(ImVec[indices], axis=0)
    return A


def calcTrans(I, BL, win):
    h, w, _ = I.shape
    BLmap = np.tile(BL, (h, w, 1))
    DLmap = np.abs(BLmap - I)

    # 归一化
    Bm = np.maximum(BL, 1 - BL)
    maxDL = np.zeros_like(DLmap)
    for ind in range(3):
        maxDL[:, :, ind] = median_filter(DLmap[:, :, ind] / Bm[ind], size=win)

    # 取三通道最大值并中值滤波
    maxDL1 = np.max(DLmap / Bm, axis=2)
    T = median_filter(maxDL1, size=win)
    return np.clip(T, 0, 1)


def getColorCast(im):
    L, a, b = getRGB2Lab(im[:, :, 0], im[:, :, 1], im[:, :, 2])
    var_ansa = np.var(a)
    var_ansb = np.var(b)
    var_sq = np.sqrt(var_ansa + var_ansb)
    u = np.sqrt(np.mean(a)**2 + np.mean(b)**2)
    D = u - var_sq
    Dl = D / var_sq
    return D, Dl


def CC(I):
    h, w, _ = I.shape
    Ivec = I.reshape(-1, 3)
    avgIC = np.mean(Ivec, axis=0)
    _, Dl = getColorCast(I)

    if Dl <= 0:
        return np.array([1, 1, 1])
    else:
        return (np.maximum(np.max(avgIC), 0.1) / np.maximum(avgIC, 0.1)) ** (1 / max(np.sqrt(Dl), 1))



def getGrad(I, win):
    I = I.astype(np.float32)
    imYUV = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)
    Gmag = cv2.Sobel(imYUV[:, :, 0], cv2.CV_64F, 1, 1, ksize=3)

    # 膨胀操作
    kernel = np.ones((win, win), np.uint8)
    imG = cv2.dilate(Gmag, kernel)

    # 归一化到[0, 1]
    return (imG - np.min(imG)) / (np.max(imG) - np.min(imG))


def GetDepth(I, win):
    GradMap = getGrad(I, win)
    RoughDepthMap = 1 - GradMap

    # 线性拟合深度与像素值关系
    DepVec = RoughDepthMap.reshape(-1)
    ImVec = I.reshape(-1, 3)
    A = np.column_stack([DepVec ** 0, DepVec ** 1])
    w = np.linalg.lstsq(A, ImVec, rcond=None)[0]

    ws = np.tanh(4 * np.abs(w[1, :]))
    s = (w[1, :] <= 0).astype(float)
    min_im = np.minimum.reduce([
        ws[0] * minimum_filter(np.abs(s[0] - I[:, :, 0]), size=win) + (1 - ws[0]),
        ws[1] * minimum_filter(np.abs(s[1] - I[:, :, 1]), size=win) + (1 - ws[1]),
        ws[2] * minimum_filter(np.abs(s[2] - I[:, :, 2]), size=win) + (1 - ws[2])
    ])
    return min_im, GradMap


def getRGB2Lab(R, G, B):
    if R.ndim == 3:  # 处理单张图像输入
        B = R[:, :, 2]
        G = R[:, :, 1]
        R = R[:, :, 0]

    RGB = np.stack([R, G, B], axis=2)
    if np.max(RGB) > 1:
        RGB = RGB / 255.0

    # RGB转XYZ
    MAT = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])
    XYZ = np.dot(RGB.reshape(-1, 3), MAT.T).reshape(RGB.shape)

    # XYZ转Lab
    X, Y, Z = XYZ[:, :, 0] / 0.950456, XYZ[:, :, 1], XYZ[:, :, 2] / 1.088754
    T = 0.008856
    f = lambda x: np.where(x > T, x ** (1 / 3), 7.787 * x + 16 / 116)
    L = np.where(Y > T, 116 * Y ** (1 / 3) - 16, 903.3 * Y)
    a = 500 * (f(X) - f(Y))
    b = 200 * (f(Y) - f(Z))
    return L, a, b



def process_image(im_path, win=15, t0=0.2):
    im = cv2.imread(im_path).astype(np.float32) / 255
    s = CC(im)
    DepthMap, GradMap = GetDepth(im, win)
    A = atmLight(im, DepthMap)
    T = calcTrans(im, A, win)

    # 调整传输系数范围
    T_pro = (T - np.min(T)) / (np.max(T) - np.min(T)) * (np.max(T) - t0) + t0
    Jc = np.zeros_like(im)
    for ind in range(3):
        Am = A[ind] / s[ind]
        Jc[:, :, ind] = Am + (im[:, :, ind] - Am) / np.maximum(T_pro, t0 * 1.5)

    Jc = np.clip(Jc, 0, 1)
    return Jc, T_pro, DepthMap, GradMap


root = r"E:\dataset\U45\raw"
for idx, name in enumerate(os.listdir(root)):
    print(f'{idx+1}/{len(os.listdir(root))}, {name}')
    im_path = root + '/' + name
    Jc, T_pro, DepthMap, GradMap = process_image(im_path)
    save_D = root + '_depth'
    save_G = root + '_grad'
    if not os.path.exists(save_D):
        os.makedirs(save_D)
    if not os.path.exists(save_G):
        os.makedirs(save_G)
    D_name = save_D + '/' + name
    G_name = save_G + '/' + name
    cv2.imwrite(D_name, DepthMap * 255)
    cv2.imwrite(G_name, GradMap * 255)
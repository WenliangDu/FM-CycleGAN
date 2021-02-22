import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from util import util
from PIL import Image

def getCrossEntropyInput(fakeImage, realCenters, realSegMap):
    fakeImageI = util.tensor2im(fakeImage)
    if len(fakeImageI.shape) > 2:
        fakeImageI = fakeImageI[:, :, 1]
    fakeSegMap = getFakeSegMap(fakeImageI, realCenters)
    # derictly get SegMap from memory
    realSegMap = realSegMap.numpy()
    realSegMapI = realSegMap[:, :, 1]
    realSegMapI = realSegMapI - 1

    fakeSegMap_reshape = fakeSegMap.reshape(1, fakeSegMap.shape[0] * fakeSegMap.shape[1])
    realSegMap_reshape = realSegMapI.reshape(1, fakeSegMap.shape[0] * fakeSegMap.shape[1])

    labels = [0, 1, 2]
    lb = LabelBinarizer()
    lb.fit(labels)
    fakeSegMap_reshape_Bi = lb.transform(fakeSegMap_reshape[0])


    fakeSegMap_reshape_Bi_T = torch.cuda.FloatTensor(fakeSegMap_reshape_Bi)
    fakeSegMap_reshape_Bi_T.requires_grad = np.bool(1)
    realSegMap_reshape_T = torch.cuda.LongTensor(realSegMap_reshape[0])

    #criterionSeg = torch.nn.CrossEntropyLoss()
    #Loss2 = criterionSeg(fakeSegMap_reshape_Bi_T, realSegMap_reshape_T)
    #X = 1
    return fakeSegMap_reshape_Bi_T, realSegMap_reshape_T


def getFakeSegMap(fakeImage, realCenters):

    NumCenters = len(realCenters)
    # Get the Diff_map by getting the absolute value of differences between the pixel value of the fake image and the value of the centers
    fakeImageF = fakeImage.astype(np.float32)
    Centers_map = np.zeros([fakeImage.shape[0], fakeImage.shape[1], NumCenters], dtype=np.float32)
    Diff_map = np.zeros([fakeImage.shape[0], fakeImage.shape[1], NumCenters], dtype=np.float32)
    for i in range(NumCenters):
        Centers_map[:, :, i] = realCenters[i]
        Diff_map[:, :, i] = np.abs(fakeImageF - Centers_map[:, :, i])
    # Get fakeSegMap by getting the order of sorting Diff_map
    fakeSegMap = np.argsort(Diff_map, axis=2)[:, :, 0]
    return fakeSegMap
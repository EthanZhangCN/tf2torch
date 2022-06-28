import torch
import vgg as VGG
import cv2
import numpy as np
import pdb



torch_model = VGG.vgg16(pretrained=True)

img = cv2.imread('./test.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = np.expand_dims(np.transpose(img,(2,0,1)),0)


img_tensor = torch.Tensor(img)

img_feat = torch_model(img_tensor).detach().numpy()


print(img_feat)
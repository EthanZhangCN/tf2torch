import torch
import vgg as VGG
import cv2
import numpy as np
import pdb

# dic_path = 'vgg16_from_tf_notop.pth'

dic_path = 'vgg16-397923af.pth'



torch_model = VGG.vgg16(pretrained=True)

# state_dict =torch.load(dic_path)

# rm_keys = [ 'classifier.0.weight',
#             'classifier.0.bias',
#             'classifier.3.weight',
#             'classifier.3.bias',
#             'classifier.6.weight',
#             'classifier.6.bias',
# ]

# for i in range(6):
#     state_dict.pop(rm_keys[i])


# torch_model.load_state_dict(state_dict,strict = True)


img = cv2.imread('test.jpg')

img = np.expand_dims(np.transpose(img,(2,0,1)),0)

img_tensor = torch.Tensor(img)
# print(img_tensor)
img_feat = torch_model(img_tensor).detach().numpy()
# pdb.set_trace()
# img_feat = img_feat / torch.norm(img_feat)

# with torch.no_grad():
#     img_tensor = torch.Tensor(img)
#     img_feat = torch_model(img_tensor)
#     img_feat = img_feat / torch.norm(img_feat)


print(img_feat)


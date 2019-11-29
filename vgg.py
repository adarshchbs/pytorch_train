import torch
from torchvision import transforms, models

from torch import nn


resnet = models.resnet18(pretrained=False)
feature_extracter = torch.nn.Sequential(*(list(resnet.children())[:-1]))


class model(nn.Module):

    def __init__(self, feature_extracter, num_classes):
        super( model, self ).__init__()

        self.name = 'resnet_18'
        self.feature_extracter = feature_extracter

        self.fc = nn.Linear(in_features = 512, out_features = num_classes )

    def forward(self, inputs):
        out = self.feature_extracter(inputs)
        out = self.fc(out)
        return out


m = model(feature_extracter,10)

for i in m.feature_extracter.parameters():
    i.requires_grad = False
    



# from image_loader import image_loader

# sketch = image_loader('/home/iacv/project/sketch/dataset/images/')

# import time
# start = time.time()

# count = 0
# for i in  sketch.image_gen('train',batch_size=128):
#     b = []
#     for j in i[0]:
#         a = data_transforms['train'](j)
#         b.append(a)
#     count += 128
#     print(count, time.time()-start)
    

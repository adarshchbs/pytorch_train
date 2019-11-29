from torchvision import transforms
import torch 

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def preprocess_image(array, split_type, use_gpu = True):
    array_preprocess = []
    for i in array:
        array_preprocess.append( data_transforms[split_type](i) )
    if( use_gpu == True ):
        array_preprocess = torch.tensor(array_preprocess,dtype=torch.float32).cuda()
    else:
        array_preprocess = torch.tensor(array_preprocess,dtype=torch.float32)

    return array_preprocess


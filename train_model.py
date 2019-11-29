import torch
from torch import optim, nn

from preprocess import preprocess_image
from utils import save_model


num_epochs = 10
log_step = 10
eval_step = 5
save_step = 5

# from vgg import model, feature_extracter
# from image_loader import image_loader
# loader = image_loader('/home/iacv/project/sketch/dataset/images/')

# model_train = model(feature_extracter,10)

def train(model_train, loader, gpu_flag,root_folder):


    optimizer = optim.Adam( model_train.parameter() )
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range( num_epochs ):
        for step, ( images, lables ) in enumerate( loader ):

            images = preprocess_image( array = images,
                                       split_type = 'train',
                                       use_gpu = gpu_flag  )

            lables = torch.tensor(lables)

            if(gpu_flag == True):
                lables = lables.cuda()

            optimizer.zero_grad()

            preds = model_train( images )

            loss = criterion( preds, lables )

            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              num_epochs,
                              step + 1,
                              loader.size['train'],
                              loss.data.item()))

        # # eval model on validation set
        # if ((epoch + 1) % eval_step == 0):
        #     eval_src(source_encoder, source_classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % save_step == 0):
            save_model( model_train,  model_train.name + "-{}.pt".format(epoch + 1),root_folder)

    # save final model
    save_model(model_train, model_train.name + "-final.pt", root_folder)

    return model_train
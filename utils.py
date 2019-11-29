import os
import torch

def save_model(net, filename, root_folder):
    """Save trained model."""
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    torch.save(net.state_dict(),
               os.path.join(root_folder, filename))
    print("save pretrained model to: {}".format(os.path.join(root_folder,
                                                             filename)))

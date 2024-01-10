""" PyTorch interface to our dataset """
import torchvision.transforms as transforms

#(0.485,0.456,0.406), (0.299,0.224,0.225)

def get_transforms_train():
    """Return the transformations applied to images during training.
    
    See https://pytorch.org/vision/stable/transforms.html for a full list of 
    available transforms.
    """
    transform = transforms.Compose(
        [ 
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.485,0.456,0.406), (0.299,0.224,0.225))  # normalize
        ]
    )
    return transform


def get_transforms_val():
    """Return the transformations applied to images during validation.

    Note: You do not need to change this function 
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # convert image to a PyTorch Tensor
            transforms.Normalize((0.485,0.456,0.406), (0.299,0.224,0.225))  # normalize
        ]
    )
    return transform
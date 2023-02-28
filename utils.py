# Import libraries
import os, torch, timm, cv2
from imagenet_classes import get_classes
from torchvision import transforms

def switch_to_eval(model, device):
    
    """
    
    Gets a model as well as gpu device type and moves the model to the gpu device.
    
    Arguments:
    
        model - a cpu model;
        device - gpu device type.
    
    """
    
    model.to(device)
    model.eval()
    
    return model

def get_fm(fm):
        
        """
        
        Gets feature map with size (bs, fm_shape, 7, 7) applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Argument:
        
            fm - feature map.
        
        """
        
        pool = torch.nn.AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))

def load_model(model_name, num_classes):
    
    """
    
    Get a model name along with number of classes in the dataset and returns created model along with input size to the model.
    
    Arguments:
    
        model_name - a model name in timm models list;
        num_classes - number of classes in the considered dataset.
    
    """
    
    assert model_name in ["vgg16", "rexnet_150", "efficientnet_b3", "darknet53"], "Please choose the avaliable version of a timm model"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    print(f"{model_name} model is successfully loaded!")
    
    return model, model.pretrained_cfg["input_size"][1:] # tuple()

def predict(model, im, device):
    
    
    
    classes = get_classes()
    model = switch_to_eval(model, device)
    print("Obtaining feature maps and predict a class...")
    print(f"Inference on the {device}!")
    fm = model.forward_features(im.unsqueeze(0).to(device))
    preds = model.forward_head(fm)
    values, indices = torch.topk(preds, k=3)
    print(f"The image is predicted as {classes[indices[0][0].item()]} with {values[0][0].item():.2f}% confidence!")

    return values.squeeze(), indices.squeeze()

def preprocess(im1, im2):
    
    invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    im1 = invTrans(im1).permute(1,2,0).detach().cpu().numpy() * 255
    im2 = invTrans(im2).permute(1,2,0).detach().cpu().numpy() * 255
    concat_im = cv2.hconcat([im1, im2])
    
    return concat_im


def compute_cos_similarity(model, im1, im2, sim_fn):
    
    im1_fm = get_fm(model.forward_features(im1.unsqueeze(0)))
    im2_fm = get_fm(model.forward_features(im2.unsqueeze(0)))
    cos_sim = sim_fn(im1_fm, im2_fm).item()
    print(f"Similarity between images is {cos_sim:.3f}\n")
    
    return cos_sim

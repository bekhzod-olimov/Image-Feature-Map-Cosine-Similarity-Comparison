# Import libraries
import os, torch, timm, cv2
from imagenet_classes import get_classes
from torchvision import transforms

def switch_to_eval(model, device):
    
    """
    
    This function gets a model as well as gpu device type and moves the model to the gpu device.
    
    Parameters:
    
        model  - a cpu model, timm model object;
        device - gpu device type, str.
        
    Output:
    
        model  - a model that is switched to evaluation mode and moved to gpu, timm model object.
    
    """
    
    model.to(device)
    
    return model.eval()

def get_fm(fm):
        
        """
        
        This function gets feature map with size (bs, fm_shape, 7, 7) applies average pooling and returns feature map with shape (bs, fm_shape).
        
        Parameter:
        
            fm  - feature map, tensor.
        
        Output:
        
            fm  - 2D feature map with the shape of (batch_size, feature map shape), tensor.
            
        """
        
        pool = torch.nn.AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))

def load_model(model_name, num_classes):
    
    """
    
    This function gets a model name along with number of classes in the dataset and returns created model along with input size to the model.
    
    Parameters:
    
        model_name  - a model name in timm models list, str;
        num_classes - number of classes in the considered dataset, int.
        
    Outputs:
    
        model       - a model to be used for the experiments, timm model object;
        im_size     - input image size for the model, tuple.
    
    """
    
    assert model_name in ["vgg16", "rexnet_150", "efficientnet_b3", "darknet53"], "Please choose the avaliable version of a timm model"
    model = timm.create_model(model_name, pretrained = True, num_classes = num_classes)
    print(f"{model_name} model is successfully loaded!")
    
    return model, model.pretrained_cfg["input_size"][1:] 

def predict(model, im, device):
    
    """
    
    This function gets model, image, gpu device and return top3 values and indices.
    
    Parameters:
    
        model  - model to be trained, timm model object;
        im     - an image, tensor;
        device - gpu device name, str.
        
   Outputs:
   
       values  - squeezed top3 values, tensor;
       indices - squeezed top3 indices, tensor.
    
    """

    # Get imagenet classes
    classes = get_classes()
    
    # Move the model to gpu
    model = switch_to_eval(model, device)
    print("Obtaining feature maps and predict a class...")
    print(f"Inference on the {device}!")
    
    # Get feature maps
    fm = model.forward_features(im.unsqueeze(0).to(device))
    
    # Get the predicted classes
    preds = model.forward_head(fm)
    
    # Get the top3 values and indices
    values, indices = torch.topk(preds, k=3)
    print(f"The image is predicted as {classes[indices[0][0].item()]} with {values[0][0].item():.2f}% confidence!")

    return values.squeeze(), indices.squeeze()

def preprocess(im1, im2):
    
    """
    
    This function gets two images, applies transformations, and returns a concatenaed image.
    
    Parameters:
    
        im1 - image number 1;
        im2 - image number 2.
    
    Output:
    
        a concatened image.
        
    """
    
    # Initialize inverse function for normalization    
    invTrans = transforms.Compose([
                                   transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
                               ])
    
    # Apply transformations and change from tensor to array
    im1 = invTrans(im1).permute(1,2,0).detach().cpu().numpy() * 255
    im2 = invTrans(im2).permute(1,2,0).detach().cpu().numpy() * 255
    
    # Return a concatenated image
    return cv2.hconcat([im1, im2])

def compute_cos_similarity(model, im1, im2, sim_fn):
    
    """
    
    This function gets model, two images, and similarity function and returns similarity score for the images.
    
    Parameters:
    
        model   - trained model;
        im1     - the first image to be compared;
        im2     - the second image to be compared;
        sim_fn  - function to compute similarity
        
    Output:
    
        cos_sim - cosine similarity score between the two images, float.
    
    """
    
    # Get feature maps of the images
    im1_fm, im2_fm  = get_fm(model.forward_features(im1.unsqueeze(0))), get_fm(model.forward_features(im2.unsqueeze(0)))
    
    # Compute similarity score
    cos_sim = sim_fn(im1_fm, im2_fm).item()
    print(f"Similarity between images is {cos_sim:.3f}\n")
    
    return cos_sim

# Import libraries
import os
from matplotlib import pyplot as plt
from utils import switch_to_eval

def get_di(model, im, model_name):
    
    """
    
    This function gets a model, image, as well as model name and returns dictionary with plot's height and width.
    
    Arguments:
    
        model      - a model to be trained, timm model;
        im         - an image, tensor;
        model_name - name of the model, str.
        
    Outputs:
    
        di         - dictionary with layer names as keys and feature maps as values, dictionary;
        height     - height of the plot, int;
        width      - width of the plot, int
    
    """
    
    # Initialize a dictionary
    di = {}
    
    # Set height and width
    height, width = 4, 6
    last_im_dim = None
    
    # Set layers to extract features based on the model name (VGG16 in this case)
    if model_name == "vgg16":
        
        # Set feature extraction layers
        layers = model.features
        height, width = 8, 8
    
    # RexNet
    elif model_name == "rexnet_150": 
        # Set feature extraction layers
        layers = model.features
        
        # Go through the stem layers
        im = model.stem(im.unsqueeze(dim=0))
        
    # EfficientNet
    elif model_name == "efficientnet_b3":
        # Set feature extraction layers
        layers = model.blocks
        
        # Go through the convolution stem layers
        im = model.conv_stem(im)
        
    # DarkNet
    elif model_name == "darknet53":
        # Set feature extraction layers
        layers = model.stages
        
        # Go through the convolution stem layers
        im = model.stem(im.unsqueeze(dim=0))
    
    # Go through the feature extraction layers
    for i, layer in enumerate(layers):
        
        # Prepare an image
        im = im.unsqueeze(0) if i == 0 and model_name == "efficientnet_b3" else im
        
        # Get the output of the features from the corresponding feature extraction layer
        im = layer(im)
        
        # Get the shape of the feature maps
        im_shape = im.shape[2:] if len(im.shape) == 4 else im.shape[1:]
        
        # Write feature maps to the dictionary
        if last_im_dim != im_shape: di[f"Layer_{i}"] = im.squeeze()
            
        # Get shape of the last feature map
        last_im_dim = im.shape[2:] if len(im.shape) == 4 else im.shape[1:]
                
    return di, height, width

def plot_fms(model, im, save_path, device, model_name):
    
    """ 
    
    This function gets a model with an image and plots feature maps from the convolution layers.
    
    Arguments:
    
        model - a model, timm model;
        im - an image after transformations application, tensor;
        save_path - a path to the directory to save images, str.

    """
    
    # Move the model to gpu device
    model = switch_to_eval(model, device)
    
    # Move the image to gpu device
    im = im.to(device)
    
    # Get dictionary as well as height and width of the plot
    di, height, width = get_di(model, im, model_name)    
    
    print("Saving feature maps images...")
    for i, fmap in enumerate(list(di.values())):
        print(f"Feature map shape: {fmap.shape}")
        ix = 1
        plt.figure()

        for _ in range(height):
            for _ in range(width):
                ax = plt.subplot(height, width, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[ix-1, :, :].detach().cpu().numpy(), cmap='gray')
                ix += 1
        os.makedirs(f"{save_path}", exist_ok=True)
        plt.savefig(f"{save_path}/{list(di.keys())[i]}.png")
        # plt.show()

        if fmap is not list(di.values())[-1]:
            print("."*48)
            
    print(f"Check the results in ./{save_path}/")

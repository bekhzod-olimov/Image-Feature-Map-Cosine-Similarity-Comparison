from matplotlib import pyplot as plt
import os
from utils import switch_to_eval

def get_di(model, im, model_name):
    
    di = {}
    height, width = 4, 6
    
    if model_name == "vgg16": 
        layers = model.features
        height, width = 8, 8
    elif model_name == "rexnet_150": 
        layers = model.features
        im = model.stem(im.unsqueeze(dim=0))
    elif model_name == "efficientnet_b3":
        layers = model.blocks
        im = model.conv_stem(im)
    elif model_name == "darknet53":
        layers = model.stages
        im = model.stem(im.unsqueeze(dim=0))
    
    for i, layer in enumerate(layers):
        im = im.unsqueeze(0) if i == 0 and model_name == "efficientnet_b3" else im
        im = layer(im)
        di[f"Layer_{i}"] = im.squeeze() if not ".Conv2d" in str(type(layer)) else 0
                
    return di, height, width

def plot_fms(model, im, save_path, device, model_name):
    
    """ 
    Gets a model with an image and plots feature maps from the convolution layers.
    
    Arguments:
        model - a timm model;
        im - tensor image with transformations;
        save_path - path to the directory to save images.

    """
    model = switch_to_eval(model, device)
    im = im.to(device)
    di, height, width = get_di(model, im, model_name)    
    
    print("Saving feature maps images...")
    for i, fmap in enumerate(list(di.values())):
        if type(fmap) == int: pass
        else:
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

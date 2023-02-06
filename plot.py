from matplotlib import pyplot as plt
import os

def plot_fms(model, im, save_path, device):
    
    """ 
    Gets a model with an image and plots feature maps from the convolution layers.
    
    Arguments:
        model - a timm model;
        im - tensor image with transformations;
        save_path - path to the directory to save images.

    """
    model.to(device)
    im = im.to(device)
    height, width = 8, 8
    di = {}
    for i, layer in enumerate(model.features):
        im = layer(im)
        if ".Conv2d" in str(type(layer)):
            di[f"Layer_{i}"] = im
    
    print("Saving feature maps images...")
    for i, fmap in enumerate(list(di.values())):
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

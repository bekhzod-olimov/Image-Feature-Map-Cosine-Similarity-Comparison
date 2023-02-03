from matplotlib import pyplot as plt

def plot_fms(model, im, save_path):
    
    """ 
    Gets a model with an image and plots feature maps from the convolution layers.
    
    Arguments:
        model - a timm model;
        im - tensor image with transformations;
        save_path - path to the directory to save images.

    """
    
    height, width = 8, 8
    di = {}
    for i, layer in enumerate(model.features):
        im = layer(im)
        if ".Conv2d" in str(type(layer)):
            di[f"Layer_{i}"] = im
    
    # Plot feature_maps for each block
    for i, fmap in enumerate(list(di.values())):
        print(f"\t\t     {list(di.keys())[i]}")
        ix = 1
        plt.figure()
        
        for _ in range(height):
            for _ in range(width):
            # specify subplot and turn of axis
                ax = plt.subplot(height, width, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(fmap[ix-1, :, :].detach().numpy(), cmap='gray')
                ix += 1
        os.makedirs(f"{save_path}", exist_ok=True)
        plt.savefig(f"{save_path}/{idx[i]}.png")
        plt.show()
        
        if fmap is not list(di.values())[-1]:
            print("-"*48)

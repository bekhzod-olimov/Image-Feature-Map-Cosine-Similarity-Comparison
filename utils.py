import os, torch, timm
from matplotlib import pyplot as plt
import torchvision.transforms as T
from PIL import Image

def apply_transformations(im_path, im_size):
    
    im = Image.open(im_path)
    tfs = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomHorizontalFlip(),
                     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ])
    
    return tfs(im)

def get_fm(fm):
        
        """
        
        Gets feature map with size (bs, fm_shape, 7, 7)
        applies average pooling and returns feature map
        with shape (bs, fm_shape).
        
        Argument:
        
        fm - feature map.
        
        """
        
        pool = torch.nn.AvgPool2d((fm.shape[2],fm.shape[3]))
        
        return torch.reshape(pool(fm), (-1, fm.shape[1]))

def load_model(model_name, num_classes):

    assert model_name in list(timm.list_models()), "Please choose the correct version of a model (refer to list(timm.list_models()))"
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    print(f"{model_name} model is successfully loaded!")
    
    return model, model.pretrained_cfg["input_size"][1:] # tuple()

def predict(model, im, device):
    
    model.to(device)
    model.eval()
    fm = model.forward_features(im.unsqueeze(0).to(device))
    preds = model.forward_head(fm)
    values, indices = torch.topk(preds, k=5)

    return values.squeeze(), indices.squeeze()

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
        # print(f"\t\t     {list(di.keys())[i]}")
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
        plt.savefig(f"{save_path}/{list(di.keys())[i]}.png")
        plt.show()
        
        if fmap is not list(di.values())[-1]:
            print("-"*48)

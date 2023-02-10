import torchvision.transforms as T
from PIL import Image

def apply_transformations(im_path, im_size, random_tfs):
    
    """
    Gets an image path, image size, as well as tranformations and applies transformations to the image.
    
    Arguments:
    
    
    
    
    """
    
    im = Image.open(im_path)
    print("Applying transformations...")
    
    tfs = T.Compose([T.ToTensor(), T.Resize(im_size), 
                     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ])
    random_rotation = T.Compose([T.RandomRotation(45)])
   
    return random_rotation(tfs(im)) if random_tfs else tfs(im)

import torchvision.transforms as T
from PIL import Image

def apply_transformations(im_path, im_size, random_tfs):
    
    """
    Gets an image path, image size, as well as tranformations and applies transformations to the image.
    
    Arguments:
    im_path - a path to the image to be transformed;
    im_size - desired size for the image;
    random_tfs - transformations to be applied.
    
    """
    
    # Load an image from the given path
    im = Image.open(im_path)
    print("Applying transformations...")
    
    # Initialize transformations
    tfs = T.Compose([T.ToTensor(), # 
                     T.Resize(im_size), 
                     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ])
    random_rotation = T.Compose([T.RandomRotation(45)])
   
    return random_rotation(tfs(im)) if random_tfs else tfs(im)

# Import libraries
import torchvision.transforms as T
from PIL import Image

def apply_transformations(im_path, im_size, random_tfs):
    
    """
    
    This function gets an image path, image size, as well as random transformations and applies transformations to the image.
    
    Arguments:
    
        im_path     - a path to the image to be transformed, str;
        im_size     - desired size for the image, tuple;
        random_tfs  - apply a random transformation or not, bool.
        
    Output:
        
        transformations, torchvision object.
    
    """
    
    # Load an image from the given path
    im = Image.open(im_path)
    print("Applying transformations...")
    
    # Initialize transformations
    tfs = T.Compose([
                    # Transform to Tensor
                    T.ToTensor(),
                     # Resize
                     T.Resize(im_size), 
                     # Data Standardization
                     T.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)) 
                     ])
    
    # Initialize rotation transformation
    random_rotation = T.Compose([T.RandomRotation(45)])
   
    # Output image depending on the random_transformations
    return random_rotation(tfs(im)) if random_tfs else tfs(im)

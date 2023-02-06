import torchvision.transforms as T
from PIL import Image

def apply_transformations(im_path, im_size):
    
    im = Image.open(im_path)
    print("Applying transformations...")
    tfs = T.Compose([T.ToTensor(), T.Resize(im_size), T.RandomHorizontalFlip(),
                     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                     ])
    
    return tfs(im)

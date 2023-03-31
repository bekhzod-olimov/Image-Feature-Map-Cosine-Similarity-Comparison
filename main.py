# Import libraries
import argparse, yaml
from utils import *
from plot import plot_fms
from transform import apply_transformations
from PIL import ImageDraw

def run(args):

    # Print arguments
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nArguments:\n\n{argstr}")
    
    # Initialize cosine similarity computation function
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    
    # Get model and input size for the loaded model
    model, inp_size = load_model(args.model_name, 1000)
    
    # Go through the number of experiments
    for i in range(args.experiments):
    
        # Apply image transformations to the image #1
        im1 = apply_transformations(args.image_path_1, inp_size, False)
        
        # Apply image transformations to the image #1
        im2 = apply_transformations(args.image_path_1, inp_size, True)
#         im2 = apply_transformations(args.image_path_2, inp_size, True)
        
        # Compute cosine similarity between two images
        cos_sim = compute_cos_similarity(model, im1, im2, cos)
        
        # Concatenate the images
        concat_im = preprocess(im1, im2)
        
        # Write cosine similarity score to the concatenated image
        I1 = ImageDraw.Draw(concat_im)
        
        # Write text
        I1.text((150, 10), f"Similarity: {cos_sim:.2f}", font = ImageFont.truetype("arial.ttf", 25), fill = (0, 0, 255))
        
        # Make a directory if does not exist
        os.makedirs(f"concat_ims/{args.model_name}", exist_ok=True)
        
        # Save the image
        cv2.imwrite(f"concat_ims/{args.model_name}/sample_{i}_{cos_sim:.2f}.jpg", np.array(concat_im))

if __name__ == "__main__":
    
    # Initialize parser
    parser = argparse.ArgumentParser(description = 'Feature Map Visualization Arguments')
    
    # Add arguments
    parser.add_argument('-ip1', '--image_path_1', default='kuvasz.jpg', help='path to the first image')
    parser.add_argument('-ip2', '--image_path_2', default='cup.jpg', help='path to the second image')
    parser.add_argument('-sp', '--save_path', default='fm_images', help='path to a directory to save feature map images')
    parser.add_argument('-ex', '--experiments', default=5, help='number of experiments')
    parser.add_argument('-d', '--device', default='cuda:3', help='gpu device name')
    parser.add_argument('-mn', '--model_name', default='vgg16', help='a model name from timm models')
    # parser.add_argument('-mn', '--model_name', default='rexnet_150', help='a model name from timm models')
    # parser.add_argument('-mn', '--model_name', default='darknet53', help='a model name from timm models')
    # parser.add_argument('-mn', '--model_name', default='efficientnet_b3', help='a model name from timm models')
    
    # Parse the arguments
    args = parser.parse_args() 
    
    # Run the script
    run(args) 

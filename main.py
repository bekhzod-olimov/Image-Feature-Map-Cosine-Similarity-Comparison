import argparse, yaml
from utils import *
from plot import plot_fms
from transform import apply_transformations

def run(args):

    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print(f"\nArguments:\n\n{argstr}")
    
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    model, inp_size = load_model(args.model_name, 1000)
    
    for i in range(10):
    
        im1 = apply_transformations(args.image_path_1, inp_size, False)
        im2 = apply_transformations(args.image_path_1, inp_size, True)
        cos_sim = compute_cos_similarity(model, im1, im2, cos)
        concat_im = preprocess(im1, im2)
        cv2.imwrite(f"concat_ims/sample_{i}_{cos_sim:.2f}.jpg", concat_im)
    # values, indices = predict(model, im, args.device)
    plot_fms(model, im1, f"{args.save_path}/{args.model_name}", args.device, args.model_name)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Feature Map Visualization Arguments')
    parser.add_argument('-ip1', '--image_path_1', default='kuvasz.jpg', help='path to the first image')
    parser.add_argument('-ip2', '--image_path_2', default='cat.jpg', help='path to the second image')
    parser.add_argument('-sp', '--save_path', default='fm_images', help='path to a directory to save feature map images')
    parser.add_argument('-d', '--device', default='cuda:3', help='gpu device name')
    # parser.add_argument('-mn', '--model_name', default='vgg16', help='a model name from timm models')
    # parser.add_argument('-mn', '--model_name', default='rexnet_150', help='a model name from timm models')
    # parser.add_argument('-mn', '--model_name', default='darknet53', help='a model name from timm models')
    parser.add_argument('-mn', '--model_name', default='efficientnet_b3', help='a model name from timm models')
    
    
    args = parser.parse_args() 
    
    run(args) 
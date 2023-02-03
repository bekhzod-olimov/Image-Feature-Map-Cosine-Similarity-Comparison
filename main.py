from imagenet_classes import get_classes
from utils import *

classes = get_classes()
device = "cuda:3"
model, inp_size = load_model("vgg16", 1000)
im = apply_transformations("kuvasz.jpg", inp_size)
plot_fms(model, im, "fm_images")



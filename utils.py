import os, torch, timm

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
    print("Obtaining feature maps and predict a class...")
    fm = model.forward_features(im.unsqueeze(0).to(device))
    preds = model.forward_head(fm)
    values, indices = torch.topk(preds, k=5)

    return values.squeeze(), indices.squeeze()


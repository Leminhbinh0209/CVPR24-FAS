from .resnet import Resnet18

def get_model(config):
    if config.MODEL.model_name  == 'resnet18':       
        return  Resnet18(pretrained=config.TRAIN.pretrained, 
                         n_classes=config.MODEL.num_classes)
    else:
        raise ValueError(f"Cannot find the model {config.MODEL.model_name}")
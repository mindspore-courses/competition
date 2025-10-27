from models.backbones.resnet import build_resnet

def build_backbone(name: str, *args, **kwargs):
    if 'resnet' in name.lower():
        return build_resnet(*args, **kwargs)
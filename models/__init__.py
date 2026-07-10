# -*- coding: utf-8 -*-

from .vgg16bn import VGG16_BN
from .emac_wrapper import EMACWrapper

def build_model(config):
    model_cls = {
        'vgg16bn': VGG16_BN,
        'emac': EMACWrapper,
    }[config.NAME.lower()]

    if config.NAME.lower() == 'emac':
        return model_cls(config), model_cls(config)
    return model_cls(config), model_cls(config)

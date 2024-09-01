from .encoder_default import encoder_default

def get_encoder(image_height, image_width, scale=0.25, sigma=1.5, encoder_type='default'):
    if encoder_type == 'default':
        encoder = encoder_default(image_height, image_width, scale, sigma)
    else:
        raise NotImplementedError
    return encoder

from .decoder_default import decoder_default


def get_decoder(decoder_type='default'):
    if decoder_type == 'default':
        decoder = decoder_default()
    else:
        raise NotImplementedError
    return decoder

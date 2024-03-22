import transformers
from packaging import version


def compatible_position_ids(state_dict, position_id_key):
    """Transformers no longer expect position_ids after transformers==4.31
       https://github.com/huggingface/transformers/pull/24505

    Args:
        position_id_key (str): position_ids key,
            such as(encoder.embeddings.position_ids)
    """
    transformer_version = version.parse('.'.join(
        transformers.__version__.split('.')[:2]))
    if transformer_version >= version.parse('4.31.0'):
        del state_dict[position_id_key]

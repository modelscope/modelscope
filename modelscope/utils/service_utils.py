import base64
import mimetypes
from io import BytesIO

import json
import numpy as np
import requests

from modelscope.outputs import TASK_OUTPUTS, OutputKeys
from modelscope.pipeline_inputs import TASK_INPUTS, InputType


# service data decoder func decodes data from network and convert it to pipeline's input
# for example
def ExampleDecoder(data):
    # Assuming the pipeline inputs is a dict contains an image and a text,
    # to decode the data from network we decode the image as base64
    data_json = json.loads(data)
    # data: {"image": "xxxxxxxx=="(base64 str), "text": "a question"}
    # pipeline(inputs) as follows:
    # pipeline({'image': image, 'text': text})
    inputs = {
        'image': decode_base64_to_image(data_json.get('image')),
        'text': data_json.get('text')
    }
    return inputs


# service data encoder func encodes data from pipeline outputs and convert to network response (such as json)
# for example
def ExampleEncoder(data):
    # Assuming the pipeline outputs is a dict contains an image and a text,
    # and transmit it through network, this func encode image to base64 and dumps into json
    # data (for e.g. python dict):
    # {"image": a numpy array represents a image, "text": "output"}
    image = data['image']
    text = data['text']
    data = {'image': encode_array_to_img_base64(image), 'text': text}
    return json.dumps(data, cls=NumpyEncoder)


CustomEncoder = {
    # Tasks.visual_question_answering: ExampleEncoder
}

CustomDecoder = {
    # Tasks.visual_question_answering: ExampleDecoder
}


class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        return json.JSONEncoder.default(self, obj)


def get_extension(encoding):
    encoding = encoding.replace('audio/wav', 'audio/x-wav')
    tp = mimetypes.guess_type(encoding)[0]
    if tp == 'audio/flac':  # flac is not supported by mimetypes
        return 'flac'
    extension = mimetypes.guess_extension(tp)
    if extension is not None and extension.startswith('.'):
        extension = extension[1:]
    return extension


def get_mimetype(filename):
    mimetype = mimetypes.guess_type(filename)[0]
    if mimetype is not None:
        mimetype = mimetype.replace('x-wav', 'wav').replace('x-flac', 'flac')
    return mimetype


def decode_base64_to_binary(encoding):
    extension = get_extension(encoding)
    data = encoding.split(',')[1]
    return base64.b64decode(data), extension


def decode_base64_to_image(encoding):
    from PIL import Image
    content = encoding.split(';')[1]
    image_encoded = content.split(',')[1]
    return Image.open(BytesIO(base64.b64decode(image_encoded)))


def encode_array_to_img_base64(image_array):
    from PIL import Image
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(image_array.astype(np.uint8))
        pil_image.save(output_bytes, 'PNG')
        bytes_data = output_bytes.getvalue()
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return 'data:image/png;base64,' + base64_str


def encode_pcm_to_base64(bytes_data):
    from scipy.io.wavfile import write
    with BytesIO() as out_mem_file:
        write(out_mem_file, 16000, bytes_data)
        base64_str = str(base64.b64encode(out_mem_file.getvalue()), 'utf-8')
    return 'data:audio/pcm;base64,' + base64_str


def encode_url_to_base64(url):
    encoded_string = base64.b64encode(requests.get(url).content)
    base64_str = str(encoded_string, 'utf-8')
    mimetype = get_mimetype(url)
    return ('data:' + (mimetype if mimetype is not None else '') + ';base64,'
            + base64_str)


def encode_file_to_base64(f):
    with open(f, 'rb') as file:
        encoded_string = base64.b64encode(file.read())
        base64_str = str(encoded_string, 'utf-8')
        mimetype = get_mimetype(f)
        return ('data:' + (mimetype if mimetype is not None else '')
                + ';base64,' + base64_str)


def encode_url_or_file_to_base64(path):
    try:
        requests.get(path)
        return encode_url_to_base64(path)
    except (requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema):
        return encode_file_to_base64(path)


def service_data_decoder(task, data):
    if CustomDecoder.get(task) is not None:
        return CustomDecoder[task](data)
    input_data = data.decode('utf-8')
    input_type = TASK_INPUTS[task]
    if isinstance(input_type, list):
        input_type = input_type[0]
    if input_type == InputType.IMAGE:
        return decode_base64_to_image(input_data)
    elif input_type == InputType.AUDIO:
        return decode_base64_to_binary(input_data)[0]
    elif input_type == InputType.TEXT:
        return input_data
    elif isinstance(input_type, dict):
        input_data = {}
        data = json.loads(data)
        for key, val in input_type.items():
            if val == InputType.IMAGE:
                input_data[key] = decode_base64_to_image(data[key])
            elif val == InputType.AUDIO:
                input_data[key] = decode_base64_to_binary(data[key])[0]
            elif val == InputType.TEXT:
                input_data[key] = data[key]
            else:
                return data

    return input_data


def service_data_encoder(task, data):
    if CustomEncoder.get(task) is not None:
        return CustomEncoder[task](data)
    output_keys = TASK_OUTPUTS[task]
    result = data
    for output_key in output_keys:
        if output_key == OutputKeys.OUTPUT_IMG:
            result[OutputKeys.OUTPUT_IMG] = encode_array_to_img_base64(
                data[OutputKeys.OUTPUT_IMG][..., ::-1])
        elif output_key == OutputKeys.OUTPUT_PCM:
            result[OutputKeys.OUTPUT_PCM] = encode_pcm_to_base64(
                data[OutputKeys.OUTPUT_PCM])
    result = bytes(json.dumps(result, cls=NumpyEncoder), encoding='utf8')
    return result

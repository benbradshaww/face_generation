import tensorflow as tf

def normalize(image, tanh:bool=False):
    '''
    Normalizes an image tensor for model input, using either tanh or standard normalization.

    This function takes an image tensor and normalizes its pixel values. If the `tanh` 
    parameter is set to True, it scales the pixel values to the range [-1, 1] suitable for 
    tanh activation. Otherwise, it scales the pixel values to the range [0, 1].

    Args:
        image (tf.Tensor): A tensor representing the image to be normalized.
        tanh (bool): If True, applies normalization for tanh activation. If False, applies 
                     standard normalization.

    Returns:
        tf.Tensor: The normalized image tensor.
    '''

    image = tf.cast(image, tf.float32)

    if tanh:
        return (image/127.5) - 1
    else:
        return image / 255.0

def preprocess(tanh: bool = False):
    '''
    Creates a function to preprocess images by normalizing them.

    This function generates a wrapper function that reads an image from a given file path, 
    decodes the image, and normalizes it using the specified normalization method.

    Args:
        tanh (bool): If True, applies a tanh activation function during normalization. 
                     If False, applies a standard normalization.

    Returns:
        function: A wrapper function that takes a file path as input, reads the image file, 
                  decodes the JPEG image, and applies the chosen normalization process.

    '''
    def wrapper(file_path: str):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = normalize(image, tanh)
        return image
    return wrapper



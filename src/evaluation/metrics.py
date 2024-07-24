import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.linalg import sqrtm

def calculate_fid(real_images, generated_images, batch_size=32):
    '''
    Calculates the Fréchet Inception Distance (FID) between real and generated images.

    Parameters:
    real_images (numpy.ndarray): Array of real images.
    generated_images (numpy.ndarray): Array of generated images.
    batch_size (int): Batch size for processing images through the InceptionV3 model.

    Returns:
    float: The FID score.
    '''
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    real_features = get_inception_features(real_images, inception_model, batch_size)
    gen_features = get_inception_features(generated_images, inception_model, batch_size)

    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid

def get_inception_features(images, model, batch_size=32):
    '''
    Extracts features from images using the InceptionV3 model.

    Parameters:
    images (numpy.ndarray): Array of images to process.
    model (tensorflow.keras.Model): Pre-trained InceptionV3 model.
    batch_size (int): Batch size for processing images.

    Returns:
    numpy.ndarray: Array of extracted features.
    '''
    n_batches = len(images) // batch_size
    features = []

    for i in range(n_batches):
        batch = images[i*batch_size:(i+1)*batch_size]
        batch_pp = preprocess_input(batch)
        batch_features = model.predict(batch_pp)
        features.append(batch_features)

    return np.concatenate(features, axis=0)


def calculate_inception_score(images, batch_size=32, num_splits=10):
    # Load pre-trained InceptionV3 model
    inception_model = InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))

    # Get the number of images
    n_images = images.shape[0]
    n_batches = n_images // batch_size

    # Get predictions
    preds = []
    for i in range(n_batches):
        batch = images[i*batch_size:(i+1)*batch_size]
        batch_pp = preprocess_input(batch)
        batch_preds = inception_model.predict(batch_pp)
        preds.append(batch_preds)
    
    preds = np.concatenate(preds, axis=0)

    # Calculate the inception score
    scores = []
    for i in range(num_splits):
        part = preds[(i * n_images // num_splits):((i + 1) * n_images // num_splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    # Return the mean and standard deviation of the inception score
    return np.mean(scores), np.std(scores)

def preprocess_images(images):
    # Assuming images are in range [0, 255]
    # Resize to 299x299 if necessary
    if images.shape[1:3] != (299, 299):
        images = tf.image.resize(images, (299, 299))
    return images.numpy()

# GAN Face Generation Project
This repository contains code and resources for a machine learning project focused on testing a range of Generative Adversarial Networks (GANs) for face generation using the CelebA dataset.

## Introduction
The goal of this project is to explore the capabilities of various Generative AI architectures in tasks of generating realistic human faces. The CelebA dataset, which contains over 200,000 celebrity images with 40 attribute annotations, is used for training and evaluating the models.

The GANs tested in this project include:
* Wasserstein GAN GP (WGAN-GP)
* StyleGAN
* Denoising Diffusion Probabilistic Models

## Project Structure
The folders and files in this repository are organized as follows:
* src/: Contains all the Python code used in the scripts.
    * models/: Contains custom TensorFlow layers, models, and utility functions.
    * data/: Data processing scripts, including loading, cleaning, and transforming data.
	* training/: Scripts and modules related to training models, including custom training loops.
	* evaluation/: Scripts for evaluating model performance and generating metrics.
    * misc/: Miscellaneous scripts. These including some plotting and model saving scripts. 
	* utils/: Helper functions and utilities used throughout the project.
* data/: Contains the datasets used in the project.
    * all_images/: Contains all the images from the celebA dataset.
	* 10000_images/: Contains a 10000 sample of the celebA dataset.
    * 10000_images_downscaled/: Contains a 10000 downscaled sample of the celebA dataset.
* notebooks/:

## Results
I have written the code for WGAN-GP and StyleGan but I have only trained the WGAN-GP model. This model produced the following images during training.

![Alt text](./src/training_images/training_gifs/wgan_gp_training_gif.gif)

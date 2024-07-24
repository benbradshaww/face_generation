import tensorflow as tf

def discriminator_loss(real_logits, fake_logits):
    '''
    Computes the Wasserstein loss for the discriminator.

    This function calculates the discriminator loss using the Wasserstein 
    loss formula, which is the difference between the mean of the fake logits 
    and the mean of the real logits.

    Parameters:
    real_logits (tensor): The logits for the real data, output from the discriminator.
    fake_logits (tensor): The logits for the fake data (generated data), output from the discriminator.

    Returns:
    tensor: The discriminator loss value, computed as the mean of the fake logits 
            minus the mean of the real logits.
    '''
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def generator_loss(fake_logits):
    '''
    Computes the Wasserstein loss for the generator.

    This function calculates the generator loss using the Wasserstein loss 
    formula, which is the negative mean of the fake logits.

    Parameters:
    fake_logits (tensor): The logits for the fake data (generated data), output from the discriminator.

    Returns:
    tensor: The generator loss value, computed as the negative mean of the fake logits.
    '''
    return -tf.reduce_mean(fake_logits)


def combined_metric(gen_loss:float, disc_loss:float, alpha=0.75):
    '''
    Computes a combined metric from generator and discriminator losses.

    This function calculates a weighted sum of the generator loss (`gen_loss`)
    and the discriminator loss (`disc_loss`). The parameter `alpha` controls
    the weighting between the two losses.

    Parameters:
    gen_loss (float): The loss value from the generator.
    disc_loss (float): The loss value from the discriminator.
    alpha (float, optional): The weighting factor for the generator loss.
                             Defaults to 0.75. Should be between 0 and 1.
                             A higher alpha gives more weight to the generator loss.

    Returns:
    float: The combined metric value, computed as a weighted sum of the 
           generator and discriminator losses.
    '''

    return alpha * gen_loss + (1 - alpha) * disc_loss


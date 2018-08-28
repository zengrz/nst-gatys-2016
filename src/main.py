import os
import sys
import scipy.io
import scipy.misc
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# from PIL import Image
from utils import *
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C, [m, -1, n_C])
    a_G_unrolled = tf.reshape(a_G, [m, -1, n_C])
    J_content = 1 / (4 * n_H * n_W * n_C) * tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled))
    return J_content

def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = 1 / (4 * n_C**2 * (n_H*n_W)**2) * tf.reduce_sum(tf.squared_difference(GS, GG))

    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 0.5):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the relative importance of the content cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha * J_content + J_style
    return J

def generate(sess, model, input_image, num_iterations, prefix):    
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        sess.run(train_step)
        input_image = sess.run(model['input'])
        if i % CONFIG.SAVE_EVERY == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + ":")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image(os.path.join(CONFIG.OUTPUT_DIR, prefix + "_" + str(i) + ".png"), input_image)
    save_image(os.path.join(CONFIG.OUTPUT_DIR, prefix + "_final.png"), input_image)
    return input_image

files = os.listdir(CONFIG.IMAGES_DIR)
c = 0
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

while True:
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    model = load_vgg_model(CONFIG.PRETRAINED_MODEL)
    layer_names = list(model.keys())

    a, b = random.choice(files), random.choice(files)
    alpha = random.uniform(CONFIG.MIN_ALPHA, CONFIG.MAX_ALPHA)
    lr = random.uniform(CONFIG.MIN_LR, CONFIG.MAX_LR)
    num_iter = CONFIG.MAX_NUM_ITER # random.randrange(CONFIG.MIN_NUM_ITER, CONFIG.MAX_NUM_ITER) // CONFIG.SAVE_EVERY * CONFIG.SAVE_EVERY
    layer_name = random.choice(layer_names)

    print("\nTask: ", c)
    with open("log.txt", "a") as f:
        f.write(str(c) + ":")
        f.write(a + ":")
        f.write(b + ":")
        f.write(str(alpha) + ":")
        f.write(str(lr) + ":")
        f.write(str(num_iter) + ":")
        f.write(layer_name)
        f.write("\n")

    content_image = scipy.misc.imread(os.path.join(CONFIG.IMAGES_DIR, a))
    content_image = random_crop(content_image, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread(os.path.join(CONFIG.IMAGES_DIR, b))
    style_image = random_crop(style_image, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH)
    style_image = reshape_and_normalize_image(style_image)

    sess.run(model['input'].assign(content_image))
    out = model[layer_name]
    a_C = sess.run(out)
    a_G = out
    J_content = compute_content_cost(a_C, a_G)

    sess.run(model['input'].assign(style_image))
    J_style = compute_style_cost(model, STYLE_LAYERS)

    J = total_cost(J_content, J_style, alpha)
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(J)

    generated_image = generate_noise_image(content_image)
    generate(sess, model, generated_image, num_iter, str(c))

    sess.close()
    c += 1

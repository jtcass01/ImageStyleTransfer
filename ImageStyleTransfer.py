import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import sys
from tkinter import filedialog
import cv2

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

def select_file():
    return filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image file", filetypes = (("jpeg files", "*.jpeg"), ("all files", "*.*"), ("png files", "*.png")))

def select_content_and_style_images():
    #Images for report
    report_images = []

    path_content_image = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image file", filetypes = (("jpeg files", "*.jpeg"), ("all files", "*.*"), ("png files", "*.png")))
    name_content_image = os.path.basename(path_content_image)
    content_image_orig = cv2.imread(path_content_image)
    content_image_orig = cv2.resize(content_image_orig, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
    report_images.append(content_image_orig)
    content_image = reshape_and_normalize_image(content_image_orig)

    path_style_image = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image file", filetypes = (("jpeg files", "*.jpeg"), ("all files", "*.*"), ("png files", "*.png")))
    name_style_image = os.path.basename(path_style_image)
    style_image_orig = cv2.imread(path_style_image)
    style_image_orig = cv2.resize(style_image_orig, dsize=(400, 300), interpolation=cv2.INTER_LINEAR)
    report_images.append(style_image_orig)
    style_image = reshape_and_normalize_image(style_image_orig)

    path_generated_image = 'output/' + name_content_image + '_' + name_style_image + '.png'

    return report_images, content_image, style_image, path_generated_image, name_content_image, name_style_image


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [-1]))
    
    # compute the cost with tensorflow (≈1 line)
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)
    ### END CODE HERE ###
    
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
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(a_S, [n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum((GS- GG)**2) / (4 * n_C**2 * (n_W*n_H)**2)    
    
    return J_style_layer

def compute_style_cost(model, sess, STYLE_LAYERS = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]):
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
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha*J_content + beta*J_style
    
    return J

def generate_image(content_image, style_image, name_content_image, name_style_image):
    # Reset the graph
    tf.reset_default_graph()

    # Start interactive session
    sess = tf.InteractiveSession()

    #Initialize a noisy image by adding random noise to the content_image
    generated_image = generate_noise_image(content_image)

    #load VGG19 model
    model = load_vgg_model("PretrainedModel/imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.  
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)


    # Assign the input of the model to be the "style" image 
    sess.run(model['input'].assign(style_image))

    # Compute the style cost
    J_style = compute_style_cost(model, sess, STYLE_LAYERS)

    alpha = 10
    beta = 40
    iterations = 200
    J = total_cost(J_content, J_style, alpha, beta) #10,40

    # define optimizer (1 line)
    optimizer = tf.train.AdamOptimizer(2.0)

    # define train_step (1 line)
    train_step = optimizer.minimize(J)


    def model_nn(sess, input_image, path_generated_image, num_iterations = 200):
    
        # Initialize global variables (you need to run the session on the initializer)
        sess.run(tf.global_variables_initializer())
    
        # Run the noisy input image (initial generated image) through the model. Use assign().
        sess.run(model["input"].assign(input_image))
    
        for i in range(num_iterations):
    
            # Run the session on the train_step to minimize the total cost
            sess.run(train_step)
        
            # Compute the generated image by running the session on the current model['input']
            generated_image = sess.run(model["input"])

            # Print every 20 iteration.
            if i%20 == 0:
                Jt, Jc, Js = sess.run([J, J_content, J_style])
                print("Iteration " + str(i) + " :")
                print("total cost = " + str(Jt))
                print("content cost = " + str(Jc))
                print("style cost = " + str(Js))
            
                # save current generated image in the "/output" directory
                print("Logging @ ", "output/" + str(i) + ".png")
                save_image("output/" + str(i) + ".png", generated_image)
    
        # save last generated image
        save_image(path_generated_image, generated_image)
    
        return generated_image

    model_nn(sess, generated_image, path_generated_image, iterations)

    #Read the images we just generated
    read_generated = cv2.imread(path_generated_image)
    report_images.append(read_generated)

    #Generate report
    fig, axes = plt.subplots(1,3, figsize = (15,4))
    titles = ['Content', 'Style', 'Generated']
    for i,ax in enumerate(axes):
        ax.set_title(titles[i])
        ax.imshow(report_images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")

    fig.savefig('summary/' + name_content_image + '_' + name_style_image +'_summary.png', dpi=fig.dpi)
    print("Finish...")

if __name__ == "__main__":
    # Read Images
    report_images, content_image, style_image, path_generated_image, name_content_image, name_style_image = select_content_and_style_images()

    generate_image(content_image, style_image, name_content_image, name_style_image)

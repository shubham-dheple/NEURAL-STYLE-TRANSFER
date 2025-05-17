# NEURAL-STYLE-TRANSFER
*Company*:CODETECH IT SOLUTION 

*NAME*:Shubham Dheple

*Intern ID *:C0DF277

*DOMAIN*:AIML

*DURATION*:4 weeks 

*Mentor*:NEELA SANTOSH

Image Loading and Preprocessing
The script begins by defining the load_image() function, which loads an image from disk and preprocesses it to be compatible with the VGG19 model input requirements. This involves resizing the image to a manageable size, converting it to a tensor, and normalizing it with specific mean and standard deviation values used during VGG19’s training. The image tensor is then unsqueezed to add a batch dimension and moved to the appropriate device (GPU or CPU).

Device Configuration
The script detects whether a CUDA-compatible GPU is available, which significantly speeds up the computation. If no GPU is found, it falls back to the CPU, although this will make the process slower.

Loading the Pre-trained VGG19 Model
VGG19, a deep CNN trained on millions of images, is used here as a feature extractor. The fully connected layers are discarded, keeping only convolutional layers, which are ideal for capturing image texture and structure. The parameters of the model are frozen to prevent training.

Extracting Features for Content and Style
The function get_features() passes an image through the VGG19 layers and extracts outputs from specific layers known to represent style and content effectively. The content is typically captured from a deeper layer (conv4_2), while style is captured from several earlier convolutional layers to represent textures and patterns.

Computing the Gram Matrix for Style Representation
The gram_matrix() function calculates the Gram matrix of a feature map, which encodes the correlations between filter responses and effectively captures the style information like textures and colors.

Initialization of the Target Image
The stylized image, called the target, is initially a clone of the content image. This image is set to require gradients since it will be optimized during the process to minimize the difference between its features and those of the content and style images.

Loss Calculation and Optimization
The script uses a weighted combination of content loss (difference between target and content features) and style loss (difference between target and style Gram matrices) to guide the image transformation. Style layers have different weights indicating their importance. The total loss is backpropagated, and the Adam optimizer updates the target image pixels directly.

Iteration and Progress
The optimization runs for a set number of steps (2000 iterations). At every 400th step, the script prints the current loss value, giving insight into the progress. Since this is a computationally intensive process involving backpropagation through a deep network for many iterations, it takes significant time to complete — especially on a CPU.

Displaying the Result
Once the optimization loop finishes, the resulting stylized image is converted back from tensor form to a NumPy array suitable for visualization. The script then displays the image using Matplotlib, showing the content image transformed with the style characteristics of the second image.

Time Consideration
It’s important to note that Neural Style Transfer is a time-consuming process because it involves iterative optimization of pixel values through multiple forward and backward passes of a deep neural network. Each iteration updates the target image slightly to better match the desired style and content features, requiring significant computational resources and time — especially if running on a CPU or with large images. Using a GPU accelerates this process substantially, but even then, depending on image size and number of iterations, it can take several minutes or more.

Summary
In essence, this script creatively blends two images — transferring artistic style onto content — using a well-established deep learning model. The combination of feature extraction, Gram matrix computation, and iterative optimization produces visually stunning results that mimic artistic painting styles or image filters. While computationally intensive, this technique opens exciting opportunities for art, design, and creative AI applications.

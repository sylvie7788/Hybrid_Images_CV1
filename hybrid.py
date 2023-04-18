import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # # raise Exception("TODO in hybrid.py not implemented")
    
    height, width = img.shape[0] , img.shape[1]
    m, n = kernel.shape[0]//2 , kernel.shape[1]//2

  
    if img.ndim ==2:
        # matrix = np.zeros((height,width)).reshape(height,width)
        matrix = np.zeros(img.shape, dtype=np.float64, order='C')
        for i in range(height):
            for j in range(width):
                for u in range( -m, m +1):
                    for v in range( -n, n +1):
                        if ( 0<=(i+u)<height ) and ( 0<=(j+v)<width ) :
                            matrix[i][j] +=  (kernel[m+u][n+v] * img[i+u][j+v])
        return matrix        
    
    if img.ndim ==3:
        # matrix = np.zeros((height,width,3)) 
        matrix = np.zeros(img.shape, dtype=np.float64, order='C')
        for i in range(height):
            for j in range(width):
                for k in range(3):
                    for u in range( -m, m+1):
                        for v in range( -n, n+1 ):
                            if (0<=(i+u)<height) and (0<=(j+v)< width ):
                                matrix[i][j][k] += (kernel[m+u][n+v]*img[i+u][j+v][k])
        return matrix        
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")

    new_kernel = np.flipud(kernel)
    flipped_kernel = np.fliplr(new_kernel)
    return cross_correlation_2d(img, flipped_kernel)

    # height, width = img.shape[0],img.shape[1]
    # m, n = kernel.shape[0]//2, kernel.shape[1]//2

    # if img.ndim ==2:
    #     matrix = np.zeros(img.shape, dtype=np.float64, order='C')
    #     for i in range(height):
    #         for j in range(width):
    #             for u in range( -m, m+1 ): #kernel height
    #                 for v in range( -n, n+1 ):
    #                     # if (0<=(i-u)< height) and (0<=(j-v)<width):
    #                     # if (i-u) in range(height) and (j-v) in range(width) :
    #                     if (i-u) in range(height) and (j-v) in range(width):
    #                         matrix[i][j]+= (kernel[m-u][n-v] * img[i-u][j-v])
    #     return matrix        
    
    # else: # img.ndim ==3
    #     matrix = np.zeros(img.shape, dtype=np.float64, order='C')
    #     for i in range(height):
    #         for j in range(width):
    #             for k in range(3):
    #                 for u in range( -m, m+1):
    #                     for v in range( -n,n+1 ):
    #                         # if (0<=(i-u)< height) and (0<=(j-v)<width):
    #                         # if (i-u) in range(height) and (j-v) in range(width):
    #                         if (i-u) in range(height) and (j-v) in range(width):
    #                             matrix[i][j][k]+= (kernel[m-u][n-v] * img[i-u][j-v][k])
                    
    #     return matrix        
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
  
    gauss = np.zeros((height, width))
    center_x = width //2
    center_y = height //2

    sum = 0
    for i in range(height):
        for j in range(width):
            x = j - center_x
            y = i - center_y
            gauss[i,j] = np.exp( -(x**2+y**2)/(2*sigma**2 ) )
            sum+= gauss[i,j]
    return gauss/sum

    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    filtered_img = convolve_2d(img, kernel)
    return filtered_img
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    filtered_img = convolve_2d(img, kernel)
    return (img-filtered_img)
    # raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

# def main():
#     # img = np.arange(105).reshape(5,7,3)
#     # print(img,"\n",img[1,1,0])
#     # kernel = np.ones((5,3))
#     # print("img:\n",img,"\nkernel\n",kernel)
#     # G=cross_correlation_2d(img, kernel)
#     # G=convolve_2d(img, kernel)
#     # print("G\n",G)

#     img = np.arange()



# if __name__ == "__main__":
#     main()
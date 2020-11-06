import os
import numpy as np
from pylab import *
import timeit
import Image
from scipy.misc import imresize
import math


def get_imlist(path, format):
    # Function from the book: Programming Computer Vision with Python by Jan Erik Solem
    """ Returns a list of filenames for all jpg images in a directory. """
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(format)]
    

 
def crop_out_borders(img, white, black):
    # Function to crop out the white and black borders of the image img
    # It goes through diagonal to find the end of the white and black borders
    # Parameters:
    #       im:     array of the image
    #       white:  int, threshold of the color white to be removed 
    #       black:  int, threshold of the color black to be removed
    #return:
    #       im1, im2, im3: the three channels of the cropped image
    i,j = 0,0
    rows = len(img)
    cols = len(img[0])
    
    while (img[i,i] > white):
        i=i+1
    while (img[rows-j-1,cols-1-j] > white):
        j=j+1
    while (img[i,i] < black):
        i=i+1
    while (img[rows-j-1,cols-1-j] < black):
        j=j+1
    
    new_rows = rows-i-j
    col = cols-i-j
    lines = new_rows/3
    
    im1 = img[i:lines+i,i:col-j]            # blue
    im2 = img[lines+i:2*lines+i,i:col-j]    # green
    im3 = img[2*lines+i:3*lines+i,i:col-j]  # red
    
    return im1, im2, im3


def get_average(img, u, v, n):
    
    s = sum(array(img[u-n:u+n+1,v-n:v+n+1]).flatten())
    return float(s)/(2*n+1)**2
  
    
def get_standard_deviation(img, avg, u, v, n):
    arr = array(img[u-n:u+n+1,v-n:v+n+1]).flatten()
    s = sum((arr-avg)**2)
    
    return (s**0.5)/(2*n+1)


def ssd(img1, img2):
    return float(np.sqrt(float(np.sum((array(img1).flatten()-array(img2).flatten())**2))))/(img1.shape[0]*img1.shape[1])
    

def ncc(img1, u1, v1, avg2, stdDeviation2, s2, n):
    # Calculates the norm cross-correlation between img1 and img2(from pre-calculated data)
    avg1 = get_average(img1, u1, v1, n)
    stdDeviation1 = get_standard_deviation(img1, avg1, u1, v1, n)
    
    s1 = array(img1[u1-n:u1+n+1,v1-n:v1+n+1]).flatten()
    s = sum((s1-avg1)*(s2-avg2))
    
    return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)


def match_template(img1, img2, d, n, alg):
    # Function to return the matrix of results of ncc
    # Paramaters:
    #       img1:   array of the image to matched
    #       img2:   array of the image to match as a template(filter)
    #       d:      range of displacement
    #       n:      dimension of the template in img2 
    #       alg:    applied algorithm (ssd/ncc)
    # Return:
    #       result: The matrix with values of the algorithm (ncc, ssd)
    result = zeros((2*d, 2*d), dtype=float64)
    
    u,v=len(img1)/2, len(img1[0])/2

    if alg=="ssd":
        im2 = img2[u-n:u+n+1,v-n:v+n+1]
        for x in range(-d,d):
            for y in range(-d,d):
                result[x+d,y+d] = ssd(img1[u+x-n:u+x+n+1, v+y-n:v+y+n+1], im2)
    elif alg=="ncc":
        avg2 = get_average(img2, u, v, n)
        std2 = get_standard_deviation(img2, avg2, u, v, n)
        s2 = array(img2[u-n:u+n+1,v-n:v+n+1]).flatten()
    
        for x in range(-d,d):
            for y in range(-d,d):
                result[x+d,y+d] = ncc(img1, u+x,v+y, avg2, std2, s2,n)
    return result


def match_images(im1, im2, im3, d,n, alg):
    # This Function matches im2 and im3 with im1.
    # Paramaters:
    #       img1:   array of the image to matched
    #       img2:   array of the image to match as a template(filter)
    #       d:      range of displacement
    #       n:      dimension of the template in img2 
    #       alg:    applied algorithm (ssd/ncc)
    # Return:
    #       x2,y2, x3,y3:   The coordinates with the best solution from the match_template function
    result2 = match_template(im1, im2, d,n, alg)
    result3 = match_template(im1, im3, d,n, alg)
    ij2,ij3=(0,0),(0,0)
    if alg=="ncc": 
        ij2 = np.unravel_index(np.argmax(result2), result2.shape)
        ij3 = np.unravel_index(np.argmax(result3), result3.shape)
    elif alg=="ssd":
        ij2 = np.unravel_index(np.argmin(result2), result2.shape)
        ij3 = np.unravel_index(np.argmin(result3), result3.shape)
    x2, y2 = ij2[::1]
    x3, y3 = ij3[::1]
    
    return x2, y2, x3, y3

def align_images(im1, im2, im3, di2, dj2, di3, dj3, ch1, ch2, ch3):
    # Mapping the new position: how much the image displaced from the center
    # Parameters:
    #       im1:    Array for image1, blue-channel
    #       im2:    Array for image2, green-channel
    #       im3:    Array for image3, red-channel      
    #       di2:    Displacement in x-axis for image 2
    #       dj2:    Displacement in y-axis for image 2
    #       di3:    Displacement in x-axis for image 3
    #       dj3:    Displacement in y-axis for image 3
    #       ch1:    Channel for the image1
    #       ch2:    Channel for the image2
    #       ch3:    Channel for the image3
    # Return:
    #       result: The new image with all the three aligned channels
    result = zeros((im2.shape[0], im2.shape[1],3))
    result = result.astype(uint8)
    result[:,:,ch1] = im1
    
    if (di2>=0 and dj2>=0):
        result[di2:im1.shape[0], dj2:im1.shape[1], ch2] = im2[0:im2.shape[0]-di2, 0:im2.shape[1]-dj2]
    elif (di2>0 and dj2<0):
        result[di2:im1.shape[0], 0:im1.shape[1]+dj2, ch2] = im2[0:im2.shape[0]-di2, -dj2:im2.shape[1]]
    elif (di2<0 and dj2<0):
        result[0:im1.shape[0]+di2, 0:im1.shape[1]+dj2, ch2] = im2[-di2:im2.shape[0], -dj2:im2.shape[1]]
    else:
        result[0:im1.shape[0]+di2, dj2:im1.shape[1], ch2] = im2[-di2:im2.shape[0], 0:im2.shape[1]-dj2]

    if (di3>=0 and dj3>=0):
        result[di3:im1.shape[0], dj3:im1.shape[1], ch3] = im3[0:im3.shape[0]-di3, 0:im3.shape[1]-dj3]
    elif (di3>0 and dj3<0):
        result[di3:im1.shape[0], 0:im1.shape[1]+dj3, ch3] = im3[0:im3.shape[0]-di3, -dj3:im3.shape[1]]
    elif (di3<0 and dj3<0):
        result[0:im1.shape[0]+di3, 0:im1.shape[1]+dj3, ch3] = im3[-di3:im3.shape[0], -dj3:im3.shape[1]]
    else:
        result[0:im1.shape[0]+di3, dj3:im1.shape[1], ch3] = im3[-di3:im3.shape[0], 0:im3.shape[1]-dj3]

    return result
  

def resize_images(im1, im2, im3, rate_x, rate_y ):
    # Resizes the three arrays of images im1, im2, im3 into a new dimension (w/rate_x, h/rate_y)
    # Parameters:
    #       im1..im3:   three images to resized
    #       rate_x:     The number of times the image will be resize in colunms   
    #       rate_y:     The number of times the image will be resize in lines
    # Return:
    #       The resized three images
    
    r_im1 = imresize(im1, (im1.shape[0]/rate_x, im1.shape[1]/rate_y))
    r_im2 = imresize(im2, (im2.shape[0]/rate_x, im2.shape[1]/rate_y))
    r_im3 = imresize(im3, (im3.shape[0]/rate_x, im3.shape[1]/rate_y))

    return r_im1, r_im2, r_im3
    

def save_image(img, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    im_tosave = Image.fromarray(img)
    print path+name
    im_tosave.save(path+name)


if __name__ == '__main__':
    start_final = timeit.default_timer()
    os.chdir('c:/Dropbox/University of Toronto/Visual Computing/assignments/project1/images/')
    n=0                 # Dimension of the template 2Nx2N
    d=10                # Range of displacement -d:+d
    ratio=10            # Ratio for resizing
    max_resol=1100      # Threshoud for resizing
    for format in ([".png", ".jpg"]):
        for alg in (["ssd", "ncc"]):
            if alg=="ncc":
                n=40
            else:
                n=70 # best result
            start_group = timeit.default_timer()
            print alg, format
            path="output/noalign/"+format[1:]+'/'+alg+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            f_output = open(path+'output.txt', 'w+')
            for image_name in get_imlist('./', format):   
                start = timeit.default_timer()
                r=1 # backup for ratio
                img = imread(image_name)
                if format==".png":
                    img[:,:] = img*255

                img = img.astype(uint8)
                print >> f_output, image_name[2:], img.shape, img.dtype
                im1, im2, im3 = crop_out_borders(img, 240, 10)
                resized=False

                if im1.shape[0]>max_resol or im1.shape[1]>max_resol:
                    resized=True; r=ratio
                    r_im1, r_im2, r_im3 = resize_images(im1,im2,im3, ratio, ratio)
                
                if resized:
                    x2, y2, x3, y3 = match_images(r_im1, r_im2, r_im3, d,n, alg)
                else:
                    x2, y2, x3, y3 = match_images(im1, im2, im3, d,n, alg)
                print >> f_output, "displacement: img2", r*(x2-d),r*(y2-d), "img3", r*(x3-d),r*(y3-d)
                              
                result = align_images(im1, im2, im3, r*(x2-d),r*(y2-d), r*(x3-d),r*(y3-d), 2, 1, 0)    
                result = result.astype(uint8)
                
                stop = timeit.default_timer()
                print >> f_output, "time:", stop-start, 'sec'
                print >> f_output, "--------------"
                save_image(result, path, image_name[2:])
            stop = timeit.default_timer()
            print >> f_output, 'group time:', stop-start_group, 'sec'
            print >> f_output, '#################'
            f_output.close()
    stop = timeit.default_timer()
    print "Total runtime:", stop-start_final, 'sec'

    
        

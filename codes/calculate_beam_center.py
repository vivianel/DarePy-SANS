def calculate_center(det_img, file_name):
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    # Read in the image
    # Note - intensities are floating point from [0,1]
    det_img = np.where(det_img<=0, 1e-8, det_img)
    det_img = np.log(det_img)
    cutoff = det_img[det_img>0].max()/1.5 #
    im = np.where(det_img<cutoff, 0, det_img)
    im = np.where(im>=cutoff, 1, im)
    # Threshold the image first then clear the border
    #img = clear_border(im > (200.0/255.0))
    
    # Find coordinates of thresholded image
    y,x = np.nonzero(im)
  
    # Find average
    xmean = x.mean()
    ymean = y.mean()
  
    # Plot on figure
    plt.figure()
    plt.imshow(np.dstack([im,im,im]))
    plt.plot(xmean, ymean, 'r+', markersize=10)
  
    bc_x = xmean
    bc_y =  ymean
    plt.title('x_center = ' + str(round(bc_x, 2)) + ', y_center =' + str(round(bc_y, 2)) + ' pixels')
    
    # Show image and make sure axis is removed
    plt.axis('off')
    #plt.show()
    plt.savefig(file_name)
    plt.close('all')
    
    return bc_x, bc_y
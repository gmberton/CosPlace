import matplotlib.pyplot as plt

#Histogram showing how many cells contain a given number of images. (Density?)


#Histogram showing number of images per class

#classes is an array of tuples (UTM_east, UTM_north, heading) = class_id
#images is a dict in which for each class_id, there is an array of images paths
def plot_histogram(classes, images):
    #x = classes
    #y = num. of images
        
    plt.figure()
    plt.title("Distribution")
    plt.xlabel("Classes")
    
    for i in range(len(classes)):
        plt.hist(images[classes[i]], bins = 10, density = True, alpha = 0.4)

    plt.legend()
    #plt.savefig('%s/%s_%d.pdf' % (folder,name,index))
    plt.show()


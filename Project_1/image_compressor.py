import numpy as np

class ImageCompressor:
    """
      This class is responsible to
          1. Learn the codebook given the training images
          2. Compress an input image using the learnt codebook
    """
    def __init__(self,n_components = 30):
        """
        Feel free to add any number of parameters here.
        But be sure to set default values. Those will be used on the evaluation server
        """
        
        # Here you can set some parameters of your algorithm, e.g.
        self.codebook = None
        self.mean_ = None
        self.eigenvectors = None
        self.n_components = n_components

    def get_codebook(self):
        """ Codebook contains all information needed for compression/reconstruction """
        # TODO ...
        return self.codebook.astype(np.float16) 


    def train(self, train_images):
        """
        Training phase of your algorithm - e.g. here you can perform PCA on training data
        
        Args:
            train_images  ... A list of NumPy arrays.
                              Each array is an image of shape H x W x C, i.e. 96 x 96 x 3
        """
        # TODO ...
        n = len(train_images)

        flat_images = np.array(train_images).reshape(n, -1) / 255.0
        flat_images = flat_images.T

        mean_ = np.mean(flat_images, axis=1).astype(np.float16)
        self.mean_ = mean_.astype(np.float16)
        centered_images = flat_images - mean_[:, np.newaxis]
        
        U, S, V = np.linalg.svd(centered_images, full_matrices=False)
        arrange_indices = np.argsort(S)[::-1]
        U = U[:, arrange_indices]

        eigenvectors = U[:, :self.n_components].astype(np.float16)
        self.eigenvectors = eigenvectors.astype(np.float16)
        self.codebook  = np.hstack((mean_[:, np.newaxis], eigenvectors)).astype(np.float16)


    def compress(self, test_image):
        """ Given an array of shape H x W x C return compressed code """
        # TODO ...
        flat_ = (test_image.flatten()/255.0 - self.mean_) 
        compressed_ = np.matmul(flat_, self.eigenvectors)
        return compressed_.astype(np.float16)

class ImageReconstructor:

    """ This class is used on the server to reconstruct images """
    def __init__(self, codebook):
        """ The only information this class may receive is the codebook """
        self.codebook = codebook

    def reconstruct(self, test_code):
        """ Given a compressed code of shape K, reconstruct the original image """
        # TODO ...
        eigenvectors = self.codebook[:,1:]
        mean = self.codebook[:,0]

        reconstructed_image = np.matmul(test_code, eigenvectors.T) 

        reconstructed_image = (reconstructed_image + mean) 
        reconstructed_image[reconstructed_image>1]=1
        reconstructed_image[reconstructed_image<0]=0
        

        reconstructed_image = reconstructed_image.reshape((96, 96, 3))

        return (reconstructed_image *255).astype(np.uint8)
    
    
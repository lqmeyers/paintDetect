## making an easy to import and call class to semantically segment images of bees
import torch
import os 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import PIL.JpegImagePlugin
from matplotlib import pyplot as plt
import sys

sys.path.insert(0,"./Pytorch-UNet/")
from utils.data_loading import BasicDataset
from unet import UNet



os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU with index 1

trained_models = {
    'full':'/home/lqmeyers/paintDetect/wandb/run-20230608_174519-mnej4wjn/files/20230608_174519_model.pth',
    'head':'/home/lqmeyers/paintDetect/wandb/run-20230814_185421-qonmcmyj/files/20230815_202743_model.pth',
    'paint':'',
    'thorax':'/home/lqmeyers/paintDetect/models/thorax_segmentation_best_run_20230526_003438.pth',
    'abdomen':'',
    'background':'',
}

###-------------necessary non-class funcs (DOCUMENT !!!!!!!!!!!!!!!!!!!!!!!!!)
def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()

def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

###------------------------------Class for segmentation of image

class Segmentation():
    def __init__(self,model_lib,image):
        if type(image) != PIL.JpegImagePlugin.JpegImageFile:
            image = Image(image)
        self.model_lib = model_lib
        self.parse_models(model_lib)
        self.image = image 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.head = self.mask_2_image(self.predict(self.image,self.load_model('head')))
        self.thorax = self.mask_2_image(self.predict(self.image,self.load_model('thorax')))
        self.full_segmentation = self.mask_2_image(self.predict(self.image,self.load_model('full')))
        #return self.full_segmentation

    """
    @staticmethod
    def set_type(type):
        '''sets vars based on type of segmentation wanted'''
    """  

    def parse_models(self,model_lib):
        '''expects a model lib containing'''
        self.full_model = model_lib['full']
        self.head_model = model_lib['head']
        self.paint_model = model_lib['paint']
        self.thorax_model = model_lib['thorax']
        self.abdomen_model = model_lib['abdomen']
        self.background_model = model_lib['background']
        self.models = [self.full_model,self.head_model,self.paint_model,self.thorax_model,self.abdomen_model,self.background_model]
        
    def load_model(self,model_key):
        model = self.model_lib[model_key]
        if model_key == 'full':
            n_class= 5 
        else:
            n_class = 2
        print("Num classes used for prediction",n_class)
        net = UNet(n_channels=3, n_classes=n_class, bilinear=False)
        print(f'Loading model {model}')
        print(f'Using device {self.device}')
        net.to(device=self.device)
        state_dict = torch.load(model, map_location=self.device)
        self.mask_values = state_dict.pop('mask_values', [0, 1])
        net.load_state_dict(state_dict)
        print('Model loaded!')
        return net

    '''
    def load_models(self.models):
        for i, m in enumerate(self.models):
            pass
    '''
    
    def predict(self,image,model):
        '''returns a segmentation image'''
        self.current_model = model
        mask = predict_img(net=model,
                        full_img=image,
                        scale_factor=1,
                        out_threshold=.5,
                        device=self.device)
        return mask
    
    def mask_2_image(self,mat):
        '''returns an rgb PIL.Image from a class map'''
        mask_vals = self.mask_values
        return mask_to_image(mat,mask_vals)
    
    '''
    def save(self,img):
        saves the PIL image
        img.save()
    '''

"""
test_image = Image.open('/home/lqmeyers/paintDetect/data/images/testing/f1.2x2022_06_22.mp4.track000004.frame001173.jpg')
plt.imshow(test_image)
plt.show()                        
im_sg = Segmentation(trained_models,test_image)

#print(vis.full_segmentation[100])

out_filename = 'test.png'
im_sg.head.save(out_filename)
print(f'Mask saved to {out_filename}')
"""
   


"""

###--------------------Ok im gonna try to do a base class implementation of this for each type of segmentation


from abc import ABC, abstract_method
 

class mask(ABC):
    '''This class is an abstract base class (ABC) for UNet segmentations of images.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call segmentation.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

        ----- need to update with specific but copied to have template for docs 
    '''
    def __init__(self,image):
        #if type(image) != PIL.JpegImagePlugin.JpegImageFile:
            #image = Image(image)
        #self.model_lib = model_lib
        #self.parse_models(model_lib)
        self.image = image 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #self.full_segmentation = self.mask_2_image(self.predict(self.image,self.load_model('full')))
    #return self.full_segmentation

    @abstract_method
    def load_model(self,input):
        '''load the specific model to be used in prediction for each type of segmentation'''
        pass 

    def predict(self,image,model):
        '''returns a segmentation image'''
        mask = predict_img(net=model,
                        full_img=image,
                        scale_factor=1,
                        out_threshold=.5,
                        device=self.device)
        return mask
    
    def mask_2_image(self,mat,vals):
        '''returns an rgb PIL.Image from a class map'''
        return mask_to_image(mat,vals)
    

class Head(Segmentation):
    '''a subclass of segmentation that creates a mask of the head from an image'''
    def __init__(self,image):
        mask.__init__(self,image)
        self.n_classes = 2 
        self.segmentation = 

    
    def load_model(self,path):
        '''loads the model at path'''
"""
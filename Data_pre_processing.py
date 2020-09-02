import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms.functional as adjust


to_Tensor = transforms.ToTensor()
to_pil_image = transforms.ToPILImage()

#parameter generation
def parameter_create(length, brightness_low = -60/255, brightness_top = 50/255, 
                           contrast_low = 0.5, contrast_top = 1.4,
                            hue_low = -0.1, hue_top = 0.2):
    np.random.seed(0)
    brightness = np.around(np.random.uniform(brightness_low, brightness_top, (length, 1)), decimals=2).astype(np.float32)
    contrast = np.around(np.random.uniform(contrast_low, contrast_top, (length, 1)), decimals=2).astype(np.float32)
    hue = np.around(np.random.uniform(hue_low, hue_top, (length, 1)), decimals=2).astype(np.float32)

    parameter = np.concatenate((brightness, contrast, hue), axis=1)
    parameter_T = parameter_inverse(parameter)
    return parameter, parameter_T

def parameter_inverse(parameter):
    #The inverse of brightness, contrast are 1/brightness and 1/contrast
    brightness_T = (-parameter[:,0]).reshape(-1,1)
    contrast_T = (1/parameter[:,1]).reshape(-1,1)
    #The inverse of hue plus hue equals 0
    hue_T = (-parameter[:,2]).reshape(-1, 1)
    
    parameter_T = np.concatenate((brightness_T, contrast_T, hue_T), axis=1)
    return parameter_T

#parameter adjusting
def parameter_adjust(img, parameter):
    #Adjust brightness and contrast
    img = torch.clamp(input = (parameter[1] * img + parameter[0]), min = 0, max =1)
    #Adjust hue
    img = adjust.adjust_hue(to_pil_image(img), parameter[2])
    img = to_Tensor(img)
    return img

#change parameters of all images
def parameter_adjust_loop(imgs, parameters, weight = 60, height = 80):
    l = torch.zeros((len(parameters),3, weight, height))
    for i in range(len(parameters)):
        l[i] = parameter_adjust(imgs[i][0], parameters[i])
    
    return l

#save images and corresponding labels as pairs in to new list
def dataset_preprocessing(dataset, parameter_T):
    new_set = []
    for i in range(len(parameter_T)):
        new_set.append([torch.Tensor(dataset[i]), parameter_T[i]])
    return new_set
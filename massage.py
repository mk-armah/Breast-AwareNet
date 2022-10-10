import os
import cv2
import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from Config import config

        
class BreastMassage:
  """
    Fun Tips:
    Use light, medium, and firm pressure. 
    Squeeze the nipple; check for discharge and lumps.
    Repeat these steps for your left breast.
    """

  def __init__(self):
    #test will be conducted three times
    """start by testing for normal conditions of the breast, but what if you spot something ?, 
      Proceed to checking for how big or small the changes you sense is;
      And thats whether its Benign or Malignant, simply make that assumption,
      Even though proper diagnosis is supposed to be done by a physician after a careful examination of your Ultrasound scan
      or by an AI predicting the cancer stage from a given Ultrasound Image.. e.g BreastAwareNet"""

    self.classes = ["benign","malignant","normal"] 

  def smooch(self,func):

    def wrapper(*args,**kwargs):

      """
      Breast Cancer Awareness Tip:
        >>> Now wrap your hands around the Breast,
            Smooch Gently, Look for any changes in the contour, any swelling, or dimpling of the skin, or changes in the nipples.
      """
      
      for i in self.classes: 

        path = os.path.join(config.working_dir,i)

        masks,images = func(path)

        Img_train, Img_test, mask_train, mask_test = train_test_split(
        images, masks, test_size = 0.30, random_state=42)

        yield Img_train, Img_test, mask_train, mask_test #yields per round

    return wrapper


  @smooch
  def squeeze(self,path:str):
    """
    splits each class of the data into segmentation mask and and original image
    and also removes repeatitive images and masks

    Breast Cancer Awareness Tip:
      >>> squeezing is will be done during smooching.. so dont forget to smooch
       You and I know you have to squeeze (gently) during smooching :) 
      """
    masks = []
    images = []
    odds = []

    for path in os.listdir(path):
      if path.__contains__("mask.png"):
        masks.append(path)
      elif path.endswith(").png"):
        images.append(path)
      else:
        odds.append(path)

    return masks,images


  def __call__(self):
      
    benign = next(self.squeeze()) #round one
    malignant = next(self.squeeze()) #round two
    normal = next(self.squeeze()) #round three

    training_images = benign[0]+malignant[0]+normal[0]
    testing_images = benign[1]+malignant[1]+normal[1]
    training_masks = benign[2]+malignant[2]+normal[2]
    testing_masks = benign[3]+malignant[3]+normal[3]

    return training_images, testing_images,training_masks,testing_masks

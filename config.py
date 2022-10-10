from pydantic import BaseModel
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class Config:
  """model configurations and setup variables"""
  batch_size:int
  working_dir:str = "/content/drive/MyDrive/Coding-Stuffs/Artificial Intelligence/Computer Vision /Breast_Cancer_Segmentation/Dataset_BUSI_with_GT/"
  seed:int = 111

  def plant_seeds(self):
    """plant seeds for reproduceability
    
    Breast Cancer Awareness Tip:
      >>> It is essential to know which point of the breast to start the massage from, it will help to assert whether you've checked every corner of the breast or not
    """
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.cuda.manual_seed(self.seed)
    torch.backends.cudnn.deterministic = True



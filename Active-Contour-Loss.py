
import torch

def active_contour_loss(y_true, y_pred, weight=10):
  '''
  y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
  weight: scalar, length term weight.
  '''
  # length term
  delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
  delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
  
  delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
  delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
  delta_pred = torch.abs(delta_r + delta_c) 

  epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
  lenth = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.
  
  # region term
  c_in  = torch.ones_like(y_pred)
  c_out = torch.zeros_like(y_pred)

  region_in  = torch.mean( y_pred     * (y_true - C_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
  region_out = torch.mean( (1-y_pred) * (y_true - C_out)**2 ) 
  region = region_in + region_out
  
  loss =  weight*lenth + region

  return loss

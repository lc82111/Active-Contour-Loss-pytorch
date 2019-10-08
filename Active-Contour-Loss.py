
import torch

def active_contour_loss(y_true, y_pred):
  '''
  y_true, y_pred: (B, C, H, W)
  '''
  # length term
  delta_r = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W) 
  delta_c = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)
  
  delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
  delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
  delta_pred = torch.abs(delta_r + delta_c) 

  epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
  w = 1  # weight?
  lenth = w * torch.sum(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper
  
  # region term
  c_in  = torch.ones_like(y_pred)
  c_out = torch.zeros_like(y_pred)

  # TODO why selecting only axis 0
  region_in  = torch.abs(torch.sum( y_pred     * ((y_true - C_in )**2) )) # equ.(12) in the paper
  region_out = torch.abs(torch.sum( (1-y_pred) * ((y_true - C_out)**2) )) # equ.(12) in the paper

  lambdaP = 1 # lambda parameter could be various.
  loss =  lenth + lambdaP * (region_in + region_out) 

  return loss

import torch


def fourier_crop(image: torch.Tensor, fourier_crop_to: int) -> torch.Tensor:
    image_f = torch.fft.fftn(image, dim=(-2, -1))
    
    image_f_shifted = torch.fft.fftshift(image_f, dim=(-2, -1))
    
    h, w = image.shape[-2], image.shape[-1]
    crop_h_start = (h - fourier_crop_to) // 2
    crop_h_end = crop_h_start + fourier_crop_to
    crop_w_start = (w - fourier_crop_to) // 2
    crop_w_end = crop_w_start + fourier_crop_to
    
    image_f_cropped = image_f_shifted[..., crop_h_start:crop_h_end, crop_w_start:crop_w_end]
    
    image_f_cropped_shifted_back = torch.fft.ifftshift(image_f_cropped, dim=(-2, -1))
    
    image_r = torch.fft.ifftn(image_f_cropped_shifted_back, dim=(-2, -1))
    
    return image_r.real
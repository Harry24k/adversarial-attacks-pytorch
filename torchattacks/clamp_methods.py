import torch


def clamp_0_1(original_image, adv_image):
    """
    Clamps the adv_image image between 0 and 1
    @param original_image The original image before perturbation - not used in this method
    @param adv_image The image or images to clamp
    """
    return torch.clamp(adv_image, 0, 1)


def clamp_0_255(original_image, adv_image):
    """
    Clamps the adv_image image between 0 and 1
    @param original_image The original image before perturbation - not used in this method
    @param adv_image The image or images to clamp
    """
    return torch.clamp(adv_image, 0, 255)


def clamp_heuristic(original_image, adv_image):
    """
    Clamps the adv_image image between the original_image min and max.
    This ensures that the adversarial image is valid iff the original_image is valid.
    However this may clamp valid images
    @param original_image The original image before perturbation - not used in this method
    @param adv_image The image or images to clamp
    """
    return torch.clamp(adv_image, torch.min(original_image).item(), torch.max(original_image).item())

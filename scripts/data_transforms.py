import torch
import torchvision.transforms as transforms

def input_transform(image):
    """
    The input MNIST images are all 28 × 28 px. This function will firstly center-crop 
    them to 24 × 24 and then down-sample them to 4 × 4 for MNIST. Then we convert 
    the image pixels into angles for passing them into Rotation gates later for encoding.
    """
    image = transforms.Grayscale(num_output_channels=1)(image)
    image = transforms.ToTensor()(image)
    image = transforms.CenterCrop(24)(image)
    image = transforms.Resize(size = (4,4), antialias=True)(image)
    image = image.squeeze()
    image_pixels = torch.flatten(image)
    angles = torch.sqrt(image_pixels / 256)
    
    return angles

def target_transform(label):
    label_tensor = torch.LongTensor([label])
    one_hot_label = torch.nn.functional.one_hot(label_tensor, 10)
    return one_hot_label.squeeze()

def target_transform_bin(label):
    label_tensor = torch.LongTensor([label])
    one_hot_label = torch.nn.functional.one_hot(label_tensor, 2)
    return one_hot_label.squeeze()
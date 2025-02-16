from torchvision import transforms

def resize_image(image, size=(256, 256)):
    # Create a transform pipeline to resize and convert the image to a tensor.
    transform_pipeline = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    return transform_pipeline(image)



import os
from PIL import Image

class ImageAnalyzer:
    def __init__(self, directory):
        self.directory = directory

    def get_largest_image_dimensions(self):
        max_width = 0
        max_height = 0

        for filename in os.listdir(self.directory):
            if filename.endswith('.jpg'):
                filepath = os.path.join(self.directory, filename)
                with Image.open(filepath) as img:
                    width, height = img.size
                    if width > max_width:
                        max_width = width
                    if height > max_height:
                        max_height = height

        return max_width, max_height

if __name__ == "__main__":
    #Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of script
    dataset_dir = os.path.join(script_dir, "../Dataset/TrainVal/color")  # Adjust path
    
    analyzer = ImageAnalyzer(dataset_dir)
    largest_dimensions = analyzer.get_largest_image_dimensions()
    print(f"Largest image dimensions: {largest_dimensions}")
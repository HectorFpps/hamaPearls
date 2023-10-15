from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from collections import Counter

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def closest_color(pixel_color, palette):
    distances = np.sum((palette - pixel_color) ** 2, axis=1)
    closest = palette[np.argmin(distances)]
    return closest

def main(image_path, hex_palette):
    rgb_palette = np.array([hex_to_rgb(c) for c in hex_palette])
    
    img = Image.open(image_path).convert("RGB")
    img = img.resize((30, 30))
    img_array = np.array(img)

    new_img_array = np.zeros(img_array.shape, dtype=np.uint8)
    color_count = {color: 0 for color in hex_palette}

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            closest = closest_color(img_array[i, j], rgb_palette)
            new_img_array[i, j] = closest
            
            closest_hex = '#' + ''.join([f"{int(x):02x}" for x in closest])
            if closest_hex in color_count:
                color_count[closest_hex] += 1

    new_img = Image.fromarray(new_img_array, 'RGB')
    new_img.save("blueprint_image.jpg")

    plt.imshow(img)
    plt.axis('off')
    plt.title('Hama Pearls Blueprint')
    plt.show()

if __name__ == "__main__":
    hex_palette = ['#feeeca', '#feea48', '#ffe604', '#ffcd00', '#ffb504', '#ff6201',
                   '#9bc702', '#00c46a', '#009c87', '#a4ecf7', '#00e1dd', '#01c9e5',
                   '#46cefe', '#0087d7', '#0055d1', '#170c69', '#9e6dc7', '#e5d0f6',
                   '#e5d0f6', '#ff9cd0', '#ff57c5', '#ff3123', '#fd0000', '#c41c29',
                   '#ffc895', '#febc8c', '#d07b14', '#bb5e3f', '#633a34', '#1e1010',
                   '#ffffff', '#f6f2f3', '#ddd8d5', '#b9b4bb', '#908e99', '#000000']
    
    main("2014ForestHillsDrive.jpg", hex_palette)

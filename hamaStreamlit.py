import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import os

image_folder = "img"
needs_verification = False

#st.markdown(
#    """
#    <style>
#    .sticky {
#        position: fixed;
#        top: 0;
#        width: 100%;
#        z-index: 9999;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True,
#)

# Existing functions
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def closest_color(pixel_color, palette, weights):
    distances = np.sum((palette - pixel_color) ** 2, axis=1)
    weighted_distances = distances * pow((1 - weights), 2)
    closest = palette[np.argmin(weighted_distances)]
    return closest

def process_image(image_path, hex_palette, weights, side=30, progress_bar=None):
    size = (side, side)
    rgb_palette = np.array([hex_to_rgb(c) for c in hex_palette])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)

    new_img_array = np.zeros(img.shape, dtype=np.uint8)
    color_counter = Counter()

    total_pixels = img.shape[0] * img.shape[1]
    processed_pixels = 0

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            closest = closest_color(img[i, j], rgb_palette, weights)
            new_img_array[i, j] = closest
            color_counter[tuple(closest)] += 1
            processed_pixels += 1

            if progress_bar is not None:
                progress_bar.progress(processed_pixels / total_pixels)

    return new_img_array, color_counter

# Streamlit app
st.title('Hama Pearls Blueprint Generator')

#list the images in the image folder
image_list = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

image_path = f"img/{st.selectbox('Select an image', image_list)}"

hex_palette = ['#feeeca', '#feea48', '#ffe604', '#ffcd00', '#ffb504', '#ff6201',
               '#9bc702', '#00c46a', '#009c87', '#a4ecf7', '#00e1dd', '#01c9e5',
               '#46cefe', '#0087d7', '#0055d1', '#170c69', '#9e6dc7', '#e5d0f6',
               '#e5d0f6', '#ff9cd0', '#ff57c5', '#ff3123', '#fd0000', '#c41c29',
               '#ffc895', '#febc8c', '#d07b14', '#bb5e3f', '#633a34', '#1e1010',
               '#ffffff', '#f6f2f3', '#ddd8d5', '#b9b4bb', '#908e99', '#000000']

initial_weights = [0.5 for _ in range(len(hex_palette))]
weights = []

progress_bar = st.progress(0)

new_img_array, color_counter = process_image(image_path, hex_palette, np.array(initial_weights), side=30, progress_bar=progress_bar)
total_pixels = new_img_array.shape[0] * new_img_array.shape[1]

for i, color in enumerate(hex_palette):
    st.markdown(f'<div style="display:inline-block; width: 20px; height: 20px; background-color: {color};"></div>', unsafe_allow_html=True)
    count = color_counter.get(hex_to_rgb(color), 0)
    percentage = (count / total_pixels) * 100
    weight = st.slider(f'Adjust Weight for Color {color} - {percentage:.2f}%', min_value=-1.0, max_value=1.0, value=initial_weights[i], step=0.1, format="%.1f", key=i)
    weights.append(weight)

side = st.slider('Adjust Size of Blueprint', min_value=10, max_value=1000, value=30, step=10, format="%d")

progress_bar = st.progress(0)

if not needs_verification:
    new_img_array, _ = process_image(image_path, hex_palette, np.array(weights), side=side, progress_bar=progress_bar)
    plt.imshow(new_img_array)
    st.image(new_img_array.astype('uint8'), channels="RGB", caption="Generated Hama Pearls Blueprint")

    #st.markdown(
    #"""
    #<div class="sticky">
    #    <img src="your_image_url_here" alt="Your Image" width="200" height="200"/>
    #</div>
    #""",
    #unsafe_allow_html=True,
 #   ##)###########

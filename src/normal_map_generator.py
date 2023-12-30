import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def generate_normal_map():
    input_image_path = filedialog.askopenfilename(title="Select Input Image")
    if not input_image_path:
        return

    image = cv2.imread(input_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (0, 0), 3)
    inverted_image = cv2.bitwise_not(blurred_image)
    gradient_x = cv2.Sobel(inverted_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(inverted_image, cv2.CV_64F, 0, 1, ksize=3)
    normal_map = np.dstack((gradient_x, gradient_y, np.ones_like(gradient_x)))
    normal_map = cv2.normalize(normal_map, None, 0, 255, cv2.NORM_MINMAX)

    output_image_path = filedialog.asksaveasfilename(title="Save Normal Map", defaultextension=".jpg")
    if output_image_path:
        cv2.imwrite(output_image_path, normal_map.astype(np.uint8))

def main():
    root = tk.Tk()
    root.title("Normal Map Generator")

    generate_button = tk.Button(root, text="Generate Normal Map", command=generate_normal_map)
    generate_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()

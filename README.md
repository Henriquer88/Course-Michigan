# Course-Michigan
 ## University of Michigan
```javascript
# Author: Henrique
# This code opens an image, converts it to RGB, creates 9 variations of the image with different color intensities, 
# and then creates a contact sheet with these variations.

from PIL import Image

# Open the image and convert to RGB
image = Image.open("readonly/msi_recruitment.gif")
image = image.convert('RGB')

# Create a list of 9 images with different color variations
variations = []
for channel in range(3):  # Color channels: 0 - Red, 1 - Green, 2 - Blue
    for intensity in [0.1, 0.5, 0.9]:
        modified_image = image.copy()
        
        # Get pixel data as a list
        pixels = list(modified_image.getdata())

        # Adjust specific color channel based on intensity
        for i in range(len(pixels)):
            pixel = list(pixels[i])
            pixel[channel] = int(pixel[channel] * intensity)
            pixels[i] = tuple(pixel)

        # Update image data
        modified_image.putdata(pixels)
        variations.append(modified_image)

# Create a contact sheet for the variations
first_image = variations[0]
contact_sheet = Image.new(first_image.mode, (first_image.width * 3, first_image.height * 3))
x = 0
y = 0

for modified_image in variations:
    # Paste the current variation onto the contact sheet
    contact_sheet.paste(modified_image, (x, y))
    
    # Update position for the next variation
    x += first_image.width
    
    # If x exceeds the width, update y and reset x
    if x == contact_sheet.width:
        x = 0
        y += first_image.height

# Resize and display the contact sheet
contact_sheet = contact_sheet.resize((int(contact_sheet.width / 2), int(contact_sheet.height / 2)))
display(contact_sheet)

```
```javascript

# Author: Henrique
# This code provides functionalities to extract text from images using Pytesseract,
# detect faces in images using OpenCV, and create contact sheets of images.
# The main function unzips a provided ZIP file, extracts text from images,
# searches for a keyword within the extracted text, detects faces in images,
# and creates a contact sheet of images with the keyword or faces.

import zipfile
import os
import pytesseract
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


# Function to extract text from an image using Pytesseract
def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to detect faces in an image using OpenCV
def detect_faces(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

# Function to create a contact sheet of images
def create_contact_sheet(images, columns=5, thumbnail_size=(100,100)):
    num_images = len(images)
    rows = (num_images + columns - 1) // columns
    contact_sheet = Image.new(images[0].mode, (columns * thumbnail_size[0], rows * thumbnail_size[1]))
    x, y = 0, 0
    for img in images:
        img.thumbnail(thumbnail_size)
        contact_sheet.paste(img, (x, y))
        if x + thumbnail_size[0] == contact_sheet.width:
            x = 0
            y += thumbnail_size[1]
        else:
            x += thumbnail_size[0]
    return contact_sheet

# Main function
def main(zip_path, keyword):
    # Unzip the provided ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('images')

    images_with_keyword = []
    # Iterate over each image in the extracted files
    for file_name in os.listdir('images'):
        if file_name.endswith('.png'):
            image_path = os.path.join('images', file_name)
            # Extract text from the image
            text = extract_text(image_path)
            # Search for the keyword within the extracted text
            if keyword in text:
                images_with_keyword.append(Image.open(image_path))
            # Detect faces in the image
            faces = detect_faces(image_path)
            if len(faces) > 0:
                # Draw rectangles around the detected faces
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)
                for (x, y, w, h) in faces:
                    draw.rectangle([x, y, x+w, y+h], outline="red")
                images_with_keyword.append(img)

    # Create a contact sheet of images with the keyword or faces
    contact_sheet = create_contact_sheet(images_with_keyword)
    contact_sheet.show()

if __name__ == "__main__":
    # Provide the path to the ZIP file and the keyword to be searched
    main('images.zip', 'Christopher')

```

# Document Scanner

A Python application that processes images of documents, automatically detects their boundaries, applies perspective correction, and converts them to enhanced PDF files.

## Features

- Automatic document boundary detection
- Perspective correction
- Text skew correction
- Image enhancement and binarization
- Batch processing of multiple images
- Automatic conversion to PDF

## Requirements

```bash
opencv-python
numpy
Pillow
```

## Installation

1. Clone this repository or download the source code
2. Install the required dependencies:
   ```bash
   pip install opencv-python numpy Pillow
   ```

## Usage

1. Create an `input` folder in the project directory
2. Place your document images (PNG, JPG, JPEG) in the `input` folder
3. Run the script:
   ```bash
   python app.py
   ```
4. Processed PDFs will be saved in the `output` folder

## How it works

The application performs the following steps:
1. Loads and preprocesses the image (grayscale conversion and blur)
2. Detects document boundaries using contour detection
3. Applies perspective transformation to get a top-down view
4. Corrects text skew using the Hough transform
5. Enhances the image using adaptive thresholding
6. Saves the result as a PDF

## Error Handling

The script includes error handling for:
- Invalid or unreadable images
- Cases where document boundaries cannot be detected
- File system operations

## Notes

- The input images should have the document clearly visible with good contrast from the background
- Supported input formats: PNG, JPG, JPEG
- Output files are saved as PDF with the same name as the input file 
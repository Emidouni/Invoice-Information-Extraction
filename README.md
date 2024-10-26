# Extract data from invoices.
***


## Target Data Extraction:

  - Date 
  - country
  - Client Name

## Prerequisites

Before you begin, ensure you have installed Anaconda. If not, download it from [Anaconda](https://www.anaconda.com/products/individual).

## Installation

### Cloning the Repository

To get started with this project, first clone the repository on your local machine:
You have two options for cloning the repository: using the command line or GitHub Desktop.

#### Option 1: Using the Command Line
```bash
git clone https://github.com/Emidouni/Invoice-Information-Extraction
```
#### Option 2: Using GitHub Desktop

For a more graphical interface, you can use GitHub Desktop:

- 1.Download and install GitHub Desktop from desktop.github.com.
- 2.Open GitHub Desktop and sign in to your GitHub account.
- 3.Click on File > Clone Repository.
- In the "URL" tab, enter the URL of the repository [https://github.com/TotalEnergiesCode/Extract-General-well-data-from-well-reports-](https://github.com/Emidouni/Invoice-Information-Extraction) and choose the local path where you want to clone the repository.
- 4.Click Clone to start the cloning process.

### Setting Up a Python Environment with Conda
After installing Anaconda , you can create a new Conda environment specifically for this project. This helps to manage dependencies and avoid conflicts with other projects.

Open the Anaconda Prompt or your terminal (make sure Conda is added to your PATH) and run
```bash
conda create --name myenv python=3.9.20
```
Replace myenv with your preferred name for the environment. This command creates a new Conda environment named myenv with Python version 3.8.18
Activate the environment with:
```bash
conda activate myenv
```
After activating the environment, you can proceed with installing other required packages as mentioned in the project's 
```bash
pip install -r requirements.txt
```
In addition to the libraries listed in `requirements.txt`, this project requires Tesseract OCR for text recognition from images.

# Installation Instructions for Tesseract OCR and Poppler

Tesseract OCR is an open-source Optical Character Recognition (OCR) engine used for text recognition in images.

## Tesseract OCR Installation and Configuration

1. **Download Tesseract OCR**:
   - **Windows Users**: Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). Follow the installation instructions provided on the website.

2. **Locate the Tesseract Installation Path**:
   - After installation, locate where Tesseract OCR has been installed on your machine. The default installation path on Windows is usually `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. **Update the Script with Your Tesseract Path**:
   - In your project's script `utils.py`, where Tesseract OCR is utilized, locate the line that sets the `tesseract_cmd` property of `pytesseract`. Replace the existing path with the actual path to your Tesseract installation.
     For example, change the line from:

     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
     ```

     To:

     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\Your\tesseract.exe"
     ```
- Ensure to replace `C:\Path\To\Your\tesseract.exe` with the correct path to where Tesseract OCR is installed on your system.

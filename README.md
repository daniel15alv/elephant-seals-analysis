# Elephant Seal Counter

An automated computer vision pipeline for detecting, counting, and classifying northern elephant seals by age and sex from drone-generated aerial imagery, built in collaboration with Dr. Liwanag’s Vertebrate Integrative Physiology (VIP) Lab at Cal Poly SLO.

## Overview 
Manually counting elephant seals from aerial beach surveys is time-consuming and error-prone. This project automates that process using a multi-stage deep learning pipeline that detects individual and clumped seals, classifies them by sex and age (Adult Male, Adult Female, Pup, Weaner), and aggregates results into population estimates — all accessible through a web UI and CLI tool.

## My Contributions

- **Data annotation & preprocessing** — Processed raw `.tif` aerial imagery into 
  JPEG format for Roboflow ingestion, manually annotated individual seal bounding 
  boxes in Roboflow, and iteratively relabeled training data as the modeling 
  approach evolved (from object detection to single-label classification)

- **Individual seal classification model** — Led development of the ResNet-18 
  classification model for individual seals, including diagnosing and resolving 
  severe class imbalance (adult males represented only 6% of training data) through 
  targeted geometric and photometric augmentation, improving F1-score for adult 
  males from 0 to 0.77

- **Baseline for clump model** — The individual classification pipeline served as 
  the foundation for the clump model architecture; individual seal bounding box 
  extraction scripts and the detection-then-classification paradigm established 
  here were directly extended to handle clumped seal groups

- **Clump classification model (collaborative)** — Contributed to the ResNet-18 
  clump classification model, which outputs Softmax confidence proportions across 
  four demographic categories (Adult Male, Adult Female, Pup, Weaner) and 
  multiplies them against object detection counts to estimate per-class 
  population totals


## Pipeline 
1. **Object Detection** — Roboflow-based YOLO model detects individual seals (blue boxes) and seal clumps (yellow boxes) in mosaic beach images 
2. **Individual Seal Classification** — ResNet-18 model classifies each detected individual seal as Adult Male, Adult Female, or Young Seal 
3. **Clump Classification** — Separate ResNet-18 model with Softmax output estimates demographic proportions within clumped seal groups 
4. **Spatial Graphing** — Distance-based nearest-neighbor system classifies Young Seals as Pups or Weaners using edge-to-edge Pythagorean distance to the nearest Adult Female bounding box (threshold: 185px) 
5. **Aggregation** — Confidence proportions across all detections are multiplied by predicted seal counts to produce final population estimates per class

## Key Challenges & Solutions 
- **Pup vs. Weaner distinction** — Visual features alone are insufficient since the key differentiator is proximity to an adult female. Solved by deferring this classification to a post-hoc spatial graphing system rather than the image classifier 
- **Class imbalance** — Adult males represented only ~6% of training data. Addressed through targeted geometric and photometric augmentation (13 augmentations per male image), growing the training set to 2,172 images 
- **Centroid distance overestimation** — Initial centroid-to-centroid distance measurements inflated weaner counts. Resolved by switching to edge-to-edge Pythagorean distance 

## Model Performance | 
Class 
| 
Precision 
| Recall 
| F1-Score 
|
|
---
|
---
|
---
|
---
| 
|
 Adult Male 
|

 0.85 
|
 0.70 
|
 0.77 
|
|
 Adult Female 
|
 0.85 
|
 0.91 
|
 0.88 
| 
|
 Young Seal 
|
 0.93 
|
 0.95 
|
 0.94 
|

 **Full pipeline accuracy:** 64.1% overall | 79.3% excluding adult males

## Tech Stack
- Python, PyTorch 
- ResNet-18 (classification), YOLOv_ (object detection) 
- Roboflow (annotation & training) 
- Docker (containerization & deployment) 
- Web UI (beach image upload, bounding box visualization, hover tooltips, history table) - CLI tool (batch processing of image directories -> CSV output)

## Deployment
The full pipeline is containerized with Docker and deployable to a remote server, accessible via web browser over Cal Poly's GlobalProtect VPN. Local installation is also supported by building the Dockerfile directly.

## Team
- Daniel Alvarez (https://github.com/daniel15alv)
- Connor Gamba (https://github.com/cgamba533)
- Tara Rajagopalan (https://github.com/tararajagopalan)

---



# Seal Counter CLI - User Guide

Welcome to the **Seal Counter CLI**! This guide will walk you through the installation and usage of the command-line interface for counting seals in images.

This program is designed to help you count seals in images using a pre-trained model. It is built using Python and packaged for easy installation and usage.

## Prerequisites
Before you start, ensure you have:
- A computer with Windows, macOS, or Linux
- An internet connection
- **Python 3.8 - 3.12**

---

## Step 1: Install Python (Version 3.8 - 3.12)

If you don't have Python installed:

1. Visit the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download a version within **3.8 to 3.12** for your operating system (This program was built using **Python 3.10.4** so we recommend using that)
3. Run the installer and **check the box** that says `Add Python to PATH` before clicking `Install Now`.
4. Verify the installation:
   - Open a terminal (Command Prompt, PowerShell, or Terminal)
   - Type:
     ```sh
     python --version
     ```
   - You should see something like `Python 3.x.x` (where `x.x` is your version).

---

## Step 2: Install ESD CLI Package Using pip
1. Open a terminal (Command Prompt, PowerShell, or Terminal).
2. Install the package using pip:
    ```sh
    pip install https://github.com/brandonhjkim/elephant-seals-CLI/releases/download/1.1.0/elephant_seals_counter-1.1.0-py3-none-any.whl
    ```
3. Ensure you are downloading the latest version of the package from the [releases page](
    https://github.com/brandonhjkim/elephant-seals-CLI/releases).
3. Verify the installation:
    - Type:
      ```sh
      pip show elephant_seals_counter
      ```
    - You should see package information including the version.

---

## Step 3: Run the Seal Counter CLI

1. Once the virtual environment is set up, run the program using:
    ```sh
    seal-counter
    ```
2. Follow the on-screen instructions to provide a API key or accept the default one.
3. When the program runs, it will ask you to **provide a folder path** within the repository. Simply type the path and press `Enter` when prompted. 
There is currently an existing folder in this repository named "beach_images" with 2 beaches already uploaded as examples. 
To have this model run on new images, upload new beach images into that folder, type in "beach_images" when prompted for a folder, and be patient!
You can also upload a new folder of images that you can specify for this program to run with. Please make sure that new images are formatted in one of the following file extensions: `.jpeg` `.png` `.jpg` `.tif` `.tiff`

---

## Troubleshooting
- **Command not found?** Ensure Python is installed and added to PATH.

If you need further help, feel free to open an issue on this repository!

---

## Additional Info
Please do not change the assets folder or the .gitignore file! They are essential to making sure this repository functions and updates properly! 

Enjoy using **Seal Counter CLI**!

## Web App

Make sure to check out [this repository](https://github.com/ishaansathaye/elephant-seals-detection?tab=readme-ov-file) for the web app interface!


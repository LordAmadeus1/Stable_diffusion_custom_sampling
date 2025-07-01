# Stable diffusion custom sampling with classifier-free guidance

This project implements a custom inference pipeline for image generation using Stable Diffusion models.  
The notebook allows fine-grained control of the sampling process and applies **Classifier-Free Guidance (CFG)** to improve prompt adherence during image synthesis.

## Project Overview

A full inference workflow was developed to:
- Load and configure Stable Diffusion components from Hugging Face's `diffusers` library.
- Tokenize prompts and optional negative prompts for image conditioning.
- Initialize random latent variables.
- Apply a custom denoising loop using a UNet model and a configurable scheduler.
- Combine unconditional and conditional predictions via CFG formula.
- Decode the final latent representation into a valid image.
- Optionally save intermediate images at defined intervals during sampling.

## Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- PIL
- NumPy
- TQDM

## Project Structure

Stable_diffusion_custom_sampling/

┣ custom_diffusion_sampling.ipynb

┣ outputs/

┣ README.md

┣ requirements.txt

## How to Run the Project

1. Clone this repository:
   
  git clone https://github.com/LordAmadeus1/Stable_diffusion_custom_sampling.git

  cd Stable_diffusion_custom_sampling

2. Install the required dependencies:
   
  pip install -r requirements.txt

3. Open the notebook:
- Launch `custom_diffusion_sampling.ipynb` in Google Colab or a local Jupyter environment.
- Run each cell sequentially to reproduce the sampling workflow and image generation process.

## Results

![avocado_vg](https://github.com/user-attachments/assets/3ae79fbd-0f59-4c56-9c57-d3a3b42526fa)


The notebook successfully generates images conditioned by text prompts using a custom denoising loop with Classifier-Free Guidance.  
Intermediate images are saved every 10 denoising steps in the `/outputs/` folder, allowing visualization of the progressive latent denoising process.

The implementation can be extended by adding additional samplers, negative prompt configurations and optimization callbacks.

# Doodle-to-Pic Generator (Pix2Pix GAN)

This repository contains a professional implementation of **Pix2Pix (Conditional Generative Adversarial Network)** for paired image-to-image translation. 

## 🚀 Overview
The model is designed to perform two primary tasks:
1. **Doodle-to-Real**: Converting hand-drawn sketches into realistic face photos using the CUHK Face Sketch database.
2. **Sketch Colorization**: Automatically adding vibrant colors to anime sketches using the Anime Sketch Colorization dataset.

## 🏗️ Architecture
- **Generator**: A U-Net based architecture with skip connections between encoder and decoder layers to preserve fine spatial details.
- **Discriminator**: A PatchGAN (70x70) that classifies image patches as real or fake, focusing on local texture and realism.
- **Optimization**: Uses a combination of Advarsarial Loss and L1 Reconstruction Loss (Lambda=100) for sharp, accurate results.

## 🛠️ Features
- **Mixed Precision Training**: Uses `torch.cuda.amp` to optimize performance on dual T4 GPUs.
- **Checkpointing**: Automated save/resume system to prevent progress loss.
- **Evaluation**: Quantitative analysis using **SSIM** (Structural Similarity Index) and **PSNR** (Peak Signal-to-Noise Ratio).
- **Deployment**: Integrated **Gradio** web app for real-time testing.

## 📂 Project Structure
- `model.py`: U-Net and PatchGAN definitions.
- `dataset.py`: Paired data handling for both separate and combined formats.
- `train_utils.py`: Training engine with AMP support.
- `eval_utils.py`: Numerical metrics calculation.
- `Pix2Pix_Doodle_to_Real.ipynb`: Main execution notebook.

## 📈 RESULTS
Training history and generated samples can be found in the provided Jupyter notebook.

## 👤 Author
Developed by **MZafeer10**

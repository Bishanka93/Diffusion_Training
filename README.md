
# Image Generation Using Diffusion Models with Hugging Face Diffusers on the Smithsonian Butterflies Dataset

## 📌 Overview

This experiment focuses on implementing and studying a diffusion-based image generation model using the **Hugging Face Diffusers library** and **PyTorch**. The model is trained on the Smithsonian Butterflies dataset to generate realistic butterfly images by progressively denoising random Gaussian noise.

The experiment explores the complete diffusion model pipeline, including dataset preparation, forward diffusion, reverse diffusion, model training, image generation, and performance evaluation through visualizations and training loss analysis.

---

## 🎯 Aim

To implement and study a diffusion-based image generation model using the Hugging Face Diffusers library on the Smithsonian Butterflies dataset, and evaluate the quality of generated images through qualitative visualization and analysis of the training process.

---

## 📚 Learning Outcomes

After completing this experiment, the following learning outcomes are achieved:

- Understand the fundamental principles of diffusion models.
- Learn the forward diffusion process (gradual addition of noise).
- Understand the reverse diffusion process (iterative noise removal).
- Prepare and preprocess the Smithsonian Butterflies dataset.
- Implement and train a diffusion model using Hugging Face Diffusers and PyTorch.
- Generate butterfly images from random Gaussian noise.
- Visualize generated images and training loss curves.
- Evaluate image quality through qualitative inspection.
- Analyze training convergence and model performance.

---

## 🧠 Theory

Diffusion models are deep generative models that learn to generate realistic images by progressively reversing a noise corruption process.

Unlike Generative Adversarial Networks (GANs), diffusion models do not use adversarial training between two competing neural networks. Instead, they train a neural network to predict and remove noise added to training images.

### 1. Forward Diffusion Process

In the forward diffusion process, Gaussian noise is gradually added to an original image over a sequence of timesteps. After a sufficient number of steps, the original image becomes almost indistinguishable from random noise.

The mathematical representation is:

\[
q(x_t|x_{t-1}) = \mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1},\beta_t I)
\]

Where:
- `x_t` = Noisy image at timestep t.
- `x_(t-1)` = Image at the previous timestep.
- `β_t` = Noise variance schedule.
- `N` = Gaussian distribution.

### 2. Reverse Diffusion Process

The reverse diffusion process learns to gradually remove noise from a noisy image to reconstruct a clean image.

A neural network predicts the noise added at each timestep, allowing the model to progressively generate realistic images.

### 3. Image Generation

After training, the model starts with random Gaussian noise and applies the learned reverse diffusion process to generate new butterfly images.

### 4. Hugging Face Diffusers

The Hugging Face Diffusers library provides tools and components for implementing diffusion models, including:

- Diffusion model architectures.
- Noise schedulers.
- Training utilities.
- Image generation pipelines.
- Model saving and loading functionality.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Hugging Face Diffusers | Diffusion model implementation |
| Hugging Face Datasets | Dataset handling |
| NumPy | Numerical computations |
| Matplotlib | Visualization and plotting |
| PIL (Pillow) | Image processing |
| Google Colab / Jupyter Notebook | Development environment |

---

## 📂 Dataset

**Dataset Name:** Smithsonian Butterflies Dataset

The dataset contains butterfly images that are used to train the diffusion model to learn visual patterns, colors, textures, and shapes of different butterflies.

Dataset preparation involves:

1. Loading the butterfly image dataset.
2. Resizing images to the required dimensions.
3. Converting images into RGB format.
4. Normalizing pixel values.
5. Preparing images for PyTorch model training.

---

## ⚙️ Methodology / Procedure

The experiment follows these implementation steps:

### Step 1: Environment Setup
- Install and import the required Python libraries.
- Configure the PyTorch environment.
- Set up the Hugging Face Diffusers library.

### Step 2: Dataset Preparation
- Load the Smithsonian Butterflies dataset.
- Preprocess and resize the images.
- Normalize the image data.
- Prepare the dataset using a DataLoader.

### Step 3: Hyperparameter Initialization
- Define the batch size.
- Set the learning rate.
- Configure the number of training epochs.
- Initialize the optimizer and other training parameters.

### Step 4: Diffusion Model Definition
- Define the diffusion model architecture.
- Configure the neural network to predict noise.
- Initialize the model parameters.

### Step 5: Noise Scheduler Configuration
- Configure the noise scheduler.
- Define the number of diffusion timesteps.
- Set up the forward and reverse diffusion processes.

### Step 6: Model Training
- Load batches of butterfly images.
- Add Gaussian noise at randomly selected timesteps.
- Train the model to predict the added noise.
- Calculate the training loss.
- Update model parameters using backpropagation.

### Step 7: Epoch-wise Monitoring
- Monitor the training loss after each epoch.
- Track model performance.
- Generate sample images during training to observe progress.

### Step 8: Image Generation
- Initialize random Gaussian noise.
- Apply the reverse diffusion process.
- Iteratively denoise the random noise.
- Generate new butterfly images.

### Step 9: Training Metrics Analysis
- Record training loss values.
- Plot the training loss curve.
- Analyze convergence and training performance.

### Step 10: Result Visualization
- Display generated butterfly images.
- Observe image quality, patterns, colors, and shapes.
- Compare generated images with the training dataset.

### Step 11: Model Saving and Inference
- Save the trained model.
- Load the saved model for inference.
- Generate new butterfly images using the trained model.

---

## 📊 Results and Observations

The diffusion model was trained using the Smithsonian Butterflies dataset to understand the image generation process.

The following results are analyzed:

1. **Training Loss:** The loss curve is used to observe how effectively the model learns to predict the noise added to the images.
2. **Generated Images:** The model generates new butterfly images by progressively denoising random Gaussian noise.
3. **Visual Quality:** Generated samples are examined for butterfly-like structures, colors, textures, and visual patterns.
4. **Training Convergence:** Changes in training loss help evaluate the learning behavior and convergence of the model.

### Expected Visualizations

Include the following outputs from your completed experiment:

- Generated butterfly image samples.
- Training loss versus epochs graph.
- Intermediate image generation samples (if available).

---

## 📦 Installation

Clone the repository:

```bash
git clone <https://github.com/Bishanka93/Diffusion_Training.git>
```

Navigate to the project directory:

```bash
cd Image-Generation-Using-Diffusion-Models
```

Install the required dependencies:

```bash
pip install torch torchvision diffusers datasets transformers accelerate matplotlib numpy pillow
```

Run the Jupyter Notebook:

```bash
jupyter notebook experiment.ipynb
```

---

## 🔍 Conclusion

In this experiment, a diffusion-based image generation model was implemented using the Hugging Face Diffusers library and PyTorch on the Smithsonian Butterflies dataset.

The experiment demonstrated the working principles of forward diffusion and reverse diffusion, where Gaussian noise is progressively added to images and subsequently removed by a trained neural network.

The training process was analyzed using loss curves, while generated butterfly images were evaluated through qualitative visual inspection.

Overall, this experiment provided practical understanding of diffusion-based generative models, dataset preprocessing, model training, noise scheduling, image synthesis, and inference.

---

## 📖 References

1. [Hugging Face Diffusers Documentation – Basic Training](https://huggingface.co/docs/diffusers/en/tutorials/basic_training)
2. [Google Skills – Course Template 541](https://www.skills.google/course_templates/541)
3. [Google AI Edge – MediaPipe Image Generator](https://developers.google.com/edge/mediapipe/solutions/vision/image_generator)
4. Chollet, F. (2021). *Deep Learning with Python*, 2nd Edition. Manning Publications, Shelter Island, NY, USA.

---

## 👨‍💻 Author

**Name:** Bishanka Sarma

**Topic:** Image Generation Using Diffusion Models

**Framework:** Hugging Face Diffusers & PyTorch

---

⭐ If you found this project useful, consider giving the repository a star!

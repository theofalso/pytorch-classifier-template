# PyTorch Generic Image Classifier

A lightweight, dynamic boilerplate for training custom image classification models using PyTorch and Transfer Learning (ResNet18). This template automatically adapts to your dataset and includes a ready-to-use Gradio web interface.

## Repository Structure

\`\`\`text
pytorch-classifier-template/
├── data/
│   ├── train/          # Place training images in subfolders named after their class
│   └── test/           # Place testing/validation images in matching subfolders
├── src/
│   ├── dataset.py      # Dynamic DataLoaders and transformations
│   ├── model.py        # ResNet18 architecture setup
│   └── train.py        # Training loop and JSON class exporter
├── app.py              # Gradio web UI for inference
└── requirements.txt    # Python dependencies
\`\`\`


### 1. Prepare your Dataset
Organize your images into subdirectories inside `data/train` and `data/test`. The names of these subdirectories will automatically become your classification labels.

*Example:*
\`\`\`text
data/train/
├── cats/
└── dogs/
\`\`\`

### 2. Install Dependencies
Ensure you have Python 3.10+ installed, then run:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 3. Train the Model
Execute the training script from the root directory. This will fine-tune a pre-trained ResNet18 model, automatically generate a `classes.json` file mapping your folders, and output a `custom_model.pth` weights file.
\`\`\`bash
python -m src.train
\`\`\`

### 4. Run the Interface
Launch the Gradio web interface to test your newly trained model locally. The application dynamically reads your classes and loads the custom weights.
\`\`\`bash
python app.py
\`\`\`
Access the UI via your browser at `http://127.0.0.1:7860`.

## Technical Details
- **Architecture:** ResNet18 (ImageNet pre-trained weights).
- **Optimization:** Adam Optimizer, Cross-Entropy Loss.
- **Transformations:** Random horizontal flip, rotation, and ImageNet normalization (RGB 224x224).
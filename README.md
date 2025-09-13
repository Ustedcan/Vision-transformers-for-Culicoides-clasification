# Vision Transformers for *Culicoides* Classification

This project uses a Vision Transformer (ViT) architecture for the automatic classification of species from the *Culicoides* genus based on images. *Culicoides* are small insects of medical and veterinary importance, and their accurate identification is crucial for epidemiological studies.

## Description

The main objective is to implement and train a model based on Vision Transformers that can efficiently and accurately classify images of *Culicoides*, thus facilitating research and monitoring of these species.

## Repository Structure

```
Images/                  # Folder containing all images organized for training and testing
models.py                # Vision Transformer architecture definition
test_res.ipynb           # Notebook for model testing/evaluation
training_res.ipynb       # Notebook for model training
utils.py                 # Auxiliary functions
README.md                # This file
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- Jupyter Notebook

You can install the basic requirements with:

```bash
pip install torch torchvision numpy matplotlib scikit-learn jupyter
```

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/Ustedcan/Vision-transformers-for-Culicoides-clasification.git
    cd Vision-transformers-for-Culicoides-clasification
    ```

2. Ensure your images are organized exactly the same way inside the `Images/` folder as in the original repository.

3. **IMPORTANT:** Before running `training_res.ipynb` and `test_res.ipynb`, review and modify, if necessary, the paths to the `Images` folder in the cells where the DataLoader is defined, so they correspond to the actual location on your computer.

4. Run the notebooks to train and evaluate the Vision Transformer model.

## Important Notes

- There are no additional folders in the repository. All relevant code is in the main files at the root level, and images must be in the `Images` folder.
- If you change the name or location of the images folder, remember to adjust the paths in the notebooks to avoid loading errors.

## Contributions

Contributions are welcome! Please open an issue or pull request for suggestions, improvements, or bug reports.

**Author:** [Ustedcan](https://github.com/Ustedcan)

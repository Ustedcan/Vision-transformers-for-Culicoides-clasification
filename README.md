# Vision Transformers for Culicoides Classification

Este repositorio contiene el código y los recursos necesarios para entrenar y evaluar modelos de clasificación de especies de *Culicoides* (mosquitos del género **Culicoides**) a partir de imágenes de alas.  
El proyecto explora arquitecturas basadas en **ResNet-18**, **Vision Transformers (ViT)** y **clasificadores superficiales** (SVM, MLP, XGBoost).

---

## 📂 Estructura del proyecto

```text
Vision-transformers-for-Culicoides-clasification/
│
├── Images/                  # Carpeta con las imágenes de entrada
├── notebooks/
│   └── training.ipynb       # Notebook principal de entrenamiento
├── models/                  # Pesos y checkpoints entrenados
├── utils/                   # Funciones auxiliares (dataloader, métricas, etc.)
└── README.md

# 🧠 Entrenamiento YOLOv8 con Roboflow y Ultralytics  
### 🇪🇸 Guía paso a paso / 🇺🇸 Step-by-step Guide

---

## 🇪🇸 **Versión en Español**

---

### 🧩 1️⃣ Instalación de librerías

Antes de ejecutar el script, asegúrate de tener instalado **Python 3.8 o superior** y luego instala las dependencias necesarias:


``pip install roboflow ultralytics``

Si deseas usar una GPU NVIDIA o DirectML, instala PyTorch con soporte CUDA o DirectML:


# Para GPU NVIDIA
``pip install torch torchvision torchaudio`` --index-url https://download.pytorch.org/whl/cu118

# Para GPU AMD o Intel (Windows)
``pip install torch-directml``

---

### ⚙️ 2️⃣ Configuración del archivo de Roboflow
🧩 Aquí debes agregar tu explicación personalizada sobre cómo:

Obtener tu API Key en Roboflow

Crear un proyecto

Generar una versión del dataset

Exportar el dataset en formato YOLOv8

---

### 💻 3️⃣ Código de entrenamiento
El siguiente script descarga un dataset directamente desde Roboflow y entrena un modelo YOLOv8 utilizando la librería Ultralytics:

```
# 🚀 Entrenamiento de un modelo YOLOv8 (detección automática de GPU o CPU)

import torch
from ultralytics import YOLO

# 🔍 Verificar dispositivo disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Dispositivo en uso: {device}")

# 🧠 Cargar el modelo base (puedes cambiar a yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")

# ⚙️ Entrenar el modelo
results = model.train(
    data="ruta/a/tu/archivo.yaml",  # Ruta al archivo de configuración de datos
    epochs=100,                     # Número de épocas
    imgsz=640,                      # Tamaño de las imágenes
    batch=16,                       # Tamaño del lote
    name="mi_modelo_yolov8",        # Nombre del experimento
    project="runs/train",           # Carpeta donde se guardan los resultados
    device=device                   # Asignar CPU o GPU automáticamente
)

# 📊 Evaluar el modelo (opcional)
metrics = model.val()

# 💾 Exportar el modelo entrenado (opcional)
model.export(format="onnx")  # También puedes usar "torchscript", "engine", etc.

```

---

### ⚙️ Parámetros principales del entrenamiento

| Parámetro | Descripción                                                                                  | Ejemplo               |
| --------- | -------------------------------------------------------------------------------------------- | --------------------- |
| `data`    | Ruta al archivo `.yaml` que contiene las rutas a las imágenes de entrenamiento y validación. | `"dataset/data.yaml"` |
| `epochs`  | Número de iteraciones completas sobre el conjunto de datos.                                  | `100`                 |
| `imgsz`   | Tamaño al que se redimensionan las imágenes.                                                 | `640`                 |
| `batch`   | Número de imágenes procesadas por paso durante el entrenamiento.                             | `16`                  |
| `name`    | Nombre del experimento o modelo.                                                             | `"mi_modelo_yolov8"`  |
| `project` | Carpeta donde se guardarán los resultados del entrenamiento.                                 | `"runs/train"`        |
| `device`  | Define si se usa CPU o GPU automáticamente.                                                  | `"cuda"` o `"cpu"`    |


---

### 📊 4️⃣ Resultados esperados

Una vez ejecutes el script, deberías ver en consola algo como esto:
```bash
📦 Descargando dataset desde Roboflow...
✅ Descarga completa.
📄 Usando data.yaml: roboflow_project/NOMBRE_PROYECTO/data.yaml
🚀 Iniciando entrenamiento...

train: Scanning images and labels... 
train: New cache created: roboflow_project/NOMBRE_PROYECTO/cache...
Epoch 1/100
...
Epoch 100/100

🏁 Training complete (100 epochs completed)
```

📁 Estructura de carpetas esperada:

```
roboflow_project/
└── NOMBRE_DE_TU_PROYECTO/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```
```
runs/
└── roboflow_yolov8/
    └── NOMBRE_PROYECTO_v1_yolov8x/
        ├── weights/
        │   ├── last.pt
        │   └── best.pt
        ├── results.png
        └── opt.yaml
```
---
### ✅ El modelo entrenado se encuentra en:

runs/roboflow_yolov8/NOMBRE_PROYECTO_v1_yolov8x/weights/best.pt

Puedes probarlo con:

from ultralytics import YOLO

model = YOLO("ruta/a/best.pt")
results = model("imagen.jpg")
results.show()

---
## 🇺🇸 English Version
---

### 🧩 1️⃣ Library Installation
Before running the script, make sure you have Python 3.8+ installed and run:

``pip install roboflow ultralytics``

If you plan to use GPU (NVIDIA or DirectML), install PyTorch with CUDA or DirectML support:

#### For NVIDIA GPUs
``pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118``

#### For AMD / Intel GPUs (Windows)
``pip install torch-directml``

---

### ⚙️ 2️⃣ Roboflow Configuration
🧩 Add your custom explanation here about how to:

Get your API Key from Roboflow

Create a project

Generate a dataset version

Export the dataset in YOLOv8 format

---

### 💻 3️⃣ Training Code
This script automatically downloads your Roboflow dataset and trains a YOLOv8 model using the Ultralytics library.

```
# 🚀 YOLOv8 Model Training (Automatic GPU/CPU Detection)

import torch
from ultralytics import YOLO

# 🔍 Check available device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# 🧠 Load the base model (you can change to yolov8s.pt, yolov8m.pt, etc.)
model = YOLO("yolov8n.pt")

# ⚙️ Train the model
results = model.train(
    data="path/to/your/data.yaml",  # Path to your dataset configuration file
    epochs=100,                     # Number of training epochs
    imgsz=640,                      # Image size (resize before training)
    batch=16,                       # Batch size per iteration
    name="my_yolov8_model",         # Experiment/model name
    project="runs/train",           # Folder where results will be saved
    device=device                   # Automatically assign CPU or GPU
)

# 📊 Evaluate the trained model (optional)
metrics = model.val()

# 💾 Export the final model (optional)
model.export(format="onnx")  # You can also use "torchscript", "engine", etc.

```

⚙️ Main Training Parameters
| Parameter | Description                                                             | Example               |
| --------- | ----------------------------------------------------------------------- | --------------------- |
| `data`    | Path to the `.yaml` file containing training and validation data paths. | `"dataset/data.yaml"` |
| `epochs`  | Number of complete passes through the dataset.                          | `100`                 |
| `imgsz`   | Image size for training (input resolution).                             | `640`                 |
| `batch`   | Number of images processed per iteration.                               | `16`                  |
| `name`    | Name for your model or experiment folder.                               | `"my_yolov8_model"`   |
| `project` | Directory where training results will be saved.                         | `"runs/train"`        |
| `device`  | Automatically selects CPU or GPU.                                       | `"cuda"` or `"cpu"`   |

---

### 📊 4️⃣ Expected Results
When executed, your console should display logs similar to this:

``` bash
📦 Downloading dataset from Roboflow...
✅ Download complete.
📄 Using data.yaml: roboflow_project/PROJECT_NAME/data.yaml
🚀 Starting training...
train: Scanning images and labels...
Epoch 1/100
...
Epoch 100/100
🏁 Training complete (100 epochs completed)
```

📁 Expected folder structure:

```
roboflow_project/
└── PROJECT_NAME/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/
```
```
runs/
└── roboflow_yolov8/
    └── PROJECT_NAME_v1_yolov8x/
        ├── weights/
        │   ├── last.pt
        │   └── best.pt
        ├── results.png
        └── opt.yaml
```

✅ The trained model will be located at:


runs/roboflow_yolov8/PROJECT_NAME_v1_yolov8x/weights/best.pt

Test your trained model with:
```
from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model("image.jpg")
results.show()
```
---

## About This Project

📌 Author: Juan

📅 Project: Robocup 2026 — YOLOv8 Training with Roboflow

🚀 Libraries Used: Ultralytics · Roboflow

💡 Goal: Simplify object detection training workflows for robotic vision systems.


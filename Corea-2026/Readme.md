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
#### ================================================
#### 🧠 train_from_roboflow_yolov8.py
#### ================================================
import os
from pathlib import Path
from roboflow import Roboflow
from ultralytics import YOLO

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY") or "TU_API_KEY_AQUI"
WORK_DIR = Path("roboflow_project")
PROJECT_NAME = "NOMBRE_DE_TU_PROYECTO"   # el slug del proyecto en Roboflow
VERSION = 1  # versión del dataset que quieres descargar
MODEL_BACKBONE = "yolov8x.pt"  # 'yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt'
IMG_SIZE = 640
EPOCHS = 100
BATCH = 16
DEVICE = 0  # GPU (0 o "cuda:0") o CPU ("cpu")
ACCUM = 2   # acumulación de gradientes
WORKERS = 8
LR = 0.01
WEIGHT_DECAY = 0.0005
def download_roboflow_dataset(api_key, project_name, version, out_dir):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    v = project.version(version)
    print("📦 Descargando dataset desde Roboflow...")
    v.download("yolov8", out_dir)
    print("✅ Descarga completa.")

def find_data_yaml(base_dir):
    for p in Path(base_dir).rglob("data.yaml"):
        return str(p)
    raise FileNotFoundError("❌ data.yaml no encontrado en el dataset descargado")

def main():
    out_dir = WORK_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    download_roboflow_dataset(ROBOFLOW_API_KEY, PROJECT_NAME, VERSION, str(out_dir))
    data_yaml = find_data_yaml(out_dir)
    print("📄 Usando data.yaml:", data_yaml)

    model = YOLO(MODEL_BACKBONE)
    print("🚀 Iniciando entrenamiento...")

    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        accumulate=ACCUM,
        lr0=LR,
        weight_decay=WEIGHT_DECAY,
        project="runs/roboflow_yolov8",
        name=f"{PROJECT_NAME}_v{VERSION}_{MODEL_BACKBONE.split('.')[0]}",
        exist_ok=True
    )

    print("🏁 Entrenamiento finalizado. Revisa runs/roboflow_yolov8/")
    
if __name__ == "__main__":
    main()
```

---

### ⚙️ Parámetros principales del entrenamiento

| Parametro         | Descripción                                                                         |
| ------------------ | -----------------------------------------------------------------------------------|
| **epochs**         | Número de ciclos de entrenamiento. A mayor valor, más precisión (pero más tiempo). |
| **imgsz**          | Tamaño de las imágenes de entrada. 640 es el estándar para YOLOv8.                 |
| **batch**          | Número de imágenes procesadas por iteración. Ajusta según la memoria de tu GPU.    |
| **device**         | Define si usar CPU o GPU.                                                          |
| **accumulate**     | Permite simular batches más grandes acumulando gradientes.                         |
| **lr0**            | Tasa de aprendizaje inicial. Controla la velocidad de convergencia.                |
| **weight_decay**   | Regularización para evitar sobreajuste.                                            |
| **project / name** | name	Define la carpeta donde se guardarán los resultados.                          |

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

(Same code as above)

⚙️ Main Training Parameters
| Parámetro          | Descripción                                                                        |
| ------------------ | ---------------------------------------------------------------------------------- |
| **epochs**         | Número de ciclos de entrenamiento. A mayor valor, más precisión (pero más tiempo). |
| **imgsz**          | Tamaño de las imágenes de entrada. 640 es el estándar para YOLOv8.                 |
| **batch**          | Número de imágenes procesadas por iteración. Ajusta según la memoria de tu GPU.    |
| **device**         | Define si usar CPU o GPU.                                                          |
| **accumulate**     | Permite simular batches más grandes acumulando gradientes.                         |
| **lr0**            | Tasa de aprendizaje inicial. Controla la velocidad de convergencia.                |
| **weight_decay**   | Regularización para evitar sobreajuste.                                            |
| **project / name** | Define la carpeta donde se guardarán los resultados.                               |


---

### 📊 4️⃣ Expected Results
When executed, your console should display logs similar to this:

📦 Downloading dataset from Roboflow...

✅ Download complete.

📄 Using data.yaml: roboflow_project/PROJECT_NAME/data.yaml

🚀 Starting training...

train: Scanning images and labels...

Epoch 1/100

...

Epoch 100/100

🏁 Training complete (100 epochs completed)

📁 Expected folder structure:

roboflow_project/

└── PROJECT_NAME/

    ├── data.yaml
    
    ├── train/
    
    ├── valid/
    
    └── test/

runs/

└── roboflow_yolov8/

    └── PROJECT_NAME_v1_yolov8x/
    
        ├── weights/
        
        │   ├── last.pt
        
        │   └── best.pt
        
        ├── results.png
        
        └── opt.yaml

✅ The trained model will be located at:


runs/roboflow_yolov8/PROJECT_NAME_v1_yolov8x/weights/best.pt

Test your trained model with:

from ultralytics import YOLO

model = YOLO("path/to/best.pt")
results = model("image.jpg")
results.show()

---

## About This Project

📌 Author: Juan

📅 Project: Robocup 2026 — YOLOv8 Training with Roboflow

🚀 Libraries Used: Ultralytics · Roboflow

💡 Goal: Simplify object detection training workflows for robotic vision systems.


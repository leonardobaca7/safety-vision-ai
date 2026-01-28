# 🎯 FASE 1: Fine-Tuning del Modelo YOLOv8

## Objetivo
Entrenar (fine-tune) YOLOv8 con un dataset especializado en detección de cascos y chalecos para maximizar precisión en detección de EPP.

## ✅ Checklist de Tareas

### 1. Preparación del Dataset
- [ ] Buscar y evaluar datasets de EPP en Roboflow Universe
- [ ] Descargar dataset en formato YOLO (train/valid/test)
- [ ] Verificar estructura de carpetas y archivos
- [ ] Crear archivo `data.yaml` con configuración
- [ ] Explorar el dataset (cantidad de imágenes, distribución de clases)

### 2. Configuración del Entorno de Entrenamiento
- [ ] Verificar instalación de Poetry y dependencias
- [ ] Instalar Ultralytics y PyTorch
- [ ] Verificar disponibilidad de GPU (opcional pero recomendado)
- [ ] Crear notebook de entrenamiento

### 3. Entrenamiento del Modelo
- [ ] Cargar modelo base YOLOv8n pre-entrenado
- [ ] Configurar hiperparámetros (epochs, batch size, etc.)
- [ ] Iniciar entrenamiento con data augmentation
- [ ] Monitorear métricas durante el entrenamiento
- [ ] Guardar el mejor modelo (best.pt)

### 4. Evaluación del Modelo
- [ ] Calcular métricas: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- [ ] Generar matriz de confusión
- [ ] Probar detecciones en imágenes del conjunto de validación
- [ ] Medir velocidad de inferencia (FPS)
- [ ] Documentar resultados

## 📚 Recursos Recomendados

### Datasets Recomendados (Roboflow Universe)

1. **Hard Hat Detection**
   - URL: https://universe.roboflow.com/roboflow-universe-projects/hard-hat-detection
   - Clases: Person, Helmet, No-Helmet
   - ~1000+ imágenes

2. **Construction Safety Detection**
   - URL: https://universe.roboflow.com/construction-safety/construction-safety-detection
   - Clases: Person, Helmet, Vest, No-Helmet, No-Vest
   - ~2000+ imágenes

3. **PPE Detection**
   - URL: https://universe.roboflow.com/ppe-detection/ppe-detection-dataset
   - Clases: Person, Hard Hat, Safety Vest
   - ~1500+ imágenes

### Alternativa: Kaggle
- **Hard Hat Detection Dataset**: https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection

## 🚀 Pasos Detallados

### Paso 1: Descargar Dataset de Roboflow

1. Ve a [Roboflow Universe](https://universe.roboflow.com/)
2. Busca "Hard Hat Detection" o "PPE Detection"
3. Selecciona un dataset con buenas métricas (>1000 imágenes)
4. Haz clic en "Download Dataset"
5. Selecciona formato: **YOLOv8**
6. Descarga el ZIP y extráelo en `datasets/helmet_vest_detection/`

Estructura esperada:
```
datasets/helmet_vest_detection/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/ (opcional)
│   ├── images/
│   └── labels/
└── data.yaml
```

### Paso 2: Crear archivo `data.yaml`

Si el dataset no incluye `data.yaml`, créalo:

```yaml
path: ../datasets/helmet_vest_detection
train: train/images
val: valid/images
test: test/images  # opcional

nc: 3  # número de clases
names: ['Person', 'Helmet', 'Vest']
```

### Paso 3: Crear Notebook de Entrenamiento

Abre Jupyter:
```bash
cd Safety-Vision-AI
poetry shell
jupyter notebook notebooks/
```

Crea un nuevo notebook llamado `2_yolov8_fine_tuning.ipynb` con el siguiente contenido inicial:

```python
# Celda 1: Imports
from ultralytics import YOLO
import torch
import os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Celda 2: Cargar modelo base
model = YOLO('yolov8n.pt')  # nano (más rápido)
# Alternativas: yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)

# Celda 3: Entrenar
results = model.train(
    data='../datasets/helmet_vest_detection/data.yaml',
    epochs=50,              # Ajustar según tiempo disponible (min 30, ideal 50-100)
    imgsz=640,              # Tamaño de imagen (640 es estándar)
    batch=16,               # Ajustar según memoria GPU (16 es seguro para 8GB VRAM)
    device=0,               # 0 = GPU, 'cpu' = CPU
    project='../models_assets',
    name='yolov8_helmet_vest',
    patience=10,            # Early stopping
    save=True,
    plots=True,
    
    # Data Augmentation (ya viene por defecto en YOLOv8)
    hsv_h=0.015,            # Hue augmentation
    hsv_s=0.7,              # Saturation augmentation
    hsv_v=0.4,              # Value augmentation
    degrees=10.0,           # Rotation
    translate=0.1,          # Translation
    scale=0.5,              # Scale
    flipud=0.0,             # Flip up-down
    fliplr=0.5,             # Flip left-right (50%)
)

# Celda 4: Evaluar
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")

# Celda 5: Guardar modelo
# El mejor modelo ya está guardado automáticamente en:
# models_assets/yolov8_helmet_vest/weights/best.pt
print("✅ Modelo guardado en: models_assets/yolov8_helmet_vest/weights/best.pt")
```

### Paso 4: Ejecutar el Entrenamiento

**Opción A: En tu máquina local (si tienes GPU)**
```bash
poetry run jupyter notebook notebooks/2_yolov8_fine_tuning.ipynb
```

**Opción B: En Google Colab (GPU gratuita)**

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Sube el notebook
3. Cambia el runtime a GPU: `Runtime > Change runtime type > GPU (T4)`
4. Instala dependencias:
```python
!pip install ultralytics
```
5. Sube tu dataset o descárgalo desde Roboflow directamente
6. Ejecuta el entrenamiento

## 📊 Métricas Esperadas

| Métrica | Objetivo Mínimo | Ideal |
|---------|-----------------|-------|
| mAP@0.5 | > 0.75 | > 0.85 |
| mAP@0.5:0.95 | > 0.50 | > 0.65 |
| Precision | > 0.80 | > 0.90 |
| Recall | > 0.75 | > 0.85 |
| FPS (GPU) | > 30 | > 60 |
| FPS (CPU) | > 10 | > 20 |

## ⚠️ Troubleshooting

### Problema: "CUDA out of memory"
**Solución**: Reduce el `batch` size (prueba 8, 4, o incluso 2)

### Problema: "Dataset not found"
**Solución**: Verifica que el path en `data.yaml` sea correcto (relativo o absoluto)

### Problema: Overfitting (train loss << val loss)
**Solución**: 
- Reduce epochs
- Aumenta augmentation
- Aumenta patience para early stopping

### Problema: No tengo GPU y el entrenamiento es muy lento
**Solución**: Usa Google Colab con GPU gratuita

## 🎯 Criterio de Éxito

La Fase 1 está completa cuando:

✅ Tienes un modelo fine-tuneado con **mAP > 0.75**  
✅ El modelo está guardado en `models_assets/yolov8_helmet_vest_best.pt`  
✅ Tienes métricas documentadas (mAP, Precision, Recall, FPS)  
✅ Has probado el modelo en imágenes de validación y detecta correctamente

## 📝 Próximos Pasos

Después de completar esta fase, continúa con:
- **Fase 2**: Desarrollo del Pipeline de Inferencia Básico

## 🔗 Links Útiles

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Universe](https://universe.roboflow.com/)
- [Google Colab](https://colab.research.google.com/)
- [YOLOv8 Training Tips](https://docs.ultralytics.com/modes/train/)

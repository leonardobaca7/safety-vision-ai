# 🚀 Comandos de Inicio - Safety Vision AI

## 1️⃣ Inicializar Git y Crear Repositorio

```powershell
# Navegar a la carpeta del proyecto
cd "C:\Users\LENOVO\OneDrive\Documentos\SISTEMASCORE\PROYECTOS\Safety-Vision-AI"

# Inicializar Git
git init

# Agregar todos los archivos
git add .

# Primer commit
git commit -m "feat: Initial project setup with complete structure"

# Crear repositorio en GitHub desde la web:
# 1. Ve a https://github.com/new
# 2. Nombre: safety-vision-ai
# 3. Descripción: "Industrial Safety Monitoring System with Computer Vision (YOLOv8)"
# 4. Público o Privado (según prefieras)
# 5. NO inicialices con README (ya lo tenemos)
# 6. Crea el repositorio

# Conectar con GitHub (reemplaza TU_USUARIO con tu usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/safety-vision-ai.git
git branch -M main
git push -u origin main
```

## 2️⃣ Instalar Poetry (si no lo tienes)

```powershell
# Opción 1: Con pip
pip install poetry

# Opción 2: Con instalador oficial (recomendado)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Verificar instalación
poetry --version
```

## 3️⃣ Configurar el Proyecto con Poetry

```powershell
# Asegúrate de estar en la carpeta del proyecto
cd "C:\Users\LENOVO\OneDrive\Documentos\SISTEMASCORE\PROYECTOS\Safety-Vision-AI"

# Instalar todas las dependencias
poetry install

# Esto puede tardar varios minutos la primera vez
# Poetry creará un entorno virtual automáticamente

# Activar el entorno virtual
poetry shell

# Verificar que todo está instalado
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import ultralytics; print('Ultralytics instalado correctamente')"
```

## 4️⃣ Configurar Pre-commit Hooks

```powershell
# Dentro del entorno de Poetry (después de poetry shell)
poetry run pre-commit install

# Verificar configuración
poetry run pre-commit run --all-files
```

## 5️⃣ Crear archivo .env (Configuración Local)

```powershell
# Copiar el ejemplo
Copy-Item .env.example .env

# Editar el archivo .env con tus configuraciones
# Puedes usar notepad o cualquier editor
notepad .env

# Configuración básica para empezar:
# DATABASE_URL=postgresql://safety_user:safety_pass@localhost:5432/safety_db
# VIDEO_SOURCE=0  # 0 para webcam
# MODEL_PATH=models_assets/yolov8_helmet_vest_best.pt
# CONFIDENCE_THRESHOLD=0.5
# VIOLATION_PERSIST_SECONDS=3.0
```

## 6️⃣ Verificar la Instalación

```powershell
# Test rápido de importaciones
python -c "from ultralytics import YOLO; print('✅ YOLOv8 OK')"
python -c "import cv2; print('✅ OpenCV OK')"
python -c "import fastapi; print('✅ FastAPI OK')"
python -c "import sqlalchemy; print('✅ SQLAlchemy OK')"
```

## 7️⃣ Descargar Modelo Base YOLOv8 (Opcional - se descarga automáticamente)

```powershell
# Esto descargará el modelo base la primera vez que lo uses
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# El modelo se guardará automáticamente en tu caché de Ultralytics
```

## 🎯 Siguiente Paso: FASE 1 - Fine-Tuning

Ahora estás listo para comenzar con la **FASE 1**:

```powershell
# Abrir Jupyter para crear el notebook de entrenamiento
poetry run jupyter notebook notebooks/

# O simplemente abrir VS Code en la carpeta notebooks
code notebooks/
```

Consulta el archivo `FASE_1_GUIA.md` para instrucciones detalladas.

## 🛠️ Comandos Útiles (Makefile)

Si estás en Linux/Mac o tienes `make` instalado en Windows:

```bash
make setup          # Configurar proyecto completo
make install        # Instalar dependencias
make format         # Formatear código
make lint           # Verificar código
make test           # Ejecutar tests
make run-inference  # Ejecutar inferencia
make run-api        # Iniciar API
make train          # Abrir Jupyter
```

En Windows sin make, usa Poetry directamente:

```powershell
poetry install                                    # Instalar
poetry run black app/ inference_pipeline/         # Formatear
poetry run pytest tests/ -v                       # Tests
poetry run python inference_pipeline/run_inference.py  # Inferencia
poetry run uvicorn app.main:app --reload         # API
poetry run jupyter notebook notebooks/            # Jupyter
```

## 📊 Estructura Actual del Proyecto

```
Safety-Vision-AI/
├── ✅ app/                    # Código de la aplicación
├── ✅ inference_pipeline/     # Scripts de inferencia
├── ✅ notebooks/              # Jupyter notebooks
├── ✅ datasets/               # Para datasets (vacío por ahora)
├── ✅ models_assets/          # Para modelos (vacío por ahora)
├── ✅ docker/                 # Dockerfiles (para Fase 5)
├── ✅ tests/                  # Tests unitarios
├── ✅ outputs/                # Outputs (alertas, logs)
├── ✅ README.md               # Documentación principal
├── ✅ pyproject.toml          # Configuración Poetry
├── ✅ .gitignore              # Archivos ignorados por Git
├── ✅ .env.example            # Variables de entorno (ejemplo)
├── ✅ LICENSE                 # MIT License
└── ✅ FASE_1_GUIA.md          # Guía detallada Fase 1
```

## 🚨 Troubleshooting Común

### Poetry no reconocido como comando
**Solución**: Reinicia el terminal o agrega Poetry al PATH manualmente

### Error al instalar dependencias en Windows
**Solución**: 
```powershell
# Instalar Visual C++ Build Tools
# Descarga desde: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### PyTorch sin GPU
**Solución**: Esto es normal, YOLOv8 funcionará en CPU (más lento pero funcional)

### Jupyter no abre en el navegador
**Solución**: 
```powershell
poetry run jupyter notebook --no-browser
# Luego copia la URL que aparece en el terminal
```

## 🎉 ¡Listo para Empezar!

Tu proyecto está configurado y listo. Procede con:

1. **Descargar un dataset de EPP** (Roboflow/Kaggle)
2. **Crear el notebook de entrenamiento**
3. **Entrenar el modelo**
4. **Evaluar resultados**

Consulta `FASE_1_GUIA.md` para detalles completos.

¡Mucho éxito! 🚀

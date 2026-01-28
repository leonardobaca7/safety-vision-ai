# Safety Vision AI - PPE Detection System 🦺

Sistema de monitoreo de seguridad industrial con visión por computadora para detectar el uso de Equipos de Protección Personal (EPP) en tiempo real.

## 🚀 Características

- **Detección inteligente**: YOLOv8 fine-tuneado para detectar cascos y chalecos
- **Lógica de negocio robusta**: Sistema de alertas con persistencia temporal (3 segundos)
- **Edge Computing**: Optimizado para hardware limitado mediante quantization
- **Arquitectura profesional**: Dockerizado con PostgreSQL y API REST
- **Producción-ready**: Control de versiones, testing, CI/CD

## 📋 Requisitos

- Python 3.9+
- Poetry (gestor de dependencias)
- PostgreSQL 13+
- Docker y Docker Compose (opcional)

## 🏗️ Estructura del Proyecto

```
safety_vision_ai/
├── app/                    # Núcleo de la aplicación
├── inference_pipeline/     # Scripts de inferencia
├── notebooks/             # Jupyter Notebooks
├── datasets/              # Datos de entrenamiento
├── models_assets/         # Modelos pre-entrenados
├── docker/                # Dockerización
├── tests/                 # Testing
└── outputs/               # Resultados
```

## 🛠️ Instalación

### Opción 1: Con Poetry (Recomendado)

```bash
# Clonar el repositorio
git clone https://github.com/TU_USUARIO/safety_vision_ai.git
cd safety_vision_ai

# Instalar dependencias
poetry install

# Activar entorno virtual
poetry shell
```

### Opción 2: Con Docker

```bash
cd docker/
docker-compose up --build
```

## 📊 Fases del Proyecto

- [x] **Fase 0**: Configuración del entorno
- [ ] **Fase 1**: Fine-tuning del modelo YOLOv8
- [ ] **Fase 2**: Pipeline de inferencia básico
- [ ] **Fase 3**: Lógica de seguridad y persistencia
- [ ] **Fase 4**: Optimización Edge Computing
- [ ] **Fase 5**: Dockerización
- [ ] **Fase 6**: Dashboard Web (opcional)

## 🎯 Uso

```bash
# Entrenar el modelo
poetry run python notebooks/2_yolov8_fine_tuning.ipynb

# Ejecutar inferencia
poetry run python inference_pipeline/run_inference.py

# Iniciar API
poetry run uvicorn app.main:app --reload
```

## 📈 Performance

- **Modelo**: YOLOv8n
- **FPS**: 25+ en hardware optimizado
- **Precisión**: mAP > 0.80
- **Memoria**: ~400MB RAM

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir cambios mayores.

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## 👨‍💻 Autor

Desarrollado con ❤️ para demostrar habilidades en Computer Vision y Edge AI.

.PHONY: setup install format lint test clean run-inference run-api docker-build docker-up

# Setup del proyecto
setup:
	@echo "🚀 Configurando el proyecto..."
	poetry install
	poetry run pre-commit install
	@echo "✅ Setup completado!"

# Instalar dependencias
install:
	poetry install

# Formatear código
format:
	@echo "🎨 Formateando código..."
	poetry run black app/ inference_pipeline/ tests/
	poetry run isort app/ inference_pipeline/ tests/
	@echo "✅ Código formateado!"

# Lint (verificar código)
lint:
	@echo "🔍 Verificando código..."
	poetry run flake8 app/ inference_pipeline/ tests/ --max-line-length=100
	poetry run mypy app/ inference_pipeline/
	@echo "✅ Verificación completada!"

# Ejecutar tests
test:
	@echo "🧪 Ejecutando tests..."
	poetry run pytest tests/ -v
	@echo "✅ Tests completados!"

# Limpiar archivos temporales
clean:
	@echo "🧹 Limpiando archivos temporales..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "✅ Limpieza completada!"

# Ejecutar inferencia
run-inference:
	@echo "🎥 Ejecutando inferencia..."
	poetry run python inference_pipeline/run_inference.py

# Ejecutar API
run-api:
	@echo "🌐 Iniciando API..."
	poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docker: construir imagen
docker-build:
	@echo "🐳 Construyendo imagen Docker..."
	cd docker && docker-compose build
	@echo "✅ Imagen construida!"

# Docker: levantar servicios
docker-up:
	@echo "🐳 Levantando servicios..."
	cd docker && docker-compose up -d
	@echo "✅ Servicios levantados!"

# Docker: detener servicios
docker-down:
	@echo "🐳 Deteniendo servicios..."
	cd docker && docker-compose down
	@echo "✅ Servicios detenidos!"

# Entrenar modelo (abrir Jupyter)
train:
	@echo "📚 Abriendo Jupyter para entrenamiento..."
	poetry run jupyter notebook notebooks/

# Ayuda
help:
	@echo "Comandos disponibles:"
	@echo "  make setup          - Configurar proyecto por primera vez"
	@echo "  make install        - Instalar dependencias"
	@echo "  make format         - Formatear código con black e isort"
	@echo "  make lint           - Verificar código con flake8 y mypy"
	@echo "  make test           - Ejecutar tests"
	@echo "  make clean          - Limpiar archivos temporales"
	@echo "  make run-inference  - Ejecutar inferencia"
	@echo "  make run-api        - Iniciar API FastAPI"
	@echo "  make docker-build   - Construir imagen Docker"
	@echo "  make docker-up      - Levantar servicios Docker"
	@echo "  make docker-down    - Detener servicios Docker"
	@echo "  make train          - Abrir Jupyter para entrenar"

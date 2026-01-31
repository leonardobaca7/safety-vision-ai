"""
Script de entrenamiento para YOLOv8 - Safety Vision AI
Fase 1: Fine-tuning para detección de EPP (cascos)
"""

import os
os.chdir(r'C:\Users\LENOVO\OneDrive\Documentos\SISTEMASCORE\PROYECTOS\Safety-Vision-AI')

from ultralytics import YOLO
import torch
from pathlib import Path
import shutil
from datetime import datetime

def print_banner(text):
    """Imprimir banner bonito"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def main():
    print_banner("🚀 SAFETY VISION AI - ENTRENAMIENTO YOLOv8")
    
    # 1. Verificar PyTorch y CUDA
    print("📋 VERIFICANDO ENTORNO:")
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        device = 0
    else:
        print("   ⚠️  Running on CPU (será más lento)")
        device = 'cpu'
    
    # 2. Verificar dataset
    print("\n📊 VERIFICANDO DATASET:")
    dataset_path = Path("datasets/helmet_vest_detection")
    data_yaml = dataset_path / "data.yaml"
    
    if not data_yaml.exists():
        print(f"❌ ERROR: No se encontró {data_yaml}")
        return
    
    # Contar imágenes
    train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
    test_images = list((dataset_path / "test" / "images").glob("*.jpg"))
    
    print(f"   🔹 Imágenes de entrenamiento: {len(train_images):,}")
    print(f"   🔹 Imágenes de validación: {len(test_images):,}")
    print(f"   🔹 Total: {len(train_images) + len(test_images):,}")
    
    if len(train_images) == 0:
        print("❌ ERROR: No se encontraron imágenes de entrenamiento")
        return
    
    print(f"   ✅ Dataset listo!")
    
    # 3. Cargar modelo base o checkpoint
    print("\n🤖 CARGANDO MODELO:")
    
    # Verificar si existe un checkpoint previo
    checkpoint_path = Path("models_assets/yolov8_helmet_detection/weights/last.pt")
    
    if checkpoint_path.exists():
        print(f"   🔄 Encontrado checkpoint previo: {checkpoint_path}")
        print("   📥 Cargando desde checkpoint para continuar entrenamiento...")
        model = YOLO(str(checkpoint_path))
        print(f"   ✅ Checkpoint cargado - Se continuará el entrenamiento")
    else:
        print("   📦 No hay checkpoint previo, cargando modelo base...")
        model = YOLO('yolov8n.pt')
        print(f"   ✅ YOLOv8n base cargado correctamente")
    
    # 4. Configuración de entrenamiento
    print("\n⚙️  CONFIGURACIÓN DE ENTRENAMIENTO:")
    
    # Ajustar batch size según disponibilidad de GPU
    if device == 0:
        batch_size = 16  # Con GPU
        print("   🔥 Modo GPU - Batch size: 16")
    else:
        batch_size = 4   # Con CPU
        print("   ⚠️  Modo CPU - Batch size: 4 (será lento)")
    
    config = {
        'data': str(data_yaml),
        'epochs': 50,
        'imgsz': 640,
        'batch': batch_size,
        'device': device,
        'project': 'models_assets',
        'name': 'yolov8_helmet_detection',
        'patience': 10,
        'save': True,
        'plots': True,
        'workers': 8,
        'cache': False,
        'verbose': True,
        
        # Data augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,
        'translate': 0.1,
        'scale': 0.5,
        'flipud': 0.0,
        'fliplr': 0.5,
    }
    
    print(f"   📝 Epochs: {config['epochs']}")
    print(f"   📝 Image size: {config['imgsz']}")
    print(f"   📝 Batch size: {config['batch']}")
    print(f"   📝 Device: {config['device']}")
    
    # 5. Entrenar
    print_banner("🔥 INICIANDO ENTRENAMIENTO")
    
    start_time = datetime.now()
    print(f"⏰ Inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        results = model.train(**config)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print_banner("🎉 ENTRENAMIENTO COMPLETADO")
        print(f"⏰ Duración: {duration}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  ENTRENAMIENTO INTERRUMPIDO POR EL USUARIO")
        return
    except Exception as e:
        print(f"\n\n❌ ERROR DURANTE EL ENTRENAMIENTO: {e}")
        return
    
    # 6. Evaluar modelo
    print("\n📊 EVALUANDO MODELO...")
    try:
        metrics = model.val()
        
        print_banner("📈 MÉTRICAS FINALES")
        print(f"   🎯 mAP@0.5:        {metrics.box.map50:.4f}")
        print(f"   🎯 mAP@0.5:0.95:   {metrics.box.map:.4f}")
        print(f"   🎯 Precision:      {metrics.box.mp:.4f}")
        print(f"   🎯 Recall:         {metrics.box.mr:.4f}")
        
        # Verificar si cumple objetivos
        if metrics.box.map50 >= 0.75:
            print("\n   ✅ ¡EXCELENTE! El modelo cumple el objetivo (mAP@0.5 >= 0.75)")
        else:
            print("\n   ⚠️  mAP@0.5 por debajo del objetivo. Considera entrenar más epochs.")
        
    except Exception as e:
        print(f"⚠️  No se pudo evaluar: {e}")
        metrics = None
    
    # 7. Guardar modelo
    print("\n💾 GUARDANDO MODELO...")
    best_model_path = Path("models_assets/yolov8_helmet_detection/weights/best.pt")
    
    if best_model_path.exists():
        destination = Path("models_assets/yolov8_helmet_vest_best.pt")
        shutil.copy(best_model_path, destination)
        
        size_mb = destination.stat().st_size / (1024 * 1024)
        
        print_banner("✅ MODELO GUARDADO EXITOSAMENTE")
        print(f"   📁 Ubicación: {destination}")
        print(f"   💾 Tamaño: {size_mb:.2f} MB")
        print(f"\n   📂 Resultados completos en:")
        print(f"      {best_model_path.parent.parent}")
    else:
        print("❌ No se encontró el modelo entrenado")
        return
    
    # 8. Resumen final
    print_banner("🎉 FASE 1 COMPLETADA - FINE-TUNING EXITOSO")
    
    print("📊 RESUMEN:")
    print(f"   🔹 Dataset: {len(train_images):,} imágenes de entrenamiento")
    print(f"   🔹 Modelo: YOLOv8n")
    print(f"   🔹 Epochs: {config['epochs']}")
    print(f"   🔹 Duración: {duration}")
    if metrics:
        print(f"   🔹 mAP@0.5: {metrics.box.map50:.4f}")
    
    print("\n🚀 PRÓXIMOS PASOS:")
    print("   1️⃣ Revisar gráficas en: models_assets/yolov8_helmet_detection/")
    print("   2️⃣ Commitear cambios:")
    print("      git add .")
    if metrics:
        print(f"      git commit -m 'feat: train YOLOv8 model with mAP={metrics.box.map50:.3f}'")
    else:
        print(f"      git commit -m 'feat: train YOLOv8 model on hard hat detection'")
    print("      git push")
    print("   3️⃣ Continuar con FASE 2: Pipeline de Inferencia")
    
    print("\n" + "="*70)
    print("  🎊 ¡EXCELENTE TRABAJO MANITO! 🎊")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()

# 🚀 GUÍA RÁPIDA: Entrenar en Google Colab

## ✅ ARCHIVOS PREPARADOS:

1. ✅ **Notebook de Colab:** `COLAB_TRAINING.ipynb` 
2. ✅ **Dataset comprimido:** `datasets/helmet_vest_detection.zip` (si terminó la compresión)

---

## 📋 PASOS PARA ENTRENAR EN COLAB:

### 1️⃣ Abrir Google Colab

Ve a: **https://colab.research.google.com/**

### 2️⃣ Subir el Notebook

1. Clic en **File → Upload notebook**
2. Selecciona: `COLAB_TRAINING.ipynb` de tu PC
3. O arrastra el archivo al navegador

### 3️⃣ Activar GPU ⚡

**CRÍTICO - HAZ ESTO PRIMERO:**

1. Menu: **Runtime → Change runtime type**
2. Hardware accelerator: **T4 GPU** (o GPU)
3. Clic en **Save**

### 4️⃣ Ejecutar el Notebook

**Opción A: Ejecutar todo de una vez**
- Menu: **Runtime → Run all**
- Espera 40-80 minutos

**Opción B: Ejecutar celda por celda (Recomendado)**
- Presiona `Shift + Enter` en cada celda
- Ve viendo los resultados

### 5️⃣ Subir el Dataset

**Cuando llegues a la celda "Subir ZIP":**

1. La celda te pedirá subir un archivo
2. Selecciona: `helmet_vest_detection.zip`
3. Espera 5-10 minutos a que suba y descomprima

### 6️⃣ Esperar el Entrenamiento ⏰

- **Con GPU T4:** 40-80 minutos
- **NO CIERRES LA PESTAÑA** del navegador
- Puedes hacer otras cosas pero deja Colab abierto

### 7️⃣ Descargar el Modelo

Cuando termine, la última celda descargará automáticamente:
- ✅ `yolov8_helmet_vest_best_mAP0.XXX.pt` (el modelo entrenado)
- ✅ `training_results.png` (gráficas)
- ✅ `confusion_matrix.png` (matriz de confusión)

### 8️⃣ Mover Archivos a tu Proyecto

1. Los archivos se descargarán a tu carpeta **Descargas**
2. Copia el modelo `.pt` a:
   ```
   C:\Users\LENOVO\OneDrive\Documentos\SISTEMASCORE\PROYECTOS\Safety-Vision-AI\models_assets\yolov8_helmet_vest_best.pt
   ```
3. Copia las imágenes a la misma carpeta (opcional)

---

## 💡 CONSEJOS IMPORTANTES:

### ✅ QUÉ HACER:
- ✓ Activa GPU ANTES de ejecutar
- ✓ Mantén la pestaña abierta
- ✓ Descarga los archivos al terminar
- ✓ Si falla, reinicia y vuelve a intentar

### ❌ QUÉ NO HACER:
- ✗ No cierres la pestaña durante el entrenamiento
- ✗ No cambies de GPU a CPU a mitad de camino
- ✗ No olvides descargar el modelo al terminar

---

## ⚠️ TROUBLESHOOTING:

### "No GPU detected"
**Solución:** Runtime → Change runtime type → T4 GPU → Save

### "Disconnected due to inactivity"
**Solución:** Haz clic en la ventana cada 30 minutos o instala esta extensión:
https://chrome.google.com/webstore/detail/colab-auto-refresh

### Error al subir ZIP
**Solución:** 
- Verifica que el archivo no sea mayor a 500 MB
- Si es muy grande, usa Google Drive (Opción A en el notebook)

### Entrenamiento muy lento
**Solución:** Verifica que la GPU esté activa:
- Ejecuta la primera celda
- Debe decir: "✅ GPU: Tesla T4" o similar

---

## 🎯 MÉTRICAS ESPERADAS:

Con 5,269 imágenes y GPU:

| Métrica | Objetivo |
|---------|----------|
| mAP@0.5 | **> 0.80** 🔥 |
| mAP@0.5:0.95 | **> 0.60** |
| Precision | **> 0.85** |
| Recall | **> 0.80** |
| Tiempo | **40-80 min** |

---

## 📝 DESPUÉS DEL ENTRENAMIENTO:

```powershell
# En tu PC, commitea los cambios
cd "C:\Users\LENOVO\OneDrive\Documentos\SISTEMASCORE\PROYECTOS\Safety-Vision-AI"

git add models_assets/yolov8_helmet_vest_best.pt
git add COLAB_TRAINING.ipynb
git commit -m "feat: train YOLOv8 model in Colab with mAP=0.XXX"
git push
```

---

## 🚀 LINKS DIRECTOS:

- **Google Colab:** https://colab.research.google.com/
- **GitHub del Proyecto:** https://github.com/leonardobaca7/safety-vision-ai

---

## ✅ CHECKLIST:

- [ ] Abrir Google Colab
- [ ] Subir notebook COLAB_TRAINING.ipynb
- [ ] Activar GPU T4
- [ ] Ejecutar todas las celdas
- [ ] Subir dataset ZIP
- [ ] Esperar 40-80 minutos
- [ ] Descargar modelo entrenado
- [ ] Copiar a models_assets/
- [ ] Commitear a Git
- [ ] ¡Celebrar! 🎉

---

**¡LISTO MANITO! A ENTRENAR CON GPU 🔥**

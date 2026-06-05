# Detección de Flotas Logísticas mediante Clasificación Vehicular con CNN

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-f7931e.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Estado-Evidencia%203%20%E2%80%94%20Completado-brightgreen.svg)](#)
[![Accuracy](https://img.shields.io/badge/Accuracy-94.4%25-blue.svg)](#)
[![AUC](https://img.shields.io/badge/AUC--ROC-0.987-blue.svg)](#)

Este proyecto desarrolla un sistema de visión artificial especializado en la clasificación de vehículos para la optimización de flotas logísticas, utilizando técnicas de **Deep Learning** y **Procesamiento Digital de Imágenes** sobre el dataset CIFAR-10.

---

## Estado del Proyecto: Evidencia 3 — Completado

### Objetivos Específicos (OE)
| ID | Objetivo | Estado | Avance |
| :--- | :--- | :--- | :--- |
| **OE1** | Analizar y preprocesar el dataset | `✅ COMPLETADO` | 100% |
| **OE2** | Aplicar técnicas de procesamiento de imágenes (Sobel, Canny, LoG) | `✅ COMPLETADO` | 100% |
| **OE3** | Diseñar y entrenar el modelo CNN | `✅ COMPLETADO` | 100% |
| **OE4** | Evaluar el rendimiento del modelo | `✅ COMPLETADO` | 100% |
| **OE5** | Cuantificar el impacto logístico (KPI) | `✅ COMPLETADO` | 100% |
| **OE6** | Informe final | `✅ COMPLETADO` | 100% |

---

## Resultados Finales

| Métrica | MLP Baseline (E2) | CNN (E3) | Mejora |
| :--- | :---: | :---: | :---: |
| Accuracy | ~70% | **94.4%** | +24.4 pp |
| AUC-ROC | ~0.72 | **0.987** | +0.267 |
| F1-Score (Pesado) | ~0.70 | **0.94** | +0.24 |
| Precision (Pesado) | — | **0.94** | — |
| Recall (Pesado) | — | **0.95** | — |
| Parámetros | ~1.450.000 | **~424.385** | 3.4× menos |
| KPI logístico (Δ real vs pred) | — | **0.9 pp** | — |

---

## Estructura del Repositorio

```
.
├── ev2_veh_classifier.ipynb       # Evidencia 2: MLP baseline
├── ev3_veh_classifier_cnn.ipynb   # Evidencia 3: CNN completa (notebook principal)
├── README.md
└── informes/
    ├── E2_ProcesamientoImagenes.pdf
    └── E3_ProcesamientoImagenes.docx
```

---

## Uso

### Requisitos
```bash
pip install tensorflow tensorflow-datasets opencv-python scikit-learn matplotlib numpy
```

### Ejecutar en Google Colab
1. Abrir `ev3_veh_classifier_cnn.ipynb` en Colab
2. Seleccionar **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)**
3. Ejecutar todas las celdas en orden (`Entorno de ejecución → Ejecutar todo`)

### Inferencia sobre imagen nueva
```python
import numpy as np
from tensorflow.keras.models import load_model

modelo = load_model('ev3_cnn_vehicular_final.keras')

# imagen: array numpy (32, 32, 3) con valores float [0.0, 1.0]
img_batch    = np.expand_dims(imagen, axis=0)          # (1, 32, 32, 3)
prob_pesado  = float(modelo.predict(img_batch)[0][0])  # P(Pesado)
clase        = 'Pesado' if prob_pesado >= 0.5 else 'Liviano'
print(f'{clase}  —  confianza: {max(prob_pesado, 1-prob_pesado)*100:.1f}%')
```

---

## Notebooks por Evidencia

| Notebook | Descripción | Accuracy |
| :--- | :--- | :---: |
| `ev2_veh_classifier.ipynb` | MLP baseline — valida el pipeline de datos | ~70% |
| `ev3_veh_classifier_cnn.ipynb` | CNN definitiva — modelo de producción | **94.4%** |

---

## Equipo

**Mario Arce · Ariel Denaro · Simón Vottero**  
ISPC — Procesamiento de Imágenes — TSCDIA 2026  
Docente: Carlos Ignacio Charletti

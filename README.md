# 🚗 Detección de Flotas Logísticas mediante Clasificación Vehicular con CNN

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Estado-Evidencia%202-green.svg)](#)

Este proyecto desarrolla un sistema de visión artificial especializado en la clasificación de vehículos para la optimización de flotas logísticas, utilizando técnicas de **Deep Learning** y **Procesamiento Digital de Imágenes**.

---

## 📊 Estado del Proyecto: Evidencia 2
En esta etapa, se ha validado el pipeline de datos completo y se ha establecido una **línea base (baseline)** de rendimiento.

### Objetivos Específicos (OE) y Avances
| ID | Objetivo | Estado | Avance |
| :--- | :--- | :--- | :--- |
| **OE1** | Analizar y preprocesar el dataset | `COMPLETADO` | 100% |
| **OE2** | Aplicar técnicas de procesamiento de imágenes | `EN CURSO` | 40% |
| **OE3** | Diseñar y entrenar el modelo CNN | `BASELINE ACTIVO` | 25% |

---

## ⚙️ Pipeline de Datos Implementado
Se utiliza el dataset **CIFAR-10**, filtrando específicamente las clases de interés para el sector logístico:

* **Clases seleccionadas:** `Automobile` (Etiqueta 1) y `Truck` (Etiqueta 9).
* **Reasignación binaria:** $0 =$ Liviano, $1 =$ Pesado.
* **Normalización:** Escala lineal de píxeles al rango $[0, 1]$ para estabilidad de gradientes.
* **Optimización:** Uso de `.cache()` en RAM para acelerar épocas posteriores y `.shuffle()` para evitar sesgos.

---

## 🧠 Arquitectura Baseline (Red Densa)
Para confirmar la integridad del flujo de datos antes de pasar a arquitecturas convolucionales complejas, se implementó un **Perceptrón Multicapa (MLP)**.

> **Justificación técnica:** Una red densa sin invariancia espacial define el umbral mínimo de precisión que la CNN final deberá superar.

```python
# Resumen de la arquitectura MLP
Input(shape=(32, 32, 3))
Flatten() # 3,072 valores de entrada
Dense(128, activation='relu')
Dense(128, activation='relu')
Dense(2, activation='softmax') # Clasificación binaria

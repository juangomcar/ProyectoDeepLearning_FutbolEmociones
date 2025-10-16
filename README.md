# 🎯 ProyectoDeepLearning_FutbolEmociones

[![Estado](https://img.shields.io/badge/estado-en%20desarrollo-4C9AFF)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](#)
[![Stars](https://img.shields.io/github/stars/estemen27/ProyectoDeepLearning_FutbolEmociones)](../../stargazers)

> Detección de **emociones** en jugadores de fútbol usando **CNNs**. El proyecto incluye notebooks reproducibles, métricas claras, visualizaciones y un informe técnico.

---

## 🗺️ Tabla de Contenidos
- [Descripción](#-descripción)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Requisitos](#-requisitos)
- [Instalación rápida](#-instalación-rápida)
- [Dataset](#-dataset)
- [Arquitectura y flujo](#-arquitectura-y-flujo)
- [Entrenamiento y evaluación](#-entrenamiento-y-evaluación)
- [Resultados](#-resultados)
- [Ejemplos](#-ejemplos)
- [Roadmap](#-roadmap)
- [Contribuir](#-contribuir)
- [Citar](#-citar)
- [Licencia](#-licencia)

---

## 🧠 Descripción
Este proyecto entrena y evalúa una **Red Neuronal Convolucional (CNN)** para clasificar emociones (ej. *surprise*, *happy*, *neutral*, etc.) en imágenes de escenas futbolísticas. Se incluyen:
- Preprocesamiento y limpieza
- Aumento de datos
- Entrenamiento con *callbacks* (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Métricas y visualizaciones (matriz de confusión, curvas de aprendizaje)
- Inferencia y exportación del modelo

El informe técnico se encuentra en **[`report/Reporte CNN.pdf`](report/Reporte%20CNN.pdf)**.

---

## 🗂️ Estructura del repositorio

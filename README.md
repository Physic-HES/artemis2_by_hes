# Simulación Numérica de la Misión Artemis II

Este repositorio contiene el desarrollo, los códigos y los resultados de una simulación numérica de la trayectoria de la cápsula Orion durante la misión **Artemis II**. El proyecto fue realizado como un trabajo inicial de la materia **Mecánica Clásica**, dictada en el primer cuatrimestre de 2026 por el **Prof. Ricardo Mindlin** en la Licenciatura en Ciencias Físicas de la **Universidad de Buenos Aires (UBA)**.

## Resumen del Proyecto

El objetivo principal es modelar con alta fidelidad la trayectoria de la nave Orion, integrando no solo las interacciones gravitatorias en un sistema de tres cuerpos (Tierra-Luna-Nave), sino también los aportes no conservativos derivados de los encendidos de los motores (*burns*).

El trabajo se divide en dos fases críticas:
1.  **Fase de Diagnóstico:** Extracción del perfil de aceleración de los motores a partir de datos de telemetría real mediante un método de "cociente incremental vectorial".
2.  **Fase de Propagación:** Reconstrucción de la trayectoria completa mediante una integración de Cowell de alto orden (DOP853), utilizando las aceleraciones extraídas y un ajuste empírico para compensar efectos no modelados (presión de radiación solar, errores de potencial, etc.).

## Estructura del Repositorio

*   `Simulacion_Artemis_II.pdf`: Informe técnico detallado con el marco teórico (formalismo Lagrangiano en sistemas no inerciales), metodología y discusión de resultados.
*   `artemis_pred_from_telemetry.py`: Script encargado de la extracción de aceleraciones no conservativas comparando la telemetría con el modelo puramente gravitatorio.
*   `artemis_pred_with_burns_v2.py`: Código principal de simulación que realiza la integración por bloques utilizando el integrador de paso adaptativo DOP853.
*   `artemis_get_burns.mp4`: Video demostrativo del proceso de captura de encendidos nominales y correcciones.
*   `OEM_2026.04.02_post_USS_to_EI_v2.asc`: Datos de telemetría original en formato OEM (Orbit Ephemeris Message).
*   `aceleracion_motores.asc`: Aceleraciones obtenidas a partir de la comparacion de velocidades Modelo-Realidad.

## Marco Teórico

El movimiento se describe en un sistema de referencia geocéntrico no inercial. La ecuación de movimiento fundamental derivada en el trabajo es:

$$\ddot{\mathbf{r}} = -\mu_T \frac{\mathbf{r}}{r^3} - \mu_L \left( \frac{\mathbf{r} - \mathbf{r}_L}{|\mathbf{r} - \mathbf{r}_L|^3} + \frac{\mathbf{r}_L}{r_L^3} \right) + \mathbf{a}_{burn}$$

Donde se considera la perturbación lunar (términos directo e indirecto) y la aceleración efectiva de los motores $\mathbf{a}_{burn}$.

## Requisitos

Para ejecutar las simulaciones se requiere Python 3.x con las siguientes librerías:
*   `numpy`
*   `scipy` (específicamente para el integrador `DOP853`)
*   `matplotlib` (para la generación de gráficos y trayectorias 3D)
*   `pandas` (para el manejo de datos de telemetría en CSV)

## Autor
*   **Hugo Ernesto Sosa** - Facultad de Ciencias Exactas y Naturales, UBA.

---
*Este trabajo fue presentado en Abril de 2026 como parte de la currícula de Mecánica Clásica.*

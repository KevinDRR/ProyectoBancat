# Implementacion del Modelo de Riesgo — Banco Gatuno

## Descripcion general

La aplicacion web usa un modelo de riesgo crediticio en PyTorch para estimar la probabilidad de alto riesgo de cada solicitud. El objetivo ya no es solo clasificar con una red minima, sino sostener una inferencia mas estable para la app del banco, con entrenamiento reproducible, umbral calibrado y explicaciones basicas para el frontend y la vista admin.

Importante: el pipeline sigue usando datos sinteticos. Eso permite una demo mucho mas robusta que antes, pero no convierte el modelo en un score productivo real. Para uso bancario real hace falta entrenarlo con historicos etiquetados de mora, fraude, castigos y comportamiento transaccional.

## Arquitectura actual

El modelo vive en `modelo/risk_model.py` y comparte la misma arquitectura para entrenamiento e inferencia.

Arquitectura:

Entrada (9 features) → 64 → 32 → 16 → salida logit

Detalles:

- Capas fully connected
- BatchNorm en capas ocultas
- Activacion SiLU
- Dropout para mejorar generalizacion
- `BCEWithLogitsLoss` en lugar de `BCELoss`

La decision final no depende de un 0.5 fijo. El entrenamiento selecciona un umbral operativo y lo guarda en `modelo/normalizacion.json` como `decision_threshold`.

## Variables del modelo

El frontend sigue pidiendo los mismos datos basicos del cliente, pero el modelo trabaja con variables derivadas mas utiles:

| Feature | Origen |
| --- | --- |
| `edad` | Formulario |
| `log_ingresos` | `log1p(ingresos)` |
| `estado_civil` | Formulario |
| `log_deudas` | `log1p(deudas_existentes)` |
| `relacion_deuda_ingreso` | `deudas_existentes / ingresos` |
| `log_saldo` | `log1p(saldo_cuentas)` |
| `reserva_meses` | `saldo_cuentas / ingresos` |
| `cobertura_deuda` | `saldo_cuentas / deudas_existentes` |
| `factor_producto` | Riesgo relativo del producto solicitado |

Esto hace que la web ya no dependa solo de montos brutos; tambien incorpora liquidez, cobertura y el tipo de producto solicitado.

## Generacion de datos sinteticos

`modelo/entrenar.py` ahora genera un dataset sintetico bastante mas realista:

- 32,000 perfiles sinteticos
- 4 segmentos de clientes con distribuciones distintas
- mezcla de productos crediticios con factores de riesgo diferentes
- ingresos, deuda y saldo correlacionados entre si
- componente latente de volatilidad para evitar un dataset demasiado trivial

La etiqueta final se genera como una probabilidad de default sintetica y no como una mediana arbitraria. Eso produce una frontera de decision mas parecida a un problema de riesgo real.

## Entrenamiento

Configuracion principal:

- Optimizador: `AdamW`
- Learning rate: `8e-4`
- Weight decay: `2e-4`
- Batch size: `512`
- Max epochs: `90`
- Early stopping: `12` epocas sin mejora
- Scheduler: `ReduceLROnPlateau`
- Split estratificado: `70% train / 15% validacion / 15% test`

Durante el entrenamiento se guarda la mejor epoca segun `val_loss`, y luego se calibra un umbral operativo usando precision, recall y especificidad sobre validacion.

## Resultados del ultimo entrenamiento

El ultimo entrenamiento ejecutado en este proyecto produjo, de forma aproximada:

- `Val AUC`: `0.9555`
- `Test AUC`: `0.9628`
- `Test Accuracy`: `0.8694`
- `Test Balanced Accuracy`: `0.9271`

Estas metricas deben interpretarse como validacion interna sobre datos sinteticos, no como evidencia de desempeno real en produccion.

## Archivos generados

- `modelo/modelo_riesgo.pth`: pesos del modelo entrenado
- `modelo/normalizacion.json`: media, desviacion, umbral operativo y metadata del modelo
- `modelo/training_report.json`: resumen del entrenamiento y metricas de validacion/test

## Inferencia en la API

La API carga el modelo entrenado desde `main.py` y usa exactamente la misma ingenieria de variables que el entrenamiento.

Flujo:

1. Se reciben edad, ingresos, estado civil, deudas, saldo y producto.
2. Se calcula el `factor_producto` segun la linea solicitada.
3. Se construyen las 9 features derivadas.
4. Se normalizan con la media y desviacion guardadas.
5. El modelo devuelve una probabilidad de alto riesgo.
6. La aprobacion se decide con `decision_threshold`, no con un `0.5` fijo.
7. La respuesta incluye `nivel_riesgo` y `factores_riesgo` para explicar la prediccion.

## Como reentrenar el modelo

```bash
# Activar entorno virtual
.\.venv\Scripts\Activate.ps1

# Reentrenar
python modelo/entrenar.py
```

Eso regenera:

- `modelo/modelo_riesgo.pth`
- `modelo/normalizacion.json`
- `modelo/training_report.json`

Reinicia la API para que la web cargue los nuevos pesos y el nuevo umbral.

## Dependencias

```txt
torch
numpy
fastapi
uvicorn[standard]
jinja2
```

## Ejecucion de la app

Direccion:
http://127.0.0.1:8000

Start:
python -m uvicorn main:app --reload --port 8000

Entorno:
.\.venv\Scripts\Activate.ps1

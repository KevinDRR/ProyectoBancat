from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


FEATURE_NAMES = [
    "edad",
    "log_ingresos",
    "estado_civil",
    "log_deudas",
    "relacion_deuda_ingreso",
    "log_saldo",
    "reserva_meses",
    "cobertura_deuda",
    "factor_producto",
]

MODEL_HIDDEN_DIMS = (64, 32, 16)
MODEL_DROPOUT = 0.15
DEFAULT_THRESHOLD = 0.46

PRODUCT_RISK_FACTORS = {
    "Tarjeta de Credito Oro": 1.02,
    "Tarjeta Black Signature": 1.14,
    "Credito Libre Inversion": 1.10,
    "Credito Hipotecario Verde": 0.86,
    "Credito Vehicular Inteligente": 0.95,
    "Microcredito Pyme": 1.22,
}

CATEGORY_RISK_FACTORS = {
    "Tarjeta": 1.08,
    "Credito": 1.00,
}


class RedNeuronalRiesgo(nn.Module):
    def __init__(
        self,
        entrada: int,
        hidden_dims: tuple[int, ...] = MODEL_HIDDEN_DIMS,
        dropout: float = MODEL_DROPOUT,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = entrada
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout if idx == 0 else dropout * 0.7))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def obtener_factor_producto(producto: dict | str | None) -> float:
    if isinstance(producto, dict):
        nombre = producto.get("nombre")
        categoria = producto.get("categoria")
    else:
        nombre = producto
        categoria = None

    if nombre in PRODUCT_RISK_FACTORS:
        return PRODUCT_RISK_FACTORS[nombre]

    if categoria in CATEGORY_RISK_FACTORS:
        return CATEGORY_RISK_FACTORS[categoria]

    return 1.0


def _to_array(value) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=np.float32))


def calcular_metricas_financieras(
    ingresos,
    deudas_existentes,
    saldo_cuentas,
) -> dict[str, np.ndarray]:
    ingresos = _to_array(ingresos)
    deudas_existentes = _to_array(deudas_existentes)
    saldo_cuentas = _to_array(saldo_cuentas)

    ingresos_seguro = np.maximum(ingresos, 1.0)
    deudas_seguro = np.maximum(deudas_existentes, 1.0)

    relacion_deuda_ingreso = np.clip(deudas_existentes / ingresos_seguro, 0.0, 6.0)
    reserva_meses = np.clip(saldo_cuentas / ingresos_seguro, 0.0, 24.0)
    cobertura_deuda = np.clip(saldo_cuentas / deudas_seguro, 0.0, 12.0)

    return {
        "relacion_deuda_ingreso": relacion_deuda_ingreso,
        "reserva_meses": reserva_meses,
        "cobertura_deuda": cobertura_deuda,
    }


def construir_features(
    edad,
    ingresos,
    estado_civil,
    deudas_existentes,
    saldo_cuentas,
    factor_producto,
) -> np.ndarray:
    edad = _to_array(edad)
    ingresos = _to_array(ingresos)
    estado_civil = _to_array(estado_civil)
    deudas_existentes = _to_array(deudas_existentes)
    saldo_cuentas = _to_array(saldo_cuentas)
    factor_producto = _to_array(factor_producto)

    metricas = calcular_metricas_financieras(ingresos, deudas_existentes, saldo_cuentas)

    features = np.column_stack(
        [
            edad,
            np.log1p(np.clip(ingresos, 0.0, None)),
            estado_civil,
            np.log1p(np.clip(deudas_existentes, 0.0, None)),
            metricas["relacion_deuda_ingreso"],
            np.log1p(np.clip(saldo_cuentas, 0.0, None)),
            metricas["reserva_meses"],
            metricas["cobertura_deuda"],
            factor_producto,
        ]
    ).astype(np.float32)

    if features.shape[0] == 1:
        return features[0]
    return features


def clasificar_nivel_riesgo(probabilidad: float, threshold: float = DEFAULT_THRESHOLD) -> str:
    if probabilidad < threshold * 0.55:
        return "bajo"
    if probabilidad < threshold:
        return "moderado"
    if probabilidad < min(0.85, threshold + 0.18):
        return "alto"
    return "critico"


def construir_alertas_riesgo(
    edad: float,
    ingresos: float,
    deudas_existentes: float,
    saldo_cuentas: float,
    factor_producto: float,
) -> list[str]:
    metricas = calcular_metricas_financieras(ingresos, deudas_existentes, saldo_cuentas)

    relacion_deuda_ingreso = float(metricas["relacion_deuda_ingreso"][0])
    reserva_meses = float(metricas["reserva_meses"][0])
    cobertura_deuda = float(metricas["cobertura_deuda"][0])

    alertas: list[str] = []
    if relacion_deuda_ingreso > 1.0:
        alertas.append("Relacion deuda/ingreso elevada")
    elif relacion_deuda_ingreso < 0.35:
        alertas.append("Carga de deuda saludable")

    if reserva_meses < 1.0:
        alertas.append("Liquidez baja frente al ingreso")
    elif reserva_meses >= 5.0:
        alertas.append("Buen colchon de liquidez")

    if cobertura_deuda < 0.5 and deudas_existentes > 0:
        alertas.append("Cobertura de deuda limitada")
    elif cobertura_deuda >= 2.0:
        alertas.append("Cobertura de deuda solida")

    if edad < 23 or edad > 70:
        alertas.append("Perfil etario con mayor volatilidad")

    if ingresos < 2_500_000:
        alertas.append("Ingreso mensual ajustado")
    elif ingresos > 9_000_000:
        alertas.append("Ingresos altos frente al portafolio")

    if factor_producto >= 1.12:
        alertas.append("Producto con politica de riesgo mas exigente")
    elif factor_producto <= 0.9:
        alertas.append("Producto con perfil historicamente mas estable")

    return alertas[:3]
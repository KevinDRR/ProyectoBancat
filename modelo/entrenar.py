from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from modelo.risk_model import (
    DEFAULT_THRESHOLD,
    FEATURE_NAMES,
    MODEL_DROPOUT,
    MODEL_HIDDEN_DIMS,
    PRODUCT_RISK_FACTORS,
    RedNeuronalRiesgo,
    construir_features,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelo" / "modelo_riesgo.pth"
STATS_PATH = BASE_DIR / "modelo" / "normalizacion.json"
REPORT_PATH = BASE_DIR / "modelo" / "training_report.json"

SEED = 42
DATASET_SIZE = 32000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
BATCH_SIZE = 512
MAX_EPOCHS = 90
PATIENCE = 12
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 2e-4

torch.manual_seed(SEED)
np.random.seed(SEED)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def roc_auc_score_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(int)
    scores = scores.astype(float)

    positivos = y_true.sum()
    negativos = len(y_true) - positivos
    if positivos == 0 or negativos == 0:
        return 0.5

    order = np.argsort(scores)
    ranks = np.zeros(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1

    sum_ranks_positive = ranks[y_true == 1].sum()
    return float((sum_ranks_positive - positivos * (positivos + 1) / 2.0) / (positivos * negativos))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y_pred == 1) & (y_true == 1))),
        "fp": int(np.sum((y_pred == 1) & (y_true == 0))),
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
        "fn": int(np.sum((y_pred == 0) & (y_true == 1))),
    }


def classification_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float | dict[str, int]]:
    y_pred = (probs >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = tp + fp + tn + fn

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / max(total, 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    brier = float(np.mean((probs - y_true) ** 2))
    auc = roc_auc_score_np(y_true, probs)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "brier": brier,
        "auc": auc,
        "confusion": counts,
    }


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float | dict[str, int]]]:
    best_threshold = DEFAULT_THRESHOLD
    best_metrics = classification_metrics(y_true, probs, best_threshold)
    best_score = 0.65 * best_metrics["recall"] + 0.2 * best_metrics["precision"] + 0.15 * best_metrics["specificity"]

    for threshold in np.linspace(0.25, 0.75, 101):
        metrics = classification_metrics(y_true, probs, float(threshold))
        score = 0.65 * metrics["recall"] + 0.2 * metrics["precision"] + 0.15 * metrics["specificity"]
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def stratified_split(y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for clase in (0, 1):
        indices = np.where(y == clase)[0]
        rng.shuffle(indices)

        n_train = int(len(indices) * TRAIN_RATIO)
        n_val = int(len(indices) * VAL_RATIO)

        train_indices.append(indices[:n_train])
        val_indices.append(indices[n_train : n_train + n_val])
        test_indices.append(indices[n_train + n_val :])

    train = np.concatenate(train_indices)
    val = np.concatenate(val_indices)
    test = np.concatenate(test_indices)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def generar_dataset_sintetico(n_muestras: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    rng = np.random.default_rng(seed)

    productos = np.array(list(PRODUCT_RISK_FACTORS.keys()))
    factores = np.array([PRODUCT_RISK_FACTORS[producto] for producto in productos], dtype=np.float32)

    segmentos = rng.choice(4, size=n_muestras, p=[0.27, 0.31, 0.26, 0.16])
    distribucion_productos = np.array(
        [
            [0.08, 0.10, 0.18, 0.28, 0.18, 0.18],
            [0.16, 0.12, 0.28, 0.14, 0.20, 0.10],
            [0.18, 0.12, 0.24, 0.12, 0.18, 0.16],
            [0.12, 0.08, 0.18, 0.10, 0.10, 0.42],
        ],
        dtype=np.float32,
    )

    producto_idx = np.array(
        [rng.choice(len(productos), p=distribucion_productos[segmento]) for segmento in segmentos],
        dtype=int,
    )
    factor_producto = factores[producto_idx]

    edad_media = np.array([48, 36, 31, 39], dtype=np.float32)
    edad_desv = np.array([8, 7, 6, 9], dtype=np.float32)
    edad = np.clip(rng.normal(edad_media[segmentos], edad_desv[segmentos]), 21, 75)

    ingreso_log_media = np.array([15.85, 15.40, 15.05, 14.70], dtype=np.float32)
    ingreso_log_desv = np.array([0.35, 0.42, 0.45, 0.48], dtype=np.float32)
    ingresos = np.exp(rng.normal(ingreso_log_media[segmentos], ingreso_log_desv[segmentos]))
    ingresos = np.clip(ingresos, 1_200_000, 30_000_000)

    prob_casado = np.clip(0.12 + 0.013 * (edad - 21) + np.array([0.10, 0.05, 0.0, -0.03])[segmentos], 0.05, 0.92)
    estado_civil = rng.binomial(1, prob_casado).astype(np.float32)

    dti_objetivo = np.clip(
        rng.normal(np.array([0.24, 0.52, 0.92, 1.48])[segmentos] + (factor_producto - 1.0) * 0.28, 0.18),
        0.0,
        4.6,
    )
    deudas_existentes = ingresos * dti_objetivo * rng.uniform(0.85, 1.25, n_muestras)
    deudas_existentes = np.clip(deudas_existentes, 0, 120_000_000)

    reserva_objetivo = np.clip(
        rng.normal(np.array([8.0, 4.2, 1.8, 0.55])[segmentos] - dti_objetivo * 0.9 + estado_civil * 0.35, 1.2),
        0.0,
        24.0,
    )
    saldo_cuentas = ingresos * reserva_objetivo * rng.uniform(0.75, 1.3, n_muestras)
    saldo_cuentas = np.clip(saldo_cuentas, 0, 140_000_000)

    if len(saldo_cuentas) > 0:
        n_sin_ahorro = int(0.08 * n_muestras)
        saldo_cuentas[rng.choice(n_muestras, size=n_sin_ahorro, replace=False)] *= rng.uniform(0.02, 0.18, n_sin_ahorro)

    if len(deudas_existentes) > 0:
        n_baja_deuda = int(0.12 * n_muestras)
        deudas_existentes[rng.choice(n_muestras, size=n_baja_deuda, replace=False)] *= rng.uniform(0.01, 0.18, n_baja_deuda)

    relacion_deuda_ingreso = np.clip(deudas_existentes / np.maximum(ingresos, 1.0), 0.0, 6.0)
    reserva_meses = np.clip(saldo_cuentas / np.maximum(ingresos, 1.0), 0.0, 24.0)
    cobertura_deuda = np.clip(saldo_cuentas / np.maximum(deudas_existentes, 1.0), 0.0, 12.0)

    edad_riesgo = np.where(edad < 24, 0.32, 0.0) + np.where(edad > 69, 0.24, 0.0)
    volatilidad_latente = rng.normal(np.array([-0.48, -0.08, 0.28, 0.74])[segmentos], 0.32)
    ruido = rng.normal(0.0, 0.28, n_muestras)

    score = (
        -1.20
        + 1.35 * relacion_deuda_ingreso
        - 0.17 * np.log1p(ingresos)
        - 0.28 * np.log1p(saldo_cuentas)
        - 0.26 * cobertura_deuda
        - 0.22 * reserva_meses
        + 0.58 * (factor_producto - 1.0)
        + 0.14 * estado_civil
        + edad_riesgo
        + volatilidad_latente
        + ruido
    )

    prob_default = sigmoid(score)
    etiquetas = rng.binomial(1, prob_default).astype(np.float32)

    X = construir_features(
        edad,
        ingresos,
        estado_civil,
        deudas_existentes,
        saldo_cuentas,
        factor_producto,
    )

    resumen = {
        "muestras": int(n_muestras),
        "prevalencia_riesgo": float(etiquetas.mean()),
        "prob_default_media": float(prob_default.mean()),
    }
    return X, etiquetas, resumen


def evaluate_model(modelo: nn.Module, features: np.ndarray) -> np.ndarray:
    modelo.eval()
    with torch.no_grad():
        logits = modelo(torch.tensor(features, dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


def main():
    print("\nGenerando dataset sintetico robusto...")
    X, y, resumen_dataset = generar_dataset_sintetico(DATASET_SIZE, SEED)
    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = stratified_split(y, rng)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    media = X_train.mean(axis=0)
    desviacion = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - media) / desviacion
    X_val_norm = (X_val - media) / desviacion
    X_test_norm = (X_test - media) / desviacion

    print(f"  Total de muestras generadas: {len(X):,}")
    print(f"  Riesgo positivo estimado:   {resumen_dataset['prevalencia_riesgo'] * 100:.1f}%")
    print(f"  Train / Val / Test:         {len(train_idx):,} / {len(val_idx):,} / {len(test_idx):,}")
    print(f"  Variables de entrada:       {len(FEATURE_NAMES)}")

    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_tensor = torch.tensor(X_val_norm, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    pos = float(y_train.sum())
    neg = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)

    modelo = RedNeuronalRiesgo(len(FEATURE_NAMES), hidden_dims=MODEL_HIDDEN_DIMS, dropout=MODEL_DROPOUT)
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizador = optim.AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="min", factor=0.5, patience=4)

    mejor_val_loss = float("inf")
    mejor_epoca = 0
    contador_paciencia = 0
    mejores_pesos = copy.deepcopy(modelo.state_dict())

    print("\nEntrenando red neuronal...")
    for epoch in range(MAX_EPOCHS):
        modelo.train()
        running_loss = 0.0

        for batch_features, batch_labels in train_loader:
            logits = modelo(batch_features)
            loss = criterio(logits, batch_labels)

            optimizador.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=2.0)
            optimizador.step()

            running_loss += loss.item() * batch_features.size(0)

        train_loss = running_loss / len(train_dataset)

        modelo.eval()
        with torch.no_grad():
            val_logits = modelo(val_tensor)
            val_loss = criterio(val_logits, y_val_tensor).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().reshape(-1)
            val_auc = roc_auc_score_np(y_val, val_probs)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoca {epoch + 1:>3}/{MAX_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}"
            )

        if val_loss < mejor_val_loss:
            mejor_val_loss = val_loss
            mejor_epoca = epoch + 1
            contador_paciencia = 0
            mejores_pesos = copy.deepcopy(modelo.state_dict())
        else:
            contador_paciencia += 1

        if contador_paciencia >= PATIENCE:
            print(f"\n  Early stopping en la epoca {epoch + 1}")
            break

    modelo.load_state_dict(mejores_pesos)
    print(f"  Mejor epoca: {mejor_epoca} con Val Loss: {mejor_val_loss:.4f}")

    val_probs = evaluate_model(modelo, X_val_norm)
    threshold, val_metrics = choose_threshold(y_val.astype(int), val_probs)
    test_probs = evaluate_model(modelo, X_test_norm)
    test_metrics = classification_metrics(y_test.astype(int), test_probs, threshold)

    print("\nResumen de validacion:")
    print(
        f"  Umbral operativo: {threshold:.3f} | "
        f"AUC: {val_metrics['auc']:.4f} | Recall: {val_metrics['recall']:.4f} | "
        f"Precision: {val_metrics['precision']:.4f}"
    )
    print("\nResumen de test:")
    print(
        f"  Accuracy: {test_metrics['accuracy']:.4f} | AUC: {test_metrics['auc']:.4f} | "
        f"Balanced Acc: {test_metrics['balanced_accuracy']:.4f} | Brier: {test_metrics['brier']:.4f}"
    )

    torch.save(modelo.state_dict(), MODEL_PATH)

    stats = {
        "version": 2,
        "seed": SEED,
        "feature_names": FEATURE_NAMES,
        "media": media.tolist(),
        "desviacion": desviacion.tolist(),
        "hidden_dims": list(MODEL_HIDDEN_DIMS),
        "dropout": MODEL_DROPOUT,
        "decision_threshold": threshold,
        "dataset_size": DATASET_SIZE,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "product_risk_factors": PRODUCT_RISK_FACTORS,
    }
    with open(STATS_PATH, "w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)

    report = {
        "dataset": resumen_dataset,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_epoch": mejor_epoca,
        "paths": {
            "model": str(MODEL_PATH),
            "stats": str(STATS_PATH),
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"\nModelo guardado en {MODEL_PATH}")
    print(f"Estadisticas guardadas en {STATS_PATH}")
    print(f"Reporte guardado en {REPORT_PATH}")


if __name__ == "__main__":
    main()

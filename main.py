from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, EmailStr
from typing import Optional
import torch
import numpy as np
import json
import sqlite3
from pathlib import Path

from modelo.risk_model import (
    DEFAULT_THRESHOLD,
    MODEL_DROPOUT,
    MODEL_HIDDEN_DIMS,
    RedNeuronalRiesgo,
    clasificar_nivel_riesgo,
    construir_alertas_riesgo,
    construir_features,
    obtener_factor_producto,
)

app = FastAPI(title="Banco Gatuno", description="El banco de los gatitos")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "bancat_local.db"

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "src")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "123"
ADMIN_COOKIE_NAME = "bancat_admin"

SOLICITUD_FIELDS = (
    "nombre",
    "correo",
    "edad",
    "ingresos",
    "estado_civil",
    "deudas_existentes",
    "saldo_cuentas",
)

SOLICITUD_SELECT = """
    SELECT
        s.id,
        s.cliente_id,
        s.nombre,
        s.correo,
        s.edad,
        s.ingresos,
        s.estado_civil,
        s.deudas_existentes,
        s.saldo_cuentas,
        s.producto_id,
        p.nombre AS producto_nombre,
        p.categoria AS producto_categoria,
        p.tasa AS producto_tasa,
        s.probabilidad_riesgo,
        s.credito_aprobado,
        s.creado_en,
        s.actualizada_en,
        s.archivada,
        s.archivada_en
    FROM solicitudes s
    INNER JOIN productos p ON p.id = s.producto_id
"""

PRODUCTOS_SEMILLA = [
    {
        "nombre": "Cuenta Ahorro Plus",
        "categoria": "Cuenta",
        "descripcion": "Cuenta de ahorro sin costo de apertura con transferencias digitales y bolsillo de metas.",
        "tasa": "Hasta 6.2% E.A.",
        "beneficio": "Sin cuota de manejo",
        "icono": "CA",
        "destacado": 1,
        "solicitable": 0,
    },
    {
        "nombre": "Cuenta Nomina Premium",
        "categoria": "Cuenta",
        "descripcion": "Recibe tu salario, retira sin costo en cajeros aliados y accede a anticipos de nomina.",
        "tasa": "Hasta 4.5% E.A.",
        "beneficio": "Retiros gratis nacionales",
        "icono": "CN",
        "destacado": 1,
        "solicitable": 0,
    },
    {
        "nombre": "Tarjeta de Credito Oro",
        "categoria": "Tarjeta",
        "descripcion": "Acumula millas, difiere compras y administra tus cupos desde la app del banco.",
        "tasa": "Desde 1.75% M.V.",
        "beneficio": "Cashback del 3% en supermercados",
        "icono": "TO",
        "destacado": 1,
        "solicitable": 1,
    },
    {
        "nombre": "Tarjeta Black Signature",
        "categoria": "Tarjeta",
        "descripcion": "Linea premium con salas VIP, asistencias internacionales y cupo flexible.",
        "tasa": "Desde 1.62% M.V.",
        "beneficio": "Acceso a salas VIP y concierge",
        "icono": "TB",
        "destacado": 0,
        "solicitable": 1,
    },
    {
        "nombre": "Credito Libre Inversion",
        "categoria": "Credito",
        "descripcion": "Financia estudios, viajes o remodelaciones con plazos flexibles y desembolso rapido.",
        "tasa": "Desde 1.28% M.V.",
        "beneficio": "Desembolso en menos de 24 horas",
        "icono": "LI",
        "destacado": 1,
        "solicitable": 1,
    },
    {
        "nombre": "Credito Hipotecario Verde",
        "categoria": "Credito",
        "descripcion": "Compra vivienda nueva o usada con apoyo especializado y opcion de cuota fija.",
        "tasa": "Desde 10.9% E.A.",
        "beneficio": "Financiacion hasta 80%",
        "icono": "HV",
        "destacado": 1,
        "solicitable": 1,
    },
    {
        "nombre": "Credito Vehicular Inteligente",
        "categoria": "Credito",
        "descripcion": "Aprobacion agil para carro nuevo o usado con seguro y recaudo automatico.",
        "tasa": "Desde 1.12% M.V.",
        "beneficio": "Preaprobacion digital",
        "icono": "CV",
        "destacado": 0,
        "solicitable": 1,
    },
    {
        "nombre": "Microcredito Pyme",
        "categoria": "Credito",
        "descripcion": "Capital de trabajo para negocios con estudio simplificado y acompanamiento comercial.",
        "tasa": "Desde 1.05% M.V.",
        "beneficio": "Respuesta en el mismo dia habil",
        "icono": "PY",
        "destacado": 0,
        "solicitable": 1,
    },
    {
        "nombre": "CDT Flexible 360",
        "categoria": "Inversion",
        "descripcion": "Invierte a termino fijo con renovacion automatica y simulador de rentabilidad.",
        "tasa": "Hasta 11.4% E.A.",
        "beneficio": "Desde $500.000",
        "icono": "CD",
        "destacado": 0,
        "solicitable": 0,
    },
    {
        "nombre": "Seguro Proteccion Familiar",
        "categoria": "Seguro",
        "descripcion": "Cobertura de vida y desempleo asociada a tus obligaciones financieras.",
        "tasa": "Cuota desde $39.900",
        "beneficio": "Cobertura 24/7",
        "icono": "SG",
        "destacado": 0,
        "solicitable": 0,
    },
]

CLIENTES_DEMO = [
    {
        "nombre": "Laura Mendoza",
        "correo": "laura.mendoza@demo.com",
        "edad": 34,
        "ingresos": 6800000,
        "estado_civil": 1,
        "deudas_existentes": 4200000,
        "saldo_cuentas": 17500000,
    },
    {
        "nombre": "Carlos Ruiz",
        "correo": "carlos.ruiz@demo.com",
        "edad": 29,
        "ingresos": 4200000,
        "estado_civil": 0,
        "deudas_existentes": 6300000,
        "saldo_cuentas": 2200000,
    },
    {
        "nombre": "Paula Gomez",
        "correo": "paula.gomez@demo.com",
        "edad": 41,
        "ingresos": 9100000,
        "estado_civil": 1,
        "deudas_existentes": 3500000,
        "saldo_cuentas": 28000000,
    },
]

with open(BASE_DIR / "modelo" / "normalizacion.json") as f:
    stats = json.load(f)
media = np.array(stats["media"], dtype=np.float32)
desviacion = np.array(stats["desviacion"], dtype=np.float32)
decision_threshold = float(stats.get("decision_threshold", DEFAULT_THRESHOLD))
hidden_dims = tuple(stats.get("hidden_dims", list(MODEL_HIDDEN_DIMS)))
dropout = float(stats.get("dropout", MODEL_DROPOUT))

modelo_ml = RedNeuronalRiesgo(len(media), hidden_dims=hidden_dims, dropout=dropout)
modelo_ml.load_state_dict(torch.load(BASE_DIR / "modelo" / "modelo_riesgo.pth", weights_only=True))
modelo_ml.eval()


def predecir_riesgo(edad, ingresos, estado_civil,
                    deudas_existentes, saldo_cuentas, producto):
    factor_producto = obtener_factor_producto(producto)
    datos = construir_features(
        edad,
        ingresos,
        estado_civil,
        deudas_existentes,
        saldo_cuentas,
        factor_producto,
    )
    datos_norm = (datos - media) / (desviacion + 1e-8)
    tensor = torch.tensor(datos_norm, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        logits = modelo_ml(tensor)
        prob = torch.sigmoid(logits).item()
    return prob


# --- Modelos Pydantic ---

class BuscarCliente(BaseModel):
    nombre: str
    correo: str


class ClienteNuevo(BaseModel):
    nombre: str
    correo: str
    edad: int
    ingresos: float
    estado_civil: int        # 0 = soltero, 1 = casado
    deudas_existentes: float
    saldo_cuentas: float
    producto_id: int


class SolicitudExistente(BaseModel):
    nombre: str
    correo: str
    producto_id: int


class AdminLogin(BaseModel):
    username: str
    password: str


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def solicitud_snapshot(cliente: dict) -> dict:
    return {field: cliente[field] for field in SOLICITUD_FIELDS}


def serializar_solicitud(fila: sqlite3.Row) -> dict:
    solicitud = dict(fila)
    solicitud["credito_aprobado"] = bool(solicitud["credito_aprobado"])
    solicitud["archivada"] = bool(solicitud["archivada"])
    factor_producto = obtener_factor_producto(
        {
            "nombre": solicitud.get("producto_nombre"),
            "categoria": solicitud.get("producto_categoria"),
        }
    )
    solicitud["nivel_riesgo"] = clasificar_nivel_riesgo(
        float(solicitud["probabilidad_riesgo"]),
        decision_threshold,
    )
    solicitud["factores_riesgo"] = construir_alertas_riesgo(
        float(solicitud["edad"]),
        float(solicitud["ingresos"]),
        float(solicitud["deudas_existentes"]),
        float(solicitud["saldo_cuentas"]),
        factor_producto,
    )
    return solicitud


def asegurar_columna(conn: sqlite3.Connection, tabla: str, columna: str, definicion: str):
    columnas = {fila["name"] for fila in conn.execute(f"PRAGMA table_info({tabla})").fetchall()}
    if columna not in columnas:
        conn.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {definicion}")


def es_admin_autenticado(request: Request) -> bool:
    return request.cookies.get(ADMIN_COOKIE_NAME) == "1"


def exigir_admin(request: Request):
    if not es_admin_autenticado(request):
        raise HTTPException(status_code=401, detail="Debes iniciar sesion como administrador")


def redirigir_login_admin(request: Request):
    return RedirectResponse(url=str(request.url_for("admin_login_page")), status_code=303)


def crear_tablas(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL UNIQUE,
            categoria TEXT NOT NULL,
            descripcion TEXT NOT NULL,
            tasa TEXT NOT NULL,
            beneficio TEXT NOT NULL,
            icono TEXT NOT NULL,
            destacado INTEGER NOT NULL DEFAULT 0,
            solicitable INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT NOT NULL,
            correo TEXT NOT NULL UNIQUE COLLATE NOCASE,
            edad INTEGER NOT NULL,
            ingresos REAL NOT NULL,
            estado_civil INTEGER NOT NULL,
            deudas_existentes REAL NOT NULL,
            saldo_cuentas REAL NOT NULL,
            creado_en TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            actualizado_en TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS solicitudes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cliente_id INTEGER NOT NULL,
            producto_id INTEGER NOT NULL,
            nombre TEXT NOT NULL,
            correo TEXT NOT NULL,
            edad INTEGER NOT NULL,
            ingresos REAL NOT NULL,
            estado_civil INTEGER NOT NULL,
            deudas_existentes REAL NOT NULL,
            saldo_cuentas REAL NOT NULL,
            probabilidad_riesgo REAL NOT NULL,
            credito_aprobado INTEGER NOT NULL,
            creado_en TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            actualizada_en TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            archivada INTEGER NOT NULL DEFAULT 0,
            archivada_en TEXT,
            FOREIGN KEY(cliente_id) REFERENCES clientes(id),
            FOREIGN KEY(producto_id) REFERENCES productos(id)
        )
        """
    )

    asegurar_columna(conn, "solicitudes", "nombre", "TEXT")
    asegurar_columna(conn, "solicitudes", "correo", "TEXT")
    asegurar_columna(conn, "solicitudes", "edad", "INTEGER")
    asegurar_columna(conn, "solicitudes", "ingresos", "REAL")
    asegurar_columna(conn, "solicitudes", "estado_civil", "INTEGER")
    asegurar_columna(conn, "solicitudes", "deudas_existentes", "REAL")
    asegurar_columna(conn, "solicitudes", "saldo_cuentas", "REAL")
    asegurar_columna(conn, "solicitudes", "actualizada_en", "TEXT")
    asegurar_columna(conn, "solicitudes", "archivada", "INTEGER DEFAULT 0")
    asegurar_columna(conn, "solicitudes", "archivada_en", "TEXT")

    conn.execute(
        """
        UPDATE solicitudes
        SET
            nombre = COALESCE(nombre, (SELECT nombre FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            correo = COALESCE(correo, (SELECT correo FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            edad = COALESCE(edad, (SELECT edad FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            ingresos = COALESCE(ingresos, (SELECT ingresos FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            estado_civil = COALESCE(estado_civil, (SELECT estado_civil FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            deudas_existentes = COALESCE(deudas_existentes, (SELECT deudas_existentes FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            saldo_cuentas = COALESCE(saldo_cuentas, (SELECT saldo_cuentas FROM clientes WHERE clientes.id = solicitudes.cliente_id)),
            actualizada_en = COALESCE(actualizada_en, creado_en),
            archivada = COALESCE(archivada, 0)
        WHERE
            nombre IS NULL OR correo IS NULL OR edad IS NULL OR ingresos IS NULL OR
            estado_civil IS NULL OR deudas_existentes IS NULL OR saldo_cuentas IS NULL OR
            actualizada_en IS NULL OR archivada IS NULL
        """
    )


def sembrar_productos(conn: sqlite3.Connection):
    total = conn.execute("SELECT COUNT(*) AS total FROM productos").fetchone()["total"]
    if total:
        return

    conn.executemany(
        """
        INSERT INTO productos (nombre, categoria, descripcion, tasa, beneficio, icono, destacado, solicitable)
        VALUES (:nombre, :categoria, :descripcion, :tasa, :beneficio, :icono, :destacado, :solicitable)
        """,
        PRODUCTOS_SEMILLA,
    )


def sembrar_clientes_demo(conn: sqlite3.Connection):
    total = conn.execute("SELECT COUNT(*) AS total FROM clientes").fetchone()["total"]
    if total:
        return

    for cliente in CLIENTES_DEMO:
        cliente_id = guardar_cliente(conn, cliente)
        producto_id = seleccionar_producto_demo(conn, cliente["correo"])
        producto = dict(conn.execute("SELECT * FROM productos WHERE id = ?", (producto_id,)).fetchone())
        prob_riesgo = predecir_riesgo(
            cliente["edad"],
            cliente["ingresos"],
            cliente["estado_civil"],
            cliente["deudas_existentes"],
            cliente["saldo_cuentas"],
            producto,
        )
        guardar_solicitud(conn, cliente_id, producto_id, prob_riesgo, prob_riesgo < decision_threshold, cliente)


def seleccionar_producto_demo(conn: sqlite3.Connection, correo: str) -> int:
    productos = conn.execute(
        "SELECT id, categoria FROM productos WHERE solicitable = 1 ORDER BY id"
    ).fetchall()
    indice = sum(ord(letra) for letra in correo) % len(productos)
    return productos[indice]["id"]


def init_db():
    DATA_DIR.mkdir(exist_ok=True)
    with get_connection() as conn:
        crear_tablas(conn)
        sembrar_productos(conn)
        sembrar_clientes_demo(conn)
        conn.commit()


def guardar_cliente(conn: sqlite3.Connection, cliente: dict) -> int:
    conn.execute(
        """
        INSERT INTO clientes (
            nombre, correo, edad, ingresos, estado_civil, deudas_existentes, saldo_cuentas, actualizado_en
        )
        VALUES (:nombre, :correo, :edad, :ingresos, :estado_civil, :deudas_existentes, :saldo_cuentas, CURRENT_TIMESTAMP)
        ON CONFLICT(correo) DO UPDATE SET
            nombre = excluded.nombre,
            edad = excluded.edad,
            ingresos = excluded.ingresos,
            estado_civil = excluded.estado_civil,
            deudas_existentes = excluded.deudas_existentes,
            saldo_cuentas = excluded.saldo_cuentas,
            actualizado_en = CURRENT_TIMESTAMP
        """,
        cliente,
    )
    fila = conn.execute(
        "SELECT id FROM clientes WHERE correo = ? COLLATE NOCASE",
        (cliente["correo"],),
    ).fetchone()
    return fila["id"]


def guardar_solicitud(
    conn: sqlite3.Connection,
    cliente_id: int,
    producto_id: int,
    prob_riesgo: float,
    aprobado: bool,
    cliente: dict,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO solicitudes (
            cliente_id, producto_id, nombre, correo, edad, ingresos, estado_civil,
            deudas_existentes, saldo_cuentas, probabilidad_riesgo, credito_aprobado,
            actualizada_en
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            cliente_id,
            producto_id,
            cliente["nombre"],
            cliente["correo"],
            cliente["edad"],
            cliente["ingresos"],
            cliente["estado_civil"],
            cliente["deudas_existentes"],
            cliente["saldo_cuentas"],
            round(prob_riesgo, 4),
            int(aprobado),
        ),
    )
    return cursor.lastrowid


def buscar_cliente_por_correo(correo: str) -> Optional[dict]:
    with get_connection() as conn:
        fila = conn.execute(
            """
            SELECT id, nombre, correo, edad, ingresos, estado_civil, deudas_existentes, saldo_cuentas
            FROM clientes
            WHERE correo = ? COLLATE NOCASE
            """,
            (correo,),
        ).fetchone()
    return dict(fila) if fila else None


def obtener_producto(producto_id: int) -> Optional[dict]:
    with get_connection() as conn:
        fila = conn.execute(
            """
            SELECT id, nombre, categoria, descripcion, tasa, beneficio, icono, destacado, solicitable
            FROM productos
            WHERE id = ?
            """,
            (producto_id,),
        ).fetchone()
    return dict(fila) if fila else None


def listar_productos(destacados: Optional[bool] = None, solo_solicitables: bool = False) -> list[dict]:
    query = """
        SELECT id, nombre, categoria, descripcion, tasa, beneficio, icono, destacado, solicitable
        FROM productos
        WHERE 1 = 1
    """
    parametros: list[object] = []

    if destacados is not None:
        query += " AND destacado = ?"
        parametros.append(int(destacados))

    if solo_solicitables:
        query += " AND solicitable = 1"

    query += " ORDER BY destacado DESC, categoria, nombre"

    with get_connection() as conn:
        filas = conn.execute(query, parametros).fetchall()
    return [dict(fila) for fila in filas]


def obtener_fila_solicitud(conn: sqlite3.Connection, solicitud_id: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        SOLICITUD_SELECT + " WHERE s.id = ?",
        (solicitud_id,),
    ).fetchone()


def obtener_solicitud(solicitud_id: int, incluir_archivadas: bool = True) -> Optional[dict]:
    with get_connection() as conn:
        fila = obtener_fila_solicitud(conn, solicitud_id)

    if not fila:
        return None

    solicitud = serializar_solicitud(fila)
    if not incluir_archivadas and solicitud["archivada"]:
        return None
    return solicitud


def listar_solicitudes(incluir_archivadas: bool = False) -> list[dict]:
    filtro_archivo = "" if incluir_archivadas else " WHERE s.archivada = 0"
    with get_connection() as conn:
        filas = conn.execute(
            SOLICITUD_SELECT + filtro_archivo + " ORDER BY s.archivada ASC, s.id DESC"
        ).fetchall()
    return [serializar_solicitud(fila) for fila in filas]


def limpiar_cliente_huerfano(conn: sqlite3.Connection, cliente_id: Optional[int]):
    if not cliente_id:
        return

    restantes = conn.execute(
        "SELECT COUNT(*) AS total FROM solicitudes WHERE cliente_id = ?",
        (cliente_id,),
    ).fetchone()["total"]

    if restantes == 0:
        conn.execute("DELETE FROM clientes WHERE id = ?", (cliente_id,))


def archivar_o_eliminar_solicitud(solicitud_id: int) -> dict:
    with get_connection() as conn:
        fila = conn.execute(
            "SELECT cliente_id, archivada FROM solicitudes WHERE id = ?",
            (solicitud_id,),
        ).fetchone()

        if not fila:
            raise HTTPException(status_code=404, detail="Solicitud no encontrada")

        if not fila["archivada"]:
            conn.execute(
                """
                UPDATE solicitudes
                SET archivada = 1, archivada_en = CURRENT_TIMESTAMP, actualizada_en = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (solicitud_id,),
            )
            solicitud = serializar_solicitud(obtener_fila_solicitud(conn, solicitud_id))
            conn.commit()
            return {"accion": "archivada", "solicitud": solicitud}

        conn.execute("DELETE FROM solicitudes WHERE id = ?", (solicitud_id,))
        limpiar_cliente_huerfano(conn, fila["cliente_id"])

        conn.commit()
    return {"accion": "eliminada"}


def restaurar_solicitud(solicitud_id: int) -> dict:
    with get_connection() as conn:
        fila = conn.execute(
            "SELECT id FROM solicitudes WHERE id = ?",
            (solicitud_id,),
        ).fetchone()

        if not fila:
            raise HTTPException(status_code=404, detail="Solicitud no encontrada")

        conn.execute(
            """
            UPDATE solicitudes
            SET archivada = 0, archivada_en = NULL, actualizada_en = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (solicitud_id,),
        )
        solicitud = serializar_solicitud(obtener_fila_solicitud(conn, solicitud_id))
        conn.commit()
    return solicitud


def actualizar_solicitud(solicitud_id: int, cliente: dict, producto_id: int) -> dict:
    producto = obtener_producto(producto_id)
    if not producto or not producto["solicitable"]:
        raise HTTPException(status_code=404, detail="Producto no disponible para solicitud")

    solicitud_actual = obtener_solicitud(solicitud_id)
    if not solicitud_actual:
        raise HTTPException(status_code=404, detail="Solicitud no encontrada")

    prob_riesgo = predecir_riesgo(
        cliente["edad"],
        cliente["ingresos"],
        cliente["estado_civil"],
        cliente["deudas_existentes"],
        cliente["saldo_cuentas"],
        producto,
    )
    aprobado = prob_riesgo < decision_threshold

    with get_connection() as conn:
        nuevo_cliente_id = guardar_cliente(conn, cliente)
        conn.execute(
            """
            UPDATE solicitudes
            SET
                cliente_id = ?,
                producto_id = ?,
                nombre = ?,
                correo = ?,
                edad = ?,
                ingresos = ?,
                estado_civil = ?,
                deudas_existentes = ?,
                saldo_cuentas = ?,
                probabilidad_riesgo = ?,
                credito_aprobado = ?,
                actualizada_en = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                nuevo_cliente_id,
                producto_id,
                cliente["nombre"],
                cliente["correo"],
                cliente["edad"],
                cliente["ingresos"],
                cliente["estado_civil"],
                cliente["deudas_existentes"],
                cliente["saldo_cuentas"],
                round(prob_riesgo, 4),
                int(aprobado),
                solicitud_id,
            ),
        )
        if solicitud_actual["cliente_id"] != nuevo_cliente_id:
            limpiar_cliente_huerfano(conn, solicitud_actual["cliente_id"])
        solicitud = serializar_solicitud(obtener_fila_solicitud(conn, solicitud_id))
        conn.commit()

    return solicitud


def resumen_auditoria() -> dict:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS total FROM solicitudes").fetchone()["total"]
        archivadas = conn.execute(
            "SELECT COUNT(*) AS total FROM solicitudes WHERE archivada = 1"
        ).fetchone()["total"]

    activas = total - archivadas
    return {
        "total": total,
        "activas": activas,
        "archivadas": archivadas,
    }


def registrar_solicitud(cliente: dict, producto_id: int) -> dict:
    producto = obtener_producto(producto_id)
    if not producto or not producto["solicitable"]:
        raise HTTPException(status_code=404, detail="Producto no disponible para solicitud")

    prob_riesgo = predecir_riesgo(
        cliente["edad"],
        cliente["ingresos"],
        cliente["estado_civil"],
        cliente["deudas_existentes"],
        cliente["saldo_cuentas"],
        producto,
    )
    aprobado = prob_riesgo < decision_threshold

    with get_connection() as conn:
        cliente_id = guardar_cliente(conn, cliente)
        solicitud_id = guardar_solicitud(conn, cliente_id, producto_id, prob_riesgo, aprobado, cliente)
        fila = obtener_fila_solicitud(conn, solicitud_id)
        conn.commit()

    return serializar_solicitud(fila)


def construir_metricas(productos: list[dict]) -> list[dict]:
    categorias = {producto["categoria"] for producto in productos}
    digitales = sum(1 for producto in productos if producto["solicitable"])
    return [
        {"valor": f"{len(productos)}+", "texto": "productos activos"},
        {"valor": f"{len(categorias)}", "texto": "lineas financieras"},
        {"valor": f"{digitales}", "texto": "solicitudes 100% digitales"},
    ]


init_db()


# --- Rutas de paginas ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    productos = listar_productos()
    contexto = {
        "request": request,
        "productos": productos,
        "productos_destacados": [producto for producto in productos if producto["destacado"]],
        "metricas": construir_metricas(productos),
    }
    return templates.TemplateResponse("index.html", contexto)


@app.get("/credito", response_class=HTMLResponse)
async def credito(request: Request):
    return templates.TemplateResponse(
        "credito.html",
        {
            "request": request,
            "productos_credito": listar_productos(solo_solicitables=True),
        },
    )


@app.get("/admin/login", response_class=HTMLResponse, name="admin_login_page")
async def admin_login_page(request: Request):
    if es_admin_autenticado(request):
        return RedirectResponse(url=str(request.url_for("admin_dashboard")), status_code=303)

    return templates.TemplateResponse(
        "admin_login.html",
        {"request": request, "error": None},
    )


@app.get("/admin", response_class=HTMLResponse, name="admin_dashboard")
async def admin_dashboard(request: Request):
    if not es_admin_autenticado(request):
        return redirigir_login_admin(request)

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "productos_credito": listar_productos(solo_solicitables=True),
            "resumen": resumen_auditoria(),
        },
    )


@app.get("/admin/logout", name="admin_logout")
async def admin_logout(request: Request):
    response = RedirectResponse(url=str(request.url_for("admin_login_page")), status_code=303)
    response.delete_cookie(ADMIN_COOKIE_NAME, path="/")
    return response


# --- API ---

@app.post("/api/buscar-cliente")
async def buscar_cliente(datos: BuscarCliente):
    cliente = buscar_cliente_por_correo(datos.correo)
    if cliente:
        return {"encontrado": True, "cliente": cliente}
    return {"encontrado": False}


@app.post("/api/solicitar-credito")
async def solicitar_credito(cliente: ClienteNuevo):
    nuevo = registrar_solicitud(
        {
        "nombre": cliente.nombre,
        "correo": cliente.correo,
        "edad": cliente.edad,
        "ingresos": cliente.ingresos,
        "estado_civil": cliente.estado_civil,
        "deudas_existentes": cliente.deudas_existentes,
        "saldo_cuentas": cliente.saldo_cuentas,
        },
        cliente.producto_id,
    )
    return {"cliente": nuevo}


@app.post("/api/solicitar-credito-existente")
async def solicitar_credito_existente(datos: SolicitudExistente):
    cliente_existente = buscar_cliente_por_correo(datos.correo)

    if not cliente_existente:
        return {"error": "Cliente no encontrado"}

    nuevo = registrar_solicitud(cliente_existente, datos.producto_id)
    return {"cliente": nuevo}


@app.get("/api/clientes")
async def listar_clientes():
    return listar_solicitudes(incluir_archivadas=False)


@app.get("/api/productos")
async def api_productos():
    return listar_productos()


@app.delete("/api/clientes/{cliente_id}")
async def eliminar_cliente(cliente_id: int, request: Request):
    exigir_admin(request)
    return archivar_o_eliminar_solicitud(cliente_id)


@app.post("/admin/api/login", name="admin_login_api")
async def admin_login_api(datos: AdminLogin, request: Request):
    if datos.username != ADMIN_USERNAME or datos.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Credenciales invalidas")

    response = JSONResponse(
        {
            "ok": True,
            "redirect": str(request.url_for("admin_dashboard")),
        }
    )
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        "1",
        httponly=True,
        samesite="lax",
        path="/",
    )
    return response


@app.get("/admin/api/auditoria", name="admin_listar_auditoria")
async def admin_listar_auditoria(request: Request):
    exigir_admin(request)
    return listar_solicitudes(incluir_archivadas=True)


@app.get("/admin/api/auditoria/{solicitud_id}", name="admin_obtener_auditoria")
async def admin_obtener_auditoria(solicitud_id: int, request: Request):
    exigir_admin(request)
    solicitud = obtener_solicitud(solicitud_id, incluir_archivadas=True)
    if not solicitud:
        raise HTTPException(status_code=404, detail="Solicitud no encontrada")
    return solicitud


@app.post("/admin/api/auditoria", name="admin_crear_auditoria")
async def admin_crear_auditoria(request: Request, solicitud: ClienteNuevo):
    exigir_admin(request)
    datos_cliente = solicitud.model_dump(exclude={"producto_id"})
    return registrar_solicitud(datos_cliente, solicitud.producto_id)


@app.put("/admin/api/auditoria/{solicitud_id}", name="admin_actualizar_auditoria")
async def admin_actualizar_auditoria(solicitud_id: int, request: Request, solicitud: ClienteNuevo):
    exigir_admin(request)
    datos_cliente = solicitud.model_dump(exclude={"producto_id"})
    return actualizar_solicitud(solicitud_id, datos_cliente, solicitud.producto_id)


@app.post("/admin/api/auditoria/{solicitud_id}/restore", name="admin_restaurar_auditoria")
async def admin_restaurar_auditoria(solicitud_id: int, request: Request):
    exigir_admin(request)
    return restaurar_solicitud(solicitud_id)


@app.delete("/admin/api/auditoria/{solicitud_id}", name="admin_eliminar_auditoria")
async def admin_eliminar_auditoria(solicitud_id: int, request: Request):
    exigir_admin(request)
    return archivar_o_eliminar_solicitud(solicitud_id)

"""Dashboard del proyecto (Opción B: página HTML estática para GitHub Pages).

Se corre desde la raíz del repositorio:   python dashboard/generar_html.py
Crea docs/index.html (un solo archivo con HTML, CSS y JavaScript).

Cómo funciona: aquí en Python se calculan todos los resultados y cada gráfica se guarda como
un diccionario (trazas + diseño). En la página, Plotly.js dibuja esos diccionarios en el navegador.
Los controles (deslizadores, menús y botones) no calculan en vivo: cambian entre resultados
que ya se calcularon aquí, prendiendo y apagando trazas con "visible".
"""
import json
import warnings
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
AQUI = Path(__file__).parent
RESULTADOS = AQUI / "resultados"

# ---------------------------------------------------------------------------
# Datos del equipo (los guardó la última celda del notebook)
# ---------------------------------------------------------------------------
equipo = json.loads((RESULTADOS / "equipo.json").read_text(encoding="utf-8"))
SEMILLA = int(equipo["SEMILLA"])
EQUIPO = equipo["EQUIPO"]
OBJETIVO = equipo["OBJETIVO"]           # Carry
ACTIVOS = equipo["ACTIVOS"].split(",")  # los 12 sectores del portafolio
INTEGRANTES = ["Alvaro Vázquez López (0267889)", "Alvaro Fernandez Lozano (0273215)"]

# mismos datos y misma división que en el notebook: 75 % entrenamiento, 25 % prueba
FACTORES = ["Mkt-RF", "SMB", "HML", "RF"]
datos = pd.read_csv(AQUI.parent / "datos" / "industrias_ff30_mensual.csv", index_col="fecha")
SECTORES = [c for c in datos.columns if c not in FACTORES]
excesos = datos[SECTORES].sub(datos["RF"], axis=0)
corte = int(0.75 * len(datos))
ENTRENA, PRUEBA = datos.index[:corte], datos.index[corte:]

# colores fijos para que cada método se vea igual en todas las gráficas
COLOR = {"MCO": "#7f7f7f", "Ridge": "#e5736b", "Lasso": "#e8833a", "Elastic Net": "#2ca02c"}
AZUL, ROJO, VERDE, GRIS = "#e5736b", "#d62728", "#2ca02c", "#9e9e9e"


def mostrar(lista_visibles, total):
    """Lista de True/False: True solo en las trazas que se deben ver."""
    return [k in lista_visibles for k in range(total)]


# ===========================================================================
# PARTE 1. Regularización
# ===========================================================================
OTROS = [s for s in SECTORES if s != OBJETIVO]
X_tr, y_tr = excesos.loc[ENTRENA, OTROS], excesos.loc[ENTRENA, OBJETIVO]
X_te, y_te = excesos.loc[PRUEBA, OTROS], excesos.loc[PRUEBA, OBJETIVO]
CV = KFold(n_splits=5, shuffle=False)


def construir_modelos(n_alfas=40):
    # los mismos 4 modelos del notebook: cada uno estandariza y luego estima
    alfas_ridge, alfas_l1 = np.logspace(-2, 4, n_alfas), np.logspace(-3, 1, n_alfas)
    return {
        "MCO": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=alfas_ridge, cv=CV)),
        "Lasso": make_pipeline(StandardScaler(), LassoCV(alphas=alfas_l1, cv=CV, max_iter=20000)),
        "Elastic Net": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=alfas_l1,
                                                                    cv=CV, max_iter=20000)),
    }


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - pred) ** 2)))


# --- Figura 1.1: coeficientes de Ridge con un deslizador de lambda ---------
log_lambdas = np.arange(-1, 4.01, 0.5)
coef_mco = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)[-1].coef_
orden_1 = list(np.array(OTROS)[np.argsort(coef_mco)])  # sectores ordenados por su coeficiente de MCO
trazas, pasos = [], []
for k, log_l in enumerate(log_lambdas):
    modelo = make_pipeline(StandardScaler(), Ridge(alpha=10 ** log_l)).fit(X_tr, y_tr)
    coef = pd.Series(modelo[-1].coef_, index=OTROS)[orden_1]
    trazas.append(dict(type="bar", x=orden_1, y=coef.round(4).tolist(), marker_color=AZUL,
                       name=f"lambda = {10 ** log_l:g}", visible=bool(k == 4)))
    pasos.append(dict(method="restyle", label=f"{log_l:.1f}", args=[{"visible": mostrar([k], len(log_lambdas))}]))
fig_ridge = dict(data=trazas, layout=dict(
    xaxis=dict(title="Sector (regresora)", tickangle=-60),
    yaxis=dict(title="Coeficiente estandarizado (% por 1 desv. est.)", range=[-0.8, 1.8]), showlegend=False,
    sliders=[dict(active=4, steps=pasos, currentvalue=dict(prefix="log10(lambda) = "), pad=dict(t=90))],
    margin=dict(t=30, b=40)))

# --- Figura 1.2: RMSE dentro y fuera de muestra según los meses de entrenamiento ---
VENTANAS = [36, 60, 120, len(X_tr)]
tabla_ventanas = {}
for n in VENTANAS:
    fila = {}
    for nombre, modelo in construir_modelos().items():
        modelo.fit(X_tr.iloc[-n:], y_tr.iloc[-n:])  # solo los últimos n meses de entrenamiento
        fila[nombre] = (rmse(y_tr.iloc[-n:], modelo.predict(X_tr.iloc[-n:])), rmse(y_te, modelo.predict(X_te)))
    tabla_ventanas[n] = fila

METODOS = list(COLOR)
trazas, botones = [], []
for k, n in enumerate(VENTANAS):
    fila = tabla_ventanas[n]
    trazas.append(dict(type="bar", name="Entrenamiento (dentro de muestra)", x=METODOS, visible=bool(k == 3),
                       y=[round(fila[m][0], 3) for m in METODOS], marker_color="#f2b8b5",
                       text=[f"{fila[m][0]:.2f}" for m in METODOS], textposition="outside"))
    trazas.append(dict(type="bar", name="Prueba 2020-2026 (fuera de muestra)", x=METODOS, visible=bool(k == 3),
                       y=[round(fila[m][1], 3) for m in METODOS], marker_color="#7f1d1d",
                       text=[f"{fila[m][1]:.2f}" for m in METODOS], textposition="outside"))
    botones.append(dict(method="update", label=f"{n} meses",
                        args=[{"visible": mostrar([2 * k, 2 * k + 1], 2 * len(VENTANAS))},
                              {"title.text": f"Error dentro y fuera de muestra al replicar {OBJETIVO} "
                                             f"(entrenando con {n} meses)"}]))
fig_ventanas = dict(data=trazas, layout=dict(
    title=dict(text=botones[3]["args"][1]["title.text"]), barmode="group",
    xaxis=dict(title="Método"), yaxis=dict(title="RMSE (% mensual)", range=[0, 16]),
    updatemenus=[dict(type="buttons", direction="right", buttons=botones, active=3, x=0, y=1.02,
                      xanchor="left", yanchor="bottom")],
    legend=dict(orientation="h", y=-0.2), margin=dict(t=125)))

w36, w240 = tabla_ventanas[36], tabla_ventanas[len(X_tr)]
texto_ventanas = (
    f"dentro de muestra MCO siempre tiene el menor error, pero fuera de muestra no. Con 36 meses, su error de "
    f"prueba es de {w36['MCO'][1]:.2f} %, contra {w36['Ridge'][1]:.2f} % de Ridge. Con 240 meses, los cuatro "
    f"métodos predicen casi igual, por lo que la regularización ayuda sobre todo cuando hay pocos datos.")

# --- Figura 1.3: resultados del Monte Carlo ---------------------------------
mc = {"RMSE de prueba": pd.read_csv(RESULTADOS / "montecarlo_rmse.csv"),
      "Error al estimar beta": pd.read_csv(RESULTADOS / "montecarlo_beta.csv")}
escenarios = [f"{f.colinealidad.replace('rho', 'ρ')}<br>{f.verdad}<br>{f.datos}"
              for f in mc["RMSE de prueba"].itertuples()]
trazas, botones = [], []
for k, (medida, tabla) in enumerate(mc.items()):
    for metodo in METODOS:
        trazas.append(dict(type="bar", name=metodo, x=escenarios, y=tabla[metodo].round(3).tolist(),
                           marker_color=COLOR[metodo], visible=bool(k == 0)))
    eje = "RMSE de prueba (media geométrica; mínimo posible 1.5)" if k == 0 else "||b - beta|| (media geométrica)"
    botones.append(dict(method="update", label=medida,
                        args=[{"visible": mostrar(range(4 * k, 4 * k + 4), 8)},
                              {"yaxis.title.text": eje, "title.text": f"Monte Carlo: {medida} por escenario"}]))
fig_mc = dict(data=trazas, layout=dict(
    title=dict(text=botones[0]["args"][1]["title.text"]), barmode="group",
    xaxis=dict(title="Escenario (colinealidad / verdad / datos)"),
    yaxis=dict(title="RMSE de prueba (media geométrica; mínimo posible 1.5)", type="log"),
    updatemenus=[dict(type="buttons", direction="right", buttons=botones, active=0, x=0, y=1.02,
                      xanchor="left", yanchor="bottom")],
    legend=dict(orientation="h", y=-0.3), margin=dict(t=125, b=40)))

# --- Tabla MANOVA -----------------------------------------------------------
manova = pd.read_csv(RESULTADOS / "tabla_manova.csv")
eta_inter = manova.loc[manova["Efecto"] == "metodo x escenario", "eta² multiv."].dropna().iloc[0]
pillai_inter = manova.query("Efecto == 'metodo x escenario' and Estadístico == 'Pillai'").iloc[0]


# ===========================================================================
# PARTE 2. Multicolinealidad (datos de Longley)
# ===========================================================================
# Longley (1967): los mismos 16 años que trae statsmodels (sm.datasets.longley)
longley = pd.DataFrame(
    [[60323, 83.0, 234289, 2356, 1590, 107608, 1947], [61122, 88.5, 259426, 2325, 1456, 108632, 1948],
     [60171, 88.2, 258054, 3682, 1616, 109773, 1949], [61187, 89.5, 284599, 3351, 1650, 110929, 1950],
     [63221, 96.2, 328975, 2099, 3099, 112075, 1951], [63639, 98.1, 346999, 1932, 3594, 113270, 1952],
     [64989, 99.0, 365385, 1870, 3547, 115094, 1953], [63761, 100.0, 363112, 3578, 3350, 116219, 1954],
     [66019, 101.2, 397469, 2904, 3048, 117388, 1955], [67857, 104.6, 419180, 2822, 2857, 118734, 1956],
     [68169, 108.4, 442769, 2936, 2798, 120445, 1957], [66513, 110.8, 444546, 4681, 2637, 121950, 1958],
     [68655, 112.6, 482704, 3813, 2552, 123366, 1959], [69564, 114.2, 502601, 3931, 2514, 125368, 1960],
     [69331, 115.7, 518173, 4806, 2572, 127852, 1961], [70551, 116.9, 554894, 4007, 2827, 130081, 1962]],
    columns=["TOTEMP", "GNPDEFL", "GNP", "UNEMP", "ARMED", "POP", "YEAR"], dtype=float)
X_l = longley.drop(columns="TOTEMP")


def tabla_vif(X):
    # VIF de cada variable = 1 / (1 - R²), con R² de la regresión de esa variable contra las demás (con constante)
    valores = []
    for col in X.columns:
        y_j = X[col].to_numpy()
        Z = np.column_stack([np.ones(len(X)), X.drop(columns=col).to_numpy()])
        residuo = y_j - Z @ np.linalg.lstsq(Z, y_j, rcond=None)[0]
        r2 = 1 - (residuo @ residuo) / np.sum((y_j - y_j.mean()) ** 2)
        valores.append(1 / (1 - r2))
    return pd.Series(valores, index=X.columns)


X_transformada = pd.DataFrame({"PIB_real_pc": X_l["GNP"] / (X_l["GNPDEFL"] * X_l["POP"]),
                               "UNEMP": X_l["UNEMP"], "ARMED": X_l["ARMED"]})
opciones_vif = {"MCO completo": X_l, "Quitar GNP y POP": X_l.drop(columns=["GNP", "POP"]),
                "Transformar": X_transformada}
remedios = pd.read_csv(RESULTADOS / "remedios.csv", index_col=0)

trazas, botones = [], []
for k, (nombre, X_r) in enumerate(opciones_vif.items()):
    vif = tabla_vif(X_r)
    trazas.append(dict(type="bar", x=list(vif.index), y=vif.round(2).tolist(), name=nombre, visible=bool(k == 0),
                       marker_color=[ROJO if v > 10 else VERDE for v in vif], text=[f"{v:,.1f}" for v in vif],
                       textposition="outside"))
    botones.append(dict(method="restyle", label=nombre, args=[{"visible": mostrar([k], len(opciones_vif))}]))
fig_vif = dict(data=trazas + [dict(type="scatter", mode="lines", x=[-0.5, 5.5], y=[10, 10], showlegend=False,
                                   line=dict(color="black", dash="dash"), name="VIF = 10", hoverinfo="skip",
                                   xaxis="x2")],
               layout=dict(showlegend=False,
                           xaxis=dict(title="Variable"),
                           xaxis2=dict(overlaying="x", range=[-0.5, 5.5], visible=False),
                           yaxis=dict(title="VIF (sin unidades, escala log)", type="log", range=[0, 3.8]),
                           updatemenus=[dict(type="buttons", direction="right", buttons=botones, active=0,
                                             x=0, y=1.02, xanchor="left", yanchor="bottom")],
                           margin=dict(t=60)))
# la línea punteada en VIF = 10 siempre se ve: se agrega "True" al final de cada lista de visibles
for b in botones:
    b["args"][0]["visible"] = b["args"][0]["visible"] + [True]

fig_loo = dict(data=[dict(type="bar", x=list(remedios.index), y=remedios["RMSE LOO"].round(1).tolist(),
                          marker_color=[VERDE if r == remedios["RMSE LOO"].min() else GRIS for r in remedios["RMSE LOO"]],
                          text=[f"{v:,.0f}" for v in remedios["RMSE LOO"]], textposition="outside")],
               layout=dict(title=dict(text="Error de validación leave-one-out de cada remedio (Longley)"),
                           xaxis=dict(title="Remedio"), yaxis=dict(title="RMSE LOO (miles de empleos)", range=[0, 1250]),
                           showlegend=False))


# ===========================================================================
# PARTE 3. Componentes principales
# ===========================================================================
# matriz Sigma del equipo (igual que en el notebook)
rng_pca = np.random.default_rng(SEMILLA + 3)
Q, _ = np.linalg.qr(rng_pca.normal(size=(5, 5)))
LAMBDAS_VERDADERAS = np.array([8.0, 3.0, 1.5, 0.5, 0.2])
Sigma = Q @ np.diag(LAMBDAS_VERDADERAS) @ Q.T
valores, vecs = np.linalg.eigh(Sigma)
a1 = vecs[:, np.argmax(valores)]  # primer vector propio "verdadero"


def angulo(u, v):
    return np.degrees(np.arccos(np.clip(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1)))


# método de la potencia, guardando el vector de cada iteración
rng = np.random.default_rng(SEMILLA)
v = rng.normal(size=5)
v = v / np.linalg.norm(v)
historia = [v]
for _ in range(40):
    v = Sigma @ v
    v = v / np.linalg.norm(v)
    historia.append(v)

ITER = 16  # iteraciones que se muestran en el deslizador
coords = [f"x{j + 1}" for j in range(5)]
trazas = [dict(type="bar", x=coords, y=a1.round(4).tolist(), name="a₁ (eigh, el verdadero)", marker_color=GRIS)]
pasos = []
for t in range(ITER):
    v_t = historia[t] * np.sign(historia[t] @ a1)  # mismo signo que a₁ (el signo de un vector propio es arbitrario)
    trazas.append(dict(type="bar", x=coords, y=v_t.round(4).tolist(), name="v<sub>t</sub> (método de la potencia)",
                       marker_color=AZUL, visible=bool(t == 0)))
    pasos.append(dict(method="restyle", label=str(t), args=[{"visible": [True] + mostrar([t], ITER)}]))
fig_potencia = dict(data=trazas, layout=dict(
    barmode="group",
    xaxis=dict(title="Coordenada del vector"), yaxis=dict(title="Valor de la coordenada (vector de norma 1)",
                                                          range=[-1, 1]),
    sliders=[dict(active=0, steps=pasos, currentvalue=dict(prefix="Iteración t = "), pad=dict(t=50))],
    legend=dict(x=0.99, xanchor="right", y=0.99, bgcolor="rgba(255,255,255,0.8)"), margin=dict(t=30)))

errores = [max(angulo(h, a1), 1e-7) for h in historia]
razon = np.sort(valores)[-2] / np.sort(valores)[-1]
teoria = [errores[3] * razon ** (t - 3) for t in range(len(errores))]
fig_error = dict(data=[
    dict(type="scatter", mode="lines+markers", x=list(range(len(errores))), y=errores, name="Ángulo observado",
         marker=dict(color=AZUL, size=5)),
    dict(type="scatter", mode="lines", x=list(range(len(errores))), y=teoria, name=f"Teoría: (λ₂/λ₁)^t = {razon:.3f}^t",
         line=dict(color=ROJO, dash="dash"))],
    layout=dict(title=dict(text="Convergencia del método de la potencia"), xaxis=dict(title="Iteración t"),
                yaxis=dict(title="Ángulo con a₁ (grados, escala log)", type="log", range=[-7, 2.2]),
                legend=dict(x=1, xanchor="right", y=1)))

# PCA de los 30 sectores (matriz de correlación del entrenamiento)
Z = (excesos.loc[ENTRENA] - excesos.loc[ENTRENA].mean()) / excesos.loc[ENTRENA].std()
R = Z.corr().to_numpy()
val_R, vec_R = np.linalg.eigh(R)
orden_R = np.argsort(val_R)[::-1]
val_R, vec_R = val_R[orden_R], vec_R[:, orden_R]
var_explicada = val_R / len(SECTORES)
pc1 = Z.to_numpy() @ vec_R[:, 0]
corr_mercado = abs(np.corrcoef(pc1, datos.loc[ENTRENA, "Mkt-RF"])[0, 1])

trazas, botones = [], []
for k in range(3):
    carga = vec_R[:, k] * (np.sign(vec_R[:, k].sum()) if k == 0 else np.sign(vec_R[np.argmax(abs(vec_R[:, k])), k]))
    carga = pd.Series(carga, index=SECTORES).sort_values()
    trazas.append(dict(type="bar", orientation="h", x=carga.round(4).tolist(), y=list(carga.index), visible=bool(k == 0),
                       marker_color=[AZUL if c > 0 else ROJO for c in carga], name=f"Componente {k + 1}"))
    botones.append(dict(method="update", label=f"Componente {k + 1}",
                        args=[{"visible": mostrar([k], 3)},
                              {"title.text": f"Cargas del componente {k + 1}: explica {100 * var_explicada[k]:.1f} % "
                                             f"de la varianza de los 30 sectores",
                               "yaxis.categoryarray": list(carga.index)}]))
fig_cargas = dict(data=trazas, layout=dict(
    title=dict(text=botones[0]["args"][1]["title.text"]), showlegend=False,
    xaxis=dict(title="Carga (peso del sector en el componente, vector de norma 1)"),
    yaxis=dict(title="Sector", categoryorder="array", categoryarray=trazas[0]["y"], tickfont=dict(size=10)),
    updatemenus=[dict(type="buttons", direction="right", buttons=botones, active=0, x=0, y=1.02,
                      xanchor="left", yanchor="bottom")], margin=dict(t=125)))


# ===========================================================================
# PARTE 4. Markowitz
# ===========================================================================
rend = datos.loc[ENTRENA, ACTIVOS]
mu_a = rend.mean().to_numpy() * 12
Sigma_a = rend.cov().to_numpy() * 12
rf_a = datos.loc[ENTRENA, "RF"].mean() * 12
uno = np.ones(len(ACTIVOS))


def constantes(mu_v, S):
    S_inv_mu, S_inv_1 = np.linalg.solve(S, mu_v), np.linalg.solve(S, np.ones(len(mu_v)))
    A, B, C = np.ones(len(mu_v)) @ S_inv_mu, mu_v @ S_inv_mu, np.ones(len(mu_v)) @ S_inv_1
    return A, B, C, B * C - A ** 2


def pesos_gmv(S):
    S_inv_1 = np.linalg.solve(S, np.ones(S.shape[0]))
    return S_inv_1 / S_inv_1.sum()


def pesos_tangente(mu_v, S, rf):
    A, B, C, D = constantes(mu_v, S)
    return np.linalg.solve(S, mu_v - rf * np.ones(len(mu_v))) / (A - C * rf)


def tangente_sin_corto(mu_v, S, rf):
    n = len(mu_v)
    return minimize(lambda w: -(w @ mu_v - rf) / np.sqrt(w @ S @ w), np.ones(n) / n, method="SLSQP",
                    bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                    options={"maxiter": 1000}).x


def frontera_sin_corto(m, mu_v, S):
    n = len(mu_v)
    restricciones = [{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "eq", "fun": lambda w: w @ mu_v - m}]
    return minimize(lambda w: 0.5 * w @ S @ w, np.ones(n) / n, method="SLSQP", constraints=restricciones,
                    bounds=[(0, 1)] * n, options={"ftol": 1e-12, "maxiter": 2000}).x


def covarianza_pca(S, k=3):
    valores_s, vecs_s = np.linalg.eigh(S)
    o = np.argsort(valores_s)[::-1][:k]
    S_k = vecs_s[:, o] @ np.diag(valores_s[o]) @ vecs_s[:, o].T
    return S_k + np.diag(np.diag(S - S_k))


A, B, C, D = constantes(mu_a, Sigma_a)
w_g = pesos_gmv(Sigma_a)
m_g, s_g = w_g @ mu_a, np.sqrt(w_g @ Sigma_a @ w_g)

# --- Figura 4.1: frontera eficiente con un deslizador de la tasa libre de riesgo ---
malla = np.linspace(A / C - 12, mu_a.max() + 15, 200)
frontera_sigma = np.sqrt((C * malla ** 2 - 2 * A * malla + B) / D)
malla_sc = np.linspace(mu_a.min() + 1e-3, mu_a.max() - 1e-3, 30)
sigma_sc = [np.sqrt(w @ Sigma_a @ w) for w in (frontera_sin_corto(m, mu_a, Sigma_a) for m in malla_sc)]
vol_activos = np.sqrt(np.diag(Sigma_a))

fijas = [
    dict(type="scatter", mode="lines", x=frontera_sigma.round(3).tolist(), y=malla.round(3).tolist(),
         name="Frontera con ventas en corto", line=dict(color=AZUL)),
    dict(type="scatter", mode="lines", x=np.round(sigma_sc, 3).tolist(), y=malla_sc.round(3).tolist(),
         name="Frontera sin ventas en corto", line=dict(color="#e8833a")),
    dict(type="scatter", mode="markers+text", x=vol_activos.round(3).tolist(), y=mu_a.round(3).tolist(), text=ACTIVOS,
         textposition="top right", textfont=dict(size=9), name="Sectores", marker=dict(color=GRIS, size=7)),
    dict(type="scatter", mode="markers", x=[round(s_g, 3)], y=[round(m_g, 3)], name="GMV (mínima varianza)",
         marker=dict(color=VERDE, size=11, symbol="diamond")),
]
tasas = sorted({0.0, 1.0, round(rf_a, 2), 2.0, 3.0, 4.0, 5.0})
trazas, pasos = list(fijas), []
x_max = 40
for k, rf in enumerate(tasas):
    w_t = pesos_tangente(mu_a, Sigma_a, rf)
    m_t, s_t = w_t @ mu_a, np.sqrt(w_t @ Sigma_a @ w_t)
    sharpe = (m_t - rf) / s_t
    inicial = bool(abs(rf - rf_a) < 1e-9)
    trazas.append(dict(type="scatter", mode="lines", x=[0, x_max], y=[rf, rf + sharpe * x_max], visible=inicial,
                       name="Línea del mercado de capitales", line=dict(color=ROJO, dash="dash", width=1.5)))
    trazas.append(dict(type="scatter", mode="markers", x=[round(s_t, 3)], y=[round(m_t, 3)], visible=inicial,
                       name="Tangente (máximo Sharpe)", marker=dict(color=ROJO, size=16, symbol="star")))
    etiqueta = f"{rf:.2f}" + (" (real)" if abs(rf - rf_a) < 1e-9 else "")
    pasos.append(dict(method="restyle", label=etiqueta,
                      args=[{"visible": [True] * len(fijas) + mostrar([2 * k, 2 * k + 1], 2 * len(tasas))}]))
activo_rf = tasas.index(round(rf_a, 2))
fig_frontera = dict(data=trazas, layout=dict(
    xaxis=dict(title="Volatilidad anual (%)", range=[0, x_max]), yaxis=dict(title="Rendimiento anual (%)", range=[-5, 45]),
    sliders=[dict(active=activo_rf, steps=pasos, currentvalue=dict(visible=False),
                  pad=dict(t=50))],
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"), margin=dict(t=30)))

# --- Figura 4.2: valor de $1 invertido en la prueba, pesos fijos vs. reestimación anual ---
Sigma_lw = LedoitWolf().fit(rend).covariance_ * 12
estrategias = {
    "1/N": uno / len(uno),
    "GMV muestral": w_g,
    "Tangente muestral": pesos_tangente(mu_a, Sigma_a, rf_a),
    "GMV Ledoit-Wolf": pesos_gmv(Sigma_lw),
    "GMV PCA": pesos_gmv(covarianza_pca(Sigma_a)),
    "Tangente sin corto": tangente_sin_corto(mu_a, Sigma_a, rf_a),
}
reglas = {
    "1/N": lambda m, S, r, h: np.ones(len(m)) / len(m),
    "GMV muestral": lambda m, S, r, h: pesos_gmv(S),
    "GMV Ledoit-Wolf": lambda m, S, r, h: pesos_gmv(LedoitWolf().fit(h).covariance_ * 12),
    "Tangente muestral": lambda m, S, r, h: pesos_tangente(m, S, r),
    "Tangente sin corto": lambda m, S, r, h: tangente_sin_corto(m, S, r),
}


def prueba_movil(regla, ventana=60):
    # cada enero se estima con los 60 meses anteriores y esos pesos se quedan 12 meses
    salida = []
    for inicio in range(corte, len(datos), 12):
        historia_m = datos.iloc[inicio - ventana:inicio]
        mu, S = historia_m[ACTIVOS].mean().to_numpy() * 12, historia_m[ACTIVOS].cov().to_numpy() * 12
        w = regla(mu, S, historia_m["RF"].mean() * 12, historia_m[ACTIVOS])
        salida.extend(datos.iloc[inicio:inicio + 12][ACTIVOS].to_numpy() @ w)
    return np.array(salida)


rf_prueba = datos.loc[PRUEBA, "RF"].to_numpy()


def sharpe_prueba(r):
    exceso = r - rf_prueba
    return exceso.mean() / exceso.std(ddof=1) * np.sqrt(12)


series = {"Pesos fijos": {n: datos.loc[PRUEBA, ACTIVOS].to_numpy() @ w for n, w in estrategias.items()},
          "Reestimación anual": {n: prueba_movil(r) for n, r in reglas.items()}}
COLOR_E = {"1/N": "#222222", "GMV muestral": VERDE, "Tangente muestral": ROJO, "GMV Ledoit-Wolf": AZUL,
           "GMV PCA": "#9467bd", "Tangente sin corto": "#e8833a"}
fechas = [f"{f}-01" for f in PRUEBA]
trazas, botones, total = [], [], sum(len(s) for s in series.values())
for modo, dic in series.items():
    indices = []
    for nombre, r in dic.items():
        valor = np.cumprod(1 + np.maximum(r, -100) / 100)
        indices.append(len(trazas))
        trazas.append(dict(type="scatter", mode="lines", x=fechas, y=valor.round(4).tolist(), visible=(modo == "Pesos fijos"),
                           name=f"{nombre} (Sharpe {sharpe_prueba(r):.2f})", line=dict(color=COLOR_E[nombre], width=2)))
    botones.append(dict(method="update", label=modo,
                        args=[{"visible": mostrar(indices, total)},
                              {"title.text": f"Valor de 1 peso invertido en enero de 2020 ({modo.lower()})"}]))
fig_acumulado = dict(data=trazas, layout=dict(
    title=dict(text=botones[0]["args"][1]["title.text"]),
    xaxis=dict(title="Mes (periodo de prueba)"), yaxis=dict(title="Valor del portafolio (pesos, escala log)", type="log"),
    updatemenus=[dict(type="buttons", direction="right", buttons=botones, active=0, x=0, y=1.02,
                      xanchor="left", yanchor="bottom")],
    legend=dict(orientation="h", y=-0.25), margin=dict(t=125)))

# --- Figura 4.3: Sharpe dentro vs. fuera de muestra ---------------------------
prueba = pd.read_csv(RESULTADOS / "markowitz_prueba.csv", index_col=0)
movil = pd.read_csv(RESULTADOS / "markowitz_movil.csv", index_col=0)
fig_sharpe = dict(data=[
    dict(type="bar", name="Entrenamiento 2000-2019 (dentro de muestra)", x=list(prueba.index),
         y=prueba["Sharpe entrenamiento (optimista)"].round(3).tolist(), marker_color="#f2b8b5",
         text=[f"{v:.2f}" for v in prueba["Sharpe entrenamiento (optimista)"]], textposition="outside"),
    dict(type="bar", name="Prueba 2020-2026 (fuera de muestra)", x=list(prueba.index),
         y=prueba["Sharpe prueba"].round(3).tolist(), marker_color="#7f1d1d",
         text=[f"{v:.2f}" for v in prueba["Sharpe prueba"]], textposition="outside")],
    layout=dict(title=dict(text="Sharpe anual de cada estrategia: dentro vs. fuera de muestra (pesos fijos)"),
                barmode="group", xaxis=dict(title="Estrategia"), yaxis=dict(title="Razón de Sharpe anual", range=[-0.5, 1.4]),
                legend=dict(orientation="h", y=-0.2)))

# revisión: lo que se calcula aquí debe coincidir con lo que guardó el notebook
for nombre, r in series["Pesos fijos"].items():
    assert abs(sharpe_prueba(r) - prueba.loc[nombre, "Sharpe prueba"]) < 1e-6, nombre
for nombre, r in series["Reestimación anual"].items():
    assert abs(sharpe_prueba(r) - movil.loc[nombre, "Sharpe"]) < 1e-6, nombre
assert abs(tabla_vif(X_l)["GNP"] - remedios.loc["MCO completo", "VIF máximo"]) < 1e-3


# ===========================================================================
# Armar la página
# ===========================================================================
def tabla_html(df, formatos, resaltar=None):
    """Convierte un DataFrame en una tabla HTML con el formato de cada columna."""
    encabezado = "".join(f"<th>{escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    filas = []
    for idx, fila in df.iterrows():
        celdas = [f"<td class='izq'>{escape(str(idx))}</td>"]
        for col in df.columns:
            valor = fila[col]
            if isinstance(valor, str):
                celdas.append(f"<td class='izq'>{escape(valor)}</td>")
            else:
                celdas.append("<td></td>" if pd.isna(valor) else f"<td>{formatos.get(col, '{:,.3f}').format(valor)}</td>")
        clase = " class='resalta'" if resaltar is not None and resaltar(idx, fila) else ""
        filas.append(f"<tr{clase}>{''.join(celdas)}</tr>")
    cuerpo = "\n".join("      " + f for f in filas)
    return (f"<div class='tabla'>\n  <table>\n    <thead><tr>{encabezado}</tr></thead>\n    <tbody>\n{cuerpo}\n"
            f"    </tbody>\n  </table>\n</div>")


graficas = []
# alto de las gráficas que necesitan más espacio (las demás miden 500 px)
fig_ridge["layout"]["height"] = 580
fig_mc["layout"]["height"] = 540
fig_cargas["layout"]["height"] = 760
fig_frontera["layout"]["height"] = 620


def grafica(fig, conclusion):
    """Div de la gráfica + frase con lo que hay que concluir de ella."""
    fig["layout"].setdefault("height", 500)
    # el título va arriba a la izquierda, para que no lo tapen los botones
    if "title" in fig["layout"]:
        fig["layout"]["title"].update(x=0.01, xanchor="left", y=0.985, yanchor="top")
    graficas.append(fig)
    return (f"<div class='grafica' id='g{len(graficas) - 1}'></div>\n"
            f"<p class='conclusion'><b>Qué concluir:</b> {conclusion}</p>")


manova_vista = manova.set_index(["Efecto", "Estadístico"])
manova_vista.index = [f"{e} · {s}" for e, s in manova_vista.index]
manova_vista.index.name = "Efecto · estadístico"
formato_manova = {"Valor": "{:.4f}", "gl num.": "{:.0f}", "gl den.": "{:.1f}", "F": "{:.2f}", "p": "{:.2e}",
                  "eta² multiv.": "{:.3f}"}
formato_mark = {c: "{:.2f}" for c in prueba.columns} | {"Sharpe prueba": "{:.3f}", "Sharpe entrenamiento (optimista)": "{:.3f}",
                                                      "Mayor peso abs.": "{:.3f}", "Suma pesos cortos": "{:.3f}"}
mejor_lasso = (w240["MCO"][1] - w240["Lasso"][1]) / w240["MCO"][1]
caida_tangente = 1 - prueba.loc["Tangente muestral", "Sharpe prueba"] / prueba.loc["Tangente muestral",
                                                                                   "Sharpe entrenamiento (optimista)"]

secciones = f"""
<section id="p1">
<h2>Parte 1. Regularización</h2>
<p>Usamos los rendimientos de los otros 29 sectores para replicar el sector objetivo. Comparamos MCO con 3 métodos
que castigan coeficientes grandes: Ridge, Lasso y Elastic Net.</p>
<h3>1.1 ¿Qué hace el castigo lambda?</h3>
{grafica(fig_ridge, "al aumentar lambda, los coeficientes se hacen más pequeños y se acercan a cero. Un castigo "
         "moderado mejora un poco la predicción fuera de muestra, pero un castigo excesivo la empeora, porque el "
         "modelo deja de usar información útil.")}
<h3>1.2 Dentro vs. fuera de muestra según los datos disponibles</h3>
<p class="instruccion">Elige cuántos meses de historia se usan para entrenar.</p>
{grafica(fig_ventanas, texto_ventanas)}
<h3>1.3 Monte Carlo: ¿cuándo conviene regularizar?</h3>
<p class="instruccion">8 escenarios × 4 métodos × 20 repeticiones; el eje está en escala log.</p>
{grafica(fig_mc, "con pocos datos (40 observaciones y 30 variables) los tres métodos regularizados reducen mucho los "
         "dos errores frente a MCO. Con muchos datos casi no ayudan a predecir, pero con alta colinealidad sí ayudan "
         "a estimar beta.")}
<h3>1.4 Tabla MANOVA</h3>
<p>El renglón resaltado es el estadístico principal (traza de Pillai), porque es el más robusto cuando no se
cumple la igualdad de covarianzas y el diseño está balanceado.</p>
{tabla_html(manova_vista, formato_manova, resaltar=lambda i, f: "Pillai" in i)}
<div class="caja"><b>Conclusión de la Parte 1.</b> La interacción método × escenario es significativa
(Pillai = {pillai_inter['Valor']:.3f}, p &lt; 0.001) y tiene un efecto grande (η² = {eta_inter:.3f}), es decir, el
mejor método depende del caso. Con muchos datos y baja colinealidad, MCO es suficiente; con alta colinealidad o pocos
datos, conviene regularizar. Para replicar {escape(OBJETIVO)}, el método más adecuado es Lasso: usa solo 12 de los 29
sectores y tiene un error de prueba {100 * mejor_lasso:.1f} % menor que MCO.</div>
</section>

<section id="p2">
<h2>Parte 2. Multicolinealidad</h2>
<p>Longley explica el empleo total de EE. UU. con 6 variables que crecen juntas en el tiempo. Un VIF mayor que 10
(línea punteada) indica multicolinealidad grave.</p>
{grafica(fig_vif, f"con todas las variables el VIF de GNP llega a {remedios.loc['MCO completo', 'VIF máximo']:,.1f}. "
         f"Quitar GNP y POP lo baja a {remedios.loc['Quitar GNP y POP', 'VIF máximo']:.1f} y transformar a PIB real "
         f"per cápita lo baja a {remedios.loc['Transformar', 'VIF máximo']:.1f}: solo transformar elimina el problema.")}
{grafica(fig_loo, f"para predecir gana Quitar GNP y POP (error LOO de {remedios.loc['Quitar GNP y POP', 'RMSE LOO']:.1f}, "
         f"6 % menos que MCO). Transformar predice peor ({remedios.loc['Transformar', 'RMSE LOO']:.1f}), pero es el "
         f"mejor para interpretar cada coeficiente. PCR y PLS son los peores aquí.")}
{tabla_html(remedios, {"RMSE LOO": "{:,.1f}", "VIF máximo": "{:,.1f}", "Condición de X": "{:,.1f}"})}
<p class="nota">PCR, PLS y Ridge no tienen VIF porque no estiman coeficientes de MCO sobre las variables originales.</p>
</section>

<section id="p3">
<h2>Parte 3. Componentes principales</h2>
<h3>3.1 Método de la potencia con la matriz Σ del equipo (λ = 8, 3, 1.5, 0.5, 0.2)</h3>
{grafica(fig_potencia, f"en pocas iteraciones, v<sub>t</sub> coincide con a₁. El ángulo entre ambos se reduce en "
         f"cada paso según la razón λ₂/λ₁ = {razon:.3f}, por lo que el método converge rápido.")}
{grafica(fig_error, "la línea observada sigue a la teórica: el error cae de forma geométrica hasta llegar a la "
         "precisión de la computadora.")}
<h3>3.2 PCA de los 30 sectores</h3>
<p class="instruccion">Elige el componente para ver qué sectores pesan más en él.</p>
{grafica(fig_cargas, f"en el primer componente todos los sectores tienen carga del mismo signo: es el factor de mercado. "
         f"Explica {100 * var_explicada[0]:.1f} % de la varianza y su correlación con el mercado es {corr_mercado:.3f}. "
         f"Ese riesgo común no se elimina diversificando entre sectores.")}
</section>

<section id="p4">
<h2>Parte 4. Markowitz</h2>
<p>Activos: {escape(", ".join(ACTIVOS))}.</p>
<h3>4.1 Frontera eficiente (tasa libre de riesgo)</h3>
<p class="instruccion">Mueve r<sub>f</sub>: la línea del mercado de capitales gira y el portafolio tangente se desplaza por la frontera.</p>
{grafica(fig_frontera, f"la GMV tiene una volatilidad de {s_g:.1f} %, menor que la de cualquier sector. Con la r<sub>f</sub> real "
         f"({rf_a:.2f} %) la tangente promete un Sharpe de {prueba.loc['Tangente muestral', 'Sharpe entrenamiento (optimista)']:.2f}, "
         f"pero solo con ventas en corto grandes. Al acercar r<sub>f</sub> a A/C = {A / C:.2f} % la tangente se va cada vez más "
         f"arriba y más apalancada.")}
<h3>4.2 ¿Cómo les fue fuera de muestra?</h3>
{grafica(fig_acumulado, f"con pesos fijos el portafolio 1/N tiene el mejor Sharpe ({prueba.loc['1/N', 'Sharpe prueba']:.3f}). "
         f"Al reestimar cada año la tangente muestral sube a {movil.loc['Tangente muestral', 'Sharpe']:.2f}, pero con "
         f"{movil.loc['Tangente muestral', 'Volatilidad anual (%)']:.0f} % de volatilidad y una caída máxima de "
         f"{movil.loc['Tangente muestral', 'Máx. caída (%)']:.0f} %: un riesgo que ningún fondo aceptaría.")}
<h3>4.3 Dentro vs. fuera de muestra</h3>
{grafica(fig_sharpe, f"todas las estrategias optimizadas prometen más de lo que dan. La tangente muestral pasa de un "
         f"Sharpe de {prueba.loc['Tangente muestral', 'Sharpe entrenamiento (optimista)']:.2f} a "
         f"{prueba.loc['Tangente muestral', 'Sharpe prueba']:.2f} ({100 * caida_tangente:.1f} % menos), porque depende de "
         f"rendimientos esperados muy difíciles de estimar. Solo 1/N, que no estima nada, mejora fuera de muestra.")}
<h4>Pesos fijos (prueba 2020-2026)</h4>
{tabla_html(prueba, formato_mark, resaltar=lambda i, f: i == "1/N")}
<h4>Reestimación anual con ventana de 60 meses (prueba 2020-2026)</h4>
{tabla_html(movil, {c: "{:.2f}" for c in movil.columns} | {"Sharpe": "{:.3f}"})}
</section>

<section id="conclusion">
<h2>Conclusión</h2>
<div class="caja">
<p><b>Método para replicar {escape(OBJETIVO)}.</b> Lasso es el método más adecuado. Con 240 meses replica
{escape(OBJETIVO)} usando solo 12 de los 29 sectores, con un error de prueba {100 * mejor_lasso:.1f} % menor que MCO:
casi la misma precisión con menos posiciones. Su ventaja es mayor cuando hay poca historia: con 36 meses, Ridge y Lasso
reducen el error de prueba más de 60 %.</p>
<p><b>Construcción del portafolio.</b> El portafolio de pesos iguales (1/N) es la mejor base. Los sectores se mueven
principalmente por el mercado ({100 * var_explicada[0]:.0f} % de su variación), y ese riesgo no se elimina al
diversificar. La tangente de Markowitz prometía un Sharpe de
{prueba.loc['Tangente muestral', 'Sharpe entrenamiento (optimista)']:.2f}, pero en 2020-2026 solo logró
{prueba.loc['Tangente muestral', 'Sharpe prueba']:.2f}, mientras que 1/N logró {prueba.loc['1/N', 'Sharpe prueba']:.2f}.
Si se decide optimizar, conviene hacerlo sin ventas en corto.</p>
<p><b>Limitación.</b> Solo se evaluó un periodo de prueba de {len(PRUEBA)} meses, que incluye la pandemia, y sin costos
de transacción, por lo que las diferencias entre estrategias no son estadísticamente concluyentes.</p>
</div>
</section>
"""

integrantes_html = " · ".join(escape(n) for n in INTEGRANTES)

# Todo va en un solo archivo, docs/index.html, con tres partes:
#   <style>  -> CSS: colores, tipos de letra y acomodo
#   <body>   -> HTML: estructura y textos de la página
#   <script> -> JavaScript: datos de cada gráfica y el código que las dibuja con Plotly.js
CSS = """/* Estilos del dashboard */

/* colores que se usan en toda la página */
:root {
  --azul: #b91c1c;
  --azul-oscuro: #7f1d1d;
  --azul-claro: #fdeeee;
  --fondo: #faf6f6;
  --borde: #eadede;
  --texto: #1d2330;
  --suave: #5b6475;
  --sombra: 0 1px 2px rgba(16, 24, 40, 0.05), 0 4px 16px rgba(16, 24, 40, 0.06);
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
  scroll-padding-top: 64px;
}

body {
  font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0;
  color: var(--texto);
  background: var(--fondo);
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

/* encabezado con degradado, título y equipo */
header {
  background: linear-gradient(135deg, var(--azul-oscuro) 0%, var(--azul) 60%, #dc2626 100%);
  color: #fff;
  padding: 48px 20px 40px;
}

header .dentro,
main {
  max-width: 1080px;
  margin: 0 auto;
}

header h1 {
  margin: 0 0 14px;
  font-size: 2.1rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

header p {
  margin: 6px 0;
  opacity: 0.9;
  font-size: 0.98rem;
}

header p:first-of-type {
  opacity: 1;
  font-size: 1.05rem;
}

/* menú que se queda pegado arriba al bajar */
nav {
  position: sticky;
  top: 0;
  z-index: 10;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--borde);
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}

nav .dentro {
  max-width: 1080px;
  margin: 0 auto;
  padding: 10px 20px;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

nav a {
  color: var(--suave);
  text-decoration: none;
  font-weight: 600;
  font-size: 0.9rem;
  padding: 6px 14px;
  border-radius: 999px;
  transition: background 0.15s, color 0.15s;
}

nav a:hover {
  background: var(--azul-claro);
  color: var(--azul);
}

main {
  padding: 12px 20px 48px;
}

/* cada parte del proyecto va en una tarjeta blanca */
section {
  background: #fff;
  border: 1px solid var(--borde);
  border-radius: 14px;
  box-shadow: var(--sombra);
  padding: 30px 34px;
  margin: 28px 0;
}

h2 {
  color: var(--azul-oscuro);
  margin: 0 0 14px;
  padding-bottom: 12px;
  font-size: 1.6rem;
  letter-spacing: -0.01em;
  border-bottom: 3px solid var(--azul-claro);
}

h3 {
  margin: 36px 0 8px;
  font-size: 1.12rem;
  color: var(--azul);
}

h4 {
  margin: 26px 0 4px;
  font-size: 0.98rem;
  color: var(--texto);
}

.grafica {
  width: 100%;
  margin-top: 8px;
}

.instruccion {
  color: var(--suave);
  font-size: 0.92rem;
  font-style: italic;
  margin-bottom: 0;
}

/* cuadro azul de "Qué concluir" */
.conclusion {
  background: var(--azul-claro);
  border-left: 4px solid var(--azul);
  padding: 12px 16px;
  border-radius: 0 8px 8px 0;
  font-size: 0.97rem;
}

/* cuadro amarillo de las conclusiones */
.caja {
  background: #fffaf0;
  border: 1px solid #f0d58c;
  border-left: 5px solid #e0a800;
  border-radius: 10px;
  padding: 8px 22px;
  margin-top: 14px;
}

.nota {
  color: var(--suave);
  font-size: 0.88rem;
}

/* tablas: si no caben, se recorren de lado */
.tabla {
  overflow-x: auto;
  margin: 12px 0;
  border: 1px solid var(--borde);
  border-radius: 10px;
}

table {
  border-collapse: collapse;
  font-size: 0.86rem;
  min-width: 100%;
  font-variant-numeric: tabular-nums;
}

th,
td {
  border-bottom: 1px solid var(--borde);
  padding: 9px 14px;
  text-align: right;
  white-space: nowrap;
}

th {
  background: var(--azul-oscuro);
  color: #fff;
  font-weight: 600;
  font-size: 0.82rem;
}

tbody tr:nth-child(even) td {
  background: #fcf8f8;
}

tbody tr:hover td {
  background: var(--azul-claro);
}

tbody tr:last-child td {
  border-bottom: none;
}

td.izq {
  text-align: left;
  font-weight: 500;
}

/* renglón resaltado en amarillo */
tr.resalta td,
tr.resalta:hover td {
  background: #fff3c4;
  font-weight: 700;
}

code {
  background: var(--azul-claro);
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 0.88em;
}

footer {
  color: var(--suave);
  font-size: 0.85rem;
  text-align: center;
  padding: 24px 16px 40px;
  border-top: 1px solid var(--borde);
  max-width: 1080px;
  margin: 0 auto;
}

footer a {
  color: var(--azul);
}

/* en celular: letra más chica y menos espacio */
@media (max-width: 600px) {
  header {
    padding: 32px 16px 28px;
  }

  header h1 {
    font-size: 1.5rem;
  }

  main {
    padding: 8px 10px 32px;
  }

  section {
    padding: 18px 14px;
    border-radius: 10px;
  }

  h2 {
    font-size: 1.3rem;
  }
}
"""

NOMBRES_GRAFICAS = ["1.1 Coeficientes de Ridge según lambda", "1.2 RMSE dentro y fuera de muestra por ventana",
                    "1.3 Monte Carlo", "2. VIF de cada variable según el remedio", "2. Error LOO de cada remedio",
                    "3.1 Método de la potencia", "3.1 Convergencia del método de la potencia",
                    "3.2 Cargas de los componentes principales", "4.1 Frontera eficiente",
                    "4.2 Valor de 1 peso invertido", "4.3 Sharpe dentro vs. fuera de muestra"]
assert len(NOMBRES_GRAFICAS) == len(graficas)

partes_js = ["// Datos de las gráficas del dashboard (los calcula dashboard/generar_html.py)",
             "// Cada gráfica tiene 'data' (las trazas) y 'layout' (títulos, ejes y controles).",
             "const FIGURAS = [];", ""]
for k, (nombre, fig) in enumerate(zip(NOMBRES_GRAFICAS, graficas)):
    partes_js += [f"// Gráfica {nombre} (se dibuja en el div con id g{k})",
                  f"FIGURAS.push({json.dumps(fig, ensure_ascii=False)});", ""]
partes_js += ["""// Diseño que comparten todas las gráficas
const BASE = {
  font: { family: "Inter, system-ui, sans-serif", size: 13, color: "#1d2330" },
  paper_bgcolor: "#fff",
  plot_bgcolor: "#fff",
  hovermode: "closest",
  margin: { l: 70, r: 20 },
};

// Dibuja cada gráfica en su div: g0, g1, g2, ...
FIGURAS.forEach((fig, i) => {
  const layout = Object.assign({}, BASE, fig.layout);
  layout.margin = Object.assign({}, BASE.margin, fig.layout.margin || {});
  Plotly.newPlot("g" + i, fig.data, layout, { responsive: true, displaylogo: false });
});
"""]
JS = "\n".join(partes_js)

pagina = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Proyecto ECO515 · {escape(EQUIPO)}</title>
  <!-- Plotly.js: la librería que dibuja las gráficas interactivas -->
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
{CSS}  </style>
</head>
<body>

<header>
  <div class="dentro">
    <h1>De la regresión regularizada al portafolio óptimo</h1>
    <p><b>{escape(EQUIPO)}</b> · {integrantes_html}</p>
    <p>Fundamentos de Econometría (ECO515), Universidad Panamericana, Otoño 2026 · Semilla {SEMILLA} ·
      Sector objetivo: <b>{escape(OBJETIVO)}</b> · 12 activos del portafolio</p>
  </div>
</header>

<nav>
  <div class="dentro">
    <a href="#p1">1. Regularización</a>
    <a href="#p2">2. Multicolinealidad</a>
    <a href="#p3">3. PCA</a>
    <a href="#p4">4. Markowitz</a>
    <a href="#conclusion">Conclusión</a>
  </div>
</nav>

<main>
{secciones}
</main>

<!-- datos de las gráficas y código que las dibuja -->
<script>
{JS}</script>
</body>
</html>
"""

carpeta = AQUI.parent / "docs"
carpeta.mkdir(exist_ok=True)
(carpeta / "index.html").write_text(pagina, encoding="utf-8")
print("Página creada en", (carpeta / "index.html").resolve())

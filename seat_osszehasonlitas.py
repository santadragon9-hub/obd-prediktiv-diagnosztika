#!/home/pi/szakdolg_env/bin/python3


import os
import sys

VENV_MAPPA = "/home/pi/szakdolg_env"
VENV_PYTHON = os.path.join(VENV_MAPPA, "bin", "python3")

if sys.prefix != VENV_MAPPA and os.path.isfile(VENV_PYTHON) and not os.environ.get("SZAKDOLG_VENV"):
    os.environ["SZAKDOLG_VENV"] = "1"
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import io
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

import seat_elemzo as se

MODELL_MAPPA = os.path.join(se.ALAP_MAPPA, "modell")
VALASZTAS_FAJL = os.path.join(MODELL_MAPPA, "valasztott_modell.json")

HANGOLASI_LEPES = 10
SVR_LEPES = 5
CV_REDOK = 5
VELETLEN = 42

KORLATOS_ALGORITMUSOK = ["Random forest", "Gradient boosting"]

BEMENETEK = {
    "LOAD": ["RPM", "SPD", "THROTTLE", "IAT", "COOLANT"],
    "STFT": ["RPM", "SPD", "COOLANT", "THROTTLE", "IAT", "LOAD"],
    "LTFT": ["RPM", "SPD", "COOLANT", "THROTTLE", "IAT", "LOAD"],
    "COOLANT": ["RPM", "SPD", "THROTTLE", "IAT", "LOAD"],
}

RACSOK = {
    "Linear regression": {},
    "SVR (RBF)": {
        "svr__C": [1, 10, 100],
        "svr__gamma": [0.05, 0.2, 1.0],
        "svr__epsilon": [0.5, 2.0],
    },
    "Neural network (MLP)": {
        "mlpregressor__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
        "mlpregressor__alpha": [1e-4, 1e-2],
    },
    "Random forest": {
        "min_samples_leaf": [1, 5, 20],
        "max_features": [1.0, 0.5],
    },
    "Gradient boosting": {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [15, 31, 63],
        "max_iter": [200, 500],
    },
}

ALAPBEALLITASOK = {
    "Linear regression": {},
    "SVR (RBF)": {"svr__C": 10, "svr__epsilon": 0.5},
    "Neural network (MLP)": {"mlpregressor__hidden_layer_sizes": (64, 32)},
    "Random forest": {"min_samples_leaf": 5},
    "Gradient boosting": {"max_iter": 200},
}


def uj_modell(nev, parameterek=None):
    if nev == "Linear regression":
        modell = make_pipeline(StandardScaler(), LinearRegression())
    elif nev == "SVR (RBF)":
        modell = make_pipeline(StandardScaler(), SVR())
    elif nev == "Neural network (MLP)":
        modell = make_pipeline(
            StandardScaler(),
            MLPRegressor(max_iter=500, early_stopping=True, random_state=VELETLEN),
        )
    elif nev == "Random forest":
        modell = RandomForestRegressor(n_estimators=100, random_state=VELETLEN, n_jobs=-1)
    elif nev == "Gradient boosting":
        modell = HistGradientBoostingRegressor(early_stopping=False, random_state=VELETLEN)
    else:
        raise ValueError(f"Ismeretlen algoritmus: {nev}")

    parameterek = dict(parameterek or {})
    for kulcs, ertek in parameterek.items():
        if kulcs.endswith("hidden_layer_sizes"):
            parameterek[kulcs] = tuple(ertek)
    return modell.set_params(**parameterek)


def tanitas(modell, nev, adat, bemenetek, cel):
    minta = adat.iloc[::SVR_LEPES] if nev == "SVR (RBF)" else adat
    return modell.fit(minta[bemenetek], minta[cel])


def felosztas(adat, cel):
    m = adat.dropna(subset=BEMENETEK[cel] + [cel])
    utak = sorted(m.ut.unique())
    teszt_utak = utak[-se.TESZT_UTAK_SZAMA:]
    return m[~m.ut.isin(teszt_utak)], m[m.ut.isin(teszt_utak)]


def adatok_betoltese():
    adat = se.tisztitas(se.beolvasas())
    return se.ervenyes_tartomany(adat)


def celvaltozok_szurese(adat):
    sorok = []
    for cel in ["STFT", "LTFT", "COOLANT", "LOAD"]:
        tanito, teszt = felosztas(adat, cel)
        for nev, parameterek in ALAPBEALLITASOK.items():
            modell = tanitas(uj_modell(nev, parameterek), nev, tanito, BEMENETEK[cel], cel)
            becsles = modell.predict(teszt[BEMENETEK[cel]])
            sorok.append({"cel": cel, "algoritmus": nev, "R2": r2_score(teszt[cel], becsles)})
    tabla = pd.DataFrame(sorok).pivot(index="cel", columns="algoritmus", values="R2")
    tabla = tabla.loc[["STFT", "LTFT", "COOLANT", "LOAD"], list(ALAPBEALLITASOK)].round(3)
    tabla.to_csv(os.path.join(se.ABRA_MAPPA, "celvaltozok_r2.csv"))

    print("\n" + "=" * 62)
    print("CELVALTOZOK SZURESE (R2 a felretett utakon, hangolatlan modellek)")
    print("=" * 62)
    print(tabla.to_string())
    return tabla


def felosztas_hatasa(adat):
    sorok = []
    for cel in ["STFT", "LOAD"]:
        bem = BEMENETEK[cel]
        tanito, teszt = felosztas(adat, cel)
        menet = RandomForestRegressor(n_estimators=100, random_state=VELETLEN, n_jobs=-1)
        menet.fit(tanito[bem], tanito[cel])
        r2_menet = r2_score(teszt[cel], menet.predict(teszt[bem]))

        m = adat.dropna(subset=bem + [cel])
        X_tan, X_tesz, y_tan, y_tesz = train_test_split(m[bem], m[cel], test_size=0.2, random_state=VELETLEN)
        veletlen = RandomForestRegressor(n_estimators=100, random_state=VELETLEN, n_jobs=-1)
        veletlen.fit(X_tan, y_tan)
        r2_veletlen = r2_score(y_tesz, veletlen.predict(X_tesz))
        sorok.append({"cel": cel, "veletlen_felosztas_R2": r2_veletlen, "menetenkenti_felosztas_R2": r2_menet})

    tabla = pd.DataFrame(sorok).round(3)
    tabla.to_csv(os.path.join(se.ABRA_MAPPA, "felosztas_hatasa.csv"), index=False)
    print("\n" + "=" * 62)
    print("A FELOSZTAS HATASA (random forest)")
    print("=" * 62)
    print(tabla.to_string(index=False))
    return tabla


def modell_merete(modell):
    puffer = io.BytesIO()
    joblib.dump(modell, puffer)
    return puffer.tell() / 1024


def egy_soros_ido(modell, sor):
    if hasattr(modell, "n_jobs"):
        modell.set_params(n_jobs=1)
    idok = []
    for _ in range(50):
        kezdet = time.perf_counter()
        modell.predict(sor)
        idok.append(time.perf_counter() - kezdet)
    return float(np.median(idok)) * 1000


def algoritmusok_osszehasonlitasa(adat):
    cel, bem = "LOAD", BEMENETEK["LOAD"]
    tanito, teszt = felosztas(adat, cel)
    hangolo = tanito.iloc[::HANGOLASI_LEPES]
    redok = GroupKFold(n_splits=CV_REDOK)

    print("\n" + "=" * 62)
    print("LOAD-BECSLO ALGORITMUSOK OSSZEHASONLITASA")
    print("=" * 62)
    print(f"Tanito utak: {tanito.ut.nunique()} ({len(tanito)} sor), teszt utak: {teszt.ut.nunique()} ({len(teszt)} sor)")
    print(f"Hangolas: minden {HANGOLASI_LEPES}. tanito sor ({len(hangolo)} sor), {CV_REDOK} csoportos red")

    alap = teszt[cel].sub(tanito[cel].mean())
    sorok = [{
        "algoritmus": "Mean (baseline)", "parameterek": "-", "CV_MAE": np.nan, "CV_MAE_szoras": np.nan,
        "teszt_R2": r2_score(teszt[cel], np.full(len(teszt), tanito[cel].mean())),
        "teszt_MAE": alap.abs().mean(), "teszt_RMSE": float(np.sqrt((alap ** 2).mean())),
        "tanitasi_ido_s": np.nan, "becslesi_ido_ms": np.nan, "meret_kB": np.nan,
    }]
    modellek = {}

    for nev, racs in RACSOK.items():
        kereses = GridSearchCV(uj_modell(nev), racs, scoring="neg_mean_absolute_error", cv=redok, n_jobs=1)
        kereses.fit(hangolo[bem], hangolo[cel], groups=hangolo.ut)
        legjobb = kereses.best_params_

        modell = uj_modell(nev, legjobb)
        kezdet = time.time()
        tanitas(modell, nev, tanito, bem, cel)
        tanitasi_ido = time.time() - kezdet
        becsles = modell.predict(teszt[bem])
        modellek[nev] = modell

        sorok.append({
            "algoritmus": nev,
            "parameterek": json.dumps({k.split("__")[-1]: v for k, v in legjobb.items()}),
            "CV_MAE": -kereses.best_score_,
            "CV_MAE_szoras": kereses.cv_results_["std_test_score"][kereses.best_index_],
            "teszt_R2": r2_score(teszt[cel], becsles),
            "teszt_MAE": mean_absolute_error(teszt[cel], becsles),
            "teszt_RMSE": float(np.sqrt(mean_squared_error(teszt[cel], becsles))),
            "tanitasi_ido_s": tanitasi_ido,
            "becslesi_ido_ms": egy_soros_ido(modell, teszt[bem].iloc[[0]]),
            "meret_kB": modell_merete(modell),
            "_legjobb": legjobb,
        })
        print(f"  {nev:22} CV MAE {-kereses.best_score_:.2f}  teszt R2 {sorok[-1]['teszt_R2']:.3f}  {sorok[-1]['parameterek']}")

    tabla = pd.DataFrame(sorok)
    rangsor = tabla.dropna(subset=["CV_MAE"]).sort_values("CV_MAE")
    legpontosabb = rangsor.iloc[0]
    nyertes = rangsor[rangsor.algoritmus.isin(KORLATOS_ALGORITMUSOK)].iloc[0]
    os.makedirs(MODELL_MAPPA, exist_ok=True)
    with open(VALASZTAS_FAJL, "w") as fajl:
        json.dump({"algoritmus": nyertes.algoritmus, "parameterek": nyertes["_legjobb"]}, fajl, indent=2)

    kiiras = tabla.drop(columns="_legjobb").round(
        {"CV_MAE": 2, "CV_MAE_szoras": 2, "teszt_R2": 3, "teszt_MAE": 2, "teszt_RMSE": 2,
         "tanitasi_ido_s": 1, "becslesi_ido_ms": 2, "meret_kB": 0}
    )
    kiiras.to_csv(os.path.join(se.ABRA_MAPPA, "algoritmusok.csv"), index=False)
    print("\n" + kiiras.drop(columns="parameterek").to_string(index=False))
    print(f"\nLegkisebb CV MAE osszesen: {legpontosabb.algoritmus}")
    print(f"Valasztott algoritmus (legkisebb CV MAE a korlatos becslesu modellek kozul): "
          f"{nyertes.algoritmus}  ->  {VALASZTAS_FAJL}")

    valasztott = modellek[nyertes.algoritmus]
    minta = teszt.sample(min(10000, len(teszt)), random_state=VELETLEN)
    fontossag = permutation_importance(
        valasztott, minta[bem], minta[cel], scoring="neg_mean_absolute_error",
        n_repeats=5, random_state=VELETLEN,
    )
    fontos = pd.Series(fontossag.importances_mean, index=bem).sort_values(ascending=False)
    fontos.round(3).to_csv(os.path.join(se.ABRA_MAPPA, "permutacios_fontossag.csv"), header=["MAE_novekedes"])
    print("\nPermutacios fontossag (a teszt-MAE novekedese, szazalekpont):")
    print(fontos.round(2).to_string())

    abrazolas(kiiras, valasztott, nyertes.algoritmus, teszt, bem, cel)
    return kiiras


def abrazolas(tabla, modell, nev, teszt, bem, cel):
    rendezett = tabla.sort_values("teszt_MAE", ascending=False)
    fig, (bal, jobb) = plt.subplots(1, 2, figsize=(13, 5))
    szinek = ["tab:blue" if a == nev else "tab:gray" for a in rendezett.algoritmus]
    bal.barh(rendezett.algoritmus, rendezett.teszt_MAE, color=szinek)
    for i, ertek in enumerate(rendezett.teszt_MAE):
        bal.text(ertek + 0.1, i, f"{ertek:.2f}", va="center", fontsize=9)
    bal.set_xlabel("MAE a felretett utakon (szazalekpont)")
    bal.set_title("A LOAD becslesenek hibaja algoritmusonkent")
    bal.grid(True, axis="x", linestyle="--", alpha=0.4)

    minta = teszt.sample(min(4000, len(teszt)), random_state=1)
    jobb.scatter(modell.predict(minta[bem]), minta[cel], s=3, alpha=0.25, color="tab:blue")
    hatar = [0, 100]
    jobb.plot(hatar, hatar, "k--", linewidth=1)
    jobb.set_xlabel("Becsult LOAD (%)")
    jobb.set_ylabel("Mert LOAD (%)")
    jobb.set_title(f"{nev}: becsles es meres")
    jobb.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(se.ABRA_MAPPA, "4_algoritmusok.png"), dpi=300)
    plt.close(fig)
    print("Abra: abrak/4_algoritmusok.png")


if __name__ == "__main__":
    kezdet = time.time()
    adat = adatok_betoltese()
    celvaltozok_szurese(adat)
    felosztas_hatasa(adat)
    algoritmusok_osszehasonlitasa(adat)
    print(f"\nKesz ({(time.time() - kezdet) / 60:.1f} perc).")

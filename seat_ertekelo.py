#!/home/pi/szakdolg_env/bin/python3

import os
import sys
 
VENV_MAPPA = "/home/pi/szakdolg_env"
VENV_PYTHON = os.path.join(VENV_MAPPA, "bin", "python3")
 
if sys.prefix != VENV_MAPPA and os.path.isfile(VENV_PYTHON) and not os.environ.get("SZAKDOLG_VENV"):
    os.environ["SZAKDOLG_VENV"] = "1"
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)
 
import contextlib
import io
import json
import math
 
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
 
import seat_elemzo as se
from seat_osszehasonlitas import BEMENETEK, MODELL_MAPPA, VALASZTAS_FAJL, tanitas as modell_tanitasa, uj_modell
 
CSOMAG_FAJL = os.path.join(MODELL_MAPPA, "load_modell.joblib")
ELOZMENY_FAJL = os.path.join(MODELL_MAPPA, "ut_elozmeny.csv")
 
BEM = BEMENETEK["LOAD"]
LAMBDA = 0.2
L_SZORZO = 3.0
MIN_SOROK = 300
CV_REDOK = 5
FOJTOSZELEP_SAVOK = [0, 10, 20, 30, 50, 100]
 
 
def valasztott_algoritmus():
    if os.path.isfile(VALASZTAS_FAJL):
        with open(VALASZTAS_FAJL) as fajl:
            valasztas = json.load(fajl)
        return valasztas["algoritmus"], valasztas["parameterek"]
    return "Random forest", {"min_samples_leaf": 5}
 
 
def ervenyes_sorok(df):
    return se.ervenyes_tartomany(df).dropna(subset=BEM + ["LOAD"])
 
 
def utankent(df, rez):
    return (
        df.assign(rez=rez)
        .groupby("ut")
        .agg(kezdet=("TIME", "min"), sorok=("rez", "size"), reziduum=("rez", "mean"))
        .reset_index()
    )
 
 
def tanitas(adat):
    m = ervenyes_sorok(adat)
    ref = m[m.TIME >= se.SZERVIZ_DATUM]
    nev, param = valasztott_algoritmus()
 
    oos = pd.Series(np.nan, index=ref.index)
    for tan_idx, ert_idx in GroupKFold(n_splits=CV_REDOK).split(ref, groups=ref.ut):
        modell = modell_tanitasa(uj_modell(nev, param), nev, ref.iloc[tan_idx], BEM, "LOAD")
        ertekelo = ref.iloc[ert_idx]
        oos.iloc[ert_idx] = ertekelo.LOAD.values - modell.predict(ertekelo[BEM])
 
    ref_utak = utankent(ref, oos)
    ref_utak = ref_utak[ref_utak.sorok >= MIN_SOROK]
    mu0 = float(ref_utak.reziduum.mean())
    sigma = float(ref_utak.reziduum.std(ddof=1))
 
    vegso = modell_tanitasa(uj_modell(nev, param), nev, ref, BEM, "LOAD")
    if hasattr(vegso, "n_jobs"):
        vegso.set_params(n_jobs=1)
 
    csomag = {
        "modell": vegso,
        "algoritmus": nev,
        "parameterek": param,
        "bemenetek": BEM,
        "mu0": mu0,
        "sigma": sigma,
        "lambda": LAMBDA,
        "L": L_SZORZO,
        "referencia_kezdete": str(se.SZERVIZ_DATUM.date()),
        "ref_utak": ref_utak,
    }
    os.makedirs(MODELL_MAPPA, exist_ok=True)
    joblib.dump(csomag, CSOMAG_FAJL)
 
    print("\n" + "=" * 62)
    print(f"REFERENCIA-MODELL ({nev}, tanitva {csomag['referencia_kezdete']} utan)")
    print("=" * 62)
    print(f"Referencia utak: {len(ref_utak)} ({len(ref)} sor), parameterek: {param}")
    print(f"Egeszseges ut atlagos reziduuma (keresztvalidalt): mu0 = {mu0:+.2f}, szoras = {sigma:.2f} szazalekpont")
    print(f"Modellcsomag: {CSOMAG_FAJL}")
    return csomag, oos
 
 
def csomag_betoltese():
    return joblib.load(CSOMAG_FAJL)
 
 
def ewma_hatar(t, csomag):
    lam, L, sigma = csomag["lambda"], csomag["L"], csomag["sigma"]
    return L * sigma * math.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * t)))
 
 
def allapot(z, also, felso):
    if z < also:
        return "ELTERES: kevesebb levego a vartnal"
    if z > felso:
        return "ELTERES: tobb levego a vartnal"
    return "NORMAL"
 
 
def ewma_sorozat(tabla, csomag):
    z = csomag["mu0"]
    sorok = []
    for t, x in enumerate(tabla.reziduum, start=1):
        z = csomag["lambda"] * x + (1 - csomag["lambda"]) * z
        fel = ewma_hatar(t, csomag)
        also, felso = csomag["mu0"] - fel, csomag["mu0"] + fel
        sorok.append({"ewma": z, "also": also, "felso": felso, "allapot": allapot(z, also, felso)})
    return pd.concat([tabla.reset_index(drop=True), pd.DataFrame(sorok)], axis=1)
 
 
def statisztikai_probak(elott, utan):
    """A szerviz elotti es utani utak atlagos reziduumanak osszehasonlitasa."""
    try:
        from scipy import stats
    except ImportError:
        print("A statisztikai probakhoz a scipy csomag szukseges.")
        return
    negativ = int((elott < 0).sum())
    welch = stats.ttest_ind(elott, utan, equal_var=False)
    mann_whitney = stats.mannwhitneyu(elott, utan, alternative="less")
    elojel = stats.binomtest(negativ, len(elott), 0.5, alternative="greater")
    print(f"Kulonbseg a szerviz utani utakhoz kepest: {elott.mean() - utan.mean():+.2f} szazalekpont")
    print(f"Welch-proba: t = {welch.statistic:.2f}, p = {welch.pvalue:.3f} (ketoldalu)")
    print(f"Mann-Whitney-proba: p = {mann_whitney.pvalue:.3f} (egyoldalu)")
    print(f"Elojelproba: {negativ} negativ / {len(elott)} ut, p = {elojel.pvalue:.3f} (egyoldalu)")
 
 
def elozmeny_epitese(adat, csomag):
    m = ervenyes_sorok(adat)
    elott = m[m.TIME < se.SZERVIZ_DATUM]
    elott_utak = utankent(elott, elott.LOAD.values - csomag["modell"].predict(elott[BEM]))
    rovid = elott_utak[elott_utak.sorok < MIN_SOROK]
    elott_utak = elott_utak[elott_utak.sorok >= MIN_SOROK]
 
    elotte = ewma_sorozat(elott_utak.sort_values("kezdet").assign(idoszak="szerviz elott"), csomag)
    utana = ewma_sorozat(
        csomag["ref_utak"].sort_values("kezdet").assign(idoszak="szerviz utan", referencia=True), csomag
    )
    elotte["referencia"] = False
    elozmeny = pd.concat([elotte, utana], ignore_index=True)
    elozmeny.round({"reziduum": 3, "ewma": 3, "also": 3, "felso": 3}).to_csv(ELOZMENY_FAJL, index=False)
 
    print("\n" + "=" * 62)
    print(f"UTANKENTI KIERTEKELES ES EWMA (lambda = {csomag['lambda']}, L = {csomag['L']})")
    print("=" * 62)
    if len(rovid):
        print(f"Tul rovid (< {MIN_SOROK} ervenyes sor), kihagyva: {', '.join(rovid.ut)}")
    print(f"Szerviz elotti utak: {len(elotte)}, atlagos reziduum {elotte.reziduum.mean():+.2f}, "
          f"median {elotte.reziduum.median():+.2f} szazalekpont, negativ: {int((elotte.reziduum < 0).sum())}")
    statisztikai_probak(elotte.reziduum, utana.reziduum)
    elso_riasztas = elotte[elotte.allapot != "NORMAL"]
    if len(elso_riasztas):
        sor = elso_riasztas.iloc[0]
        print(f"Elso eltero allapot: {sor.kezdet:%Y-%m-%d %H:%M} ({elotte.index.get_loc(sor.name) + 1}. ut), "
              f"EWMA {sor.ewma:+.2f} < also hatar {sor.also:+.2f}")
    print(f"Szerviz elott eltero allapotu utak: {int((elotte.allapot != 'NORMAL').sum())} / {len(elotte)}")
    print(f"Szerviz utan eltero allapotu utak:  {int((utana.allapot != 'NORMAL').sum())} / {len(utana)}")
    print(f"Elozmeny: {ELOZMENY_FAJL}")
 
    ewma_abra(elozmeny)
    return elozmeny
 
 
def ewma_abra(elozmeny):
    os.makedirs(se.ABRA_MAPPA, exist_ok=True)
    fig, tengely = plt.subplots(figsize=(11, 5))
    for idoszak, adat in elozmeny.groupby("idoszak", sort=False):
        tengely.scatter(adat.kezdet, adat.reziduum, s=22, color="tab:gray", zorder=3)
        tengely.plot(adat.kezdet, adat.ewma, color="tab:blue", linewidth=1.8, zorder=4)
        tengely.plot(adat.kezdet, adat.also, color="tab:red", linestyle="--", linewidth=1)
        tengely.plot(adat.kezdet, adat.felso, color="tab:red", linestyle="--", linewidth=1)
        riaszt = adat[adat.allapot != "NORMAL"]
        tengely.scatter(riaszt.kezdet, riaszt.ewma, s=40, color="tab:red", zorder=5)
    tengely.axvline(se.SZERVIZ_DATUM, color="black", linestyle=":", linewidth=1.5)
    tengely.axhline(0, color="black", linewidth=0.8)
    tengely.set_ylabel("Atlagos LOAD-reziduum (szazalekpont)")
    tengely.set_title("Utankenti reziduum (pont), EWMA (vonal) es szabalyozasi hatarok (szaggatott)")
    tengely.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(se.ABRA_MAPPA, "6_ewma.png"), dpi=300)
    plt.close(fig)
    print("Abra: abrak/6_ewma.png")
 
 
def fojtoszelep_savok(adat, csomag, oos):
    m = ervenyes_sorok(adat)
    elott = m[m.TIME < se.SZERVIZ_DATUM]
    ref = m[m.TIME >= se.SZERVIZ_DATUM]
    savok = pd.cut
    elott_rez = pd.Series(elott.LOAD.values - csomag["modell"].predict(elott[BEM]), index=elott.index)
    tabla = pd.DataFrame({
        "elott_rez": elott_rez.groupby(savok(elott.THROTTLE, FOJTOSZELEP_SAVOK)).mean(),
        "elott_sorok": elott_rez.groupby(savok(elott.THROTTLE, FOJTOSZELEP_SAVOK)).size(),
        "utan_rez": oos.groupby(savok(ref.THROTTLE, FOJTOSZELEP_SAVOK)).mean(),
        "utan_sorok": oos.groupby(savok(ref.THROTTLE, FOJTOSZELEP_SAVOK)).size(),
    })
    tabla["kulonbseg"] = tabla.elott_rez - tabla.utan_rez
    tabla.index = [f"{int(s.left)}-{int(s.right)}%" for s in tabla.index]
    tabla.round(2).to_csv(os.path.join(se.ABRA_MAPPA, "fojtoszelep_savok.csv"), index_label="fojtoszelep")
 
    print("\n" + "=" * 62)
    print("REZIDUUM FOJTOSZELEP-SAVONKENT (szazalekpont)")
    print("=" * 62)
    print(tabla.round(2).to_string())
 
    fig, tengely = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(tabla))
    tengely.bar(x - 0.2, tabla.utan_rez, width=0.4, color="tab:gray", label="szerviz utan (keresztvalidalt)")
    tengely.bar(x + 0.2, tabla.elott_rez, width=0.4, color="tab:blue", label="szerviz elott")
    tengely.set_xticks(x, tabla.index)
    tengely.axhline(0, color="black", linewidth=0.8)
    tengely.set_xlabel("Fojtoszelep-allas")
    tengely.set_ylabel("Atlagos LOAD-reziduum (szazalekpont)")
    tengely.legend()
    tengely.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(se.ABRA_MAPPA, "5_fojtoszelep_savok.png"), dpi=300)
    plt.close(fig)
    print("Abra: abrak/5_fojtoszelep_savok.png")
    return tabla
 
 
def pillanatnyi_reziduum(csomag, ertekek):
    """Egy masodperc mert ertekeibol a LOAD-reziduum, vagy None, ha nem ertekelheto."""
    try:
        bemenet = [float(ertekek[nev]) for nev in csomag["bemenetek"]]
        load = float(ertekek["LOAD"])
        rpm, coolant = float(ertekek["RPM"]), float(ertekek["COOLANT"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(math.isnan(v) for v in bemenet + [load]):
        return None
    if rpm < se.ERVENYES_RPM or coolant < se.ERVENYES_COOLANT:
        return None
    sor = pd.DataFrame([bemenet], columns=csomag["bemenetek"])
    return load - float(csomag["modell"].predict(sor)[0])
 
 
def ut_beolvasasa(fajl):
    df = pd.read_csv(fajl).rename(columns=se.OSZLOPNEVEK)
    df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
    for oszlop in se.MERT_OSZLOPOK:
        df[oszlop] = pd.to_numeric(df[oszlop], errors="coerce") if oszlop in df.columns else np.nan
    df["ut"] = os.path.basename(fajl)
    df = df.dropna(subset=["TIME"])
    with contextlib.redirect_stdout(io.StringIO()):
        return se.tisztitas(df)
 
 
def ut_ertekelese(fajl, csomag=None):
    """Egy befejezett ut kiertekelese es hozzafuzese az elozmenyhez."""
    csomag = csomag or csomag_betoltese()
    m = ervenyes_sorok(ut_beolvasasa(fajl))
    if len(m) < MIN_SOROK:
        return {"allapot": f"NEM ERTEKELHETO (csak {len(m)} ervenyes sor)", "sorok": len(m)}
 
    rez = float((m.LOAD.values - csomag["modell"].predict(m[BEM])).mean())
    nev = os.path.basename(fajl)
    if os.path.isfile(ELOZMENY_FAJL):
        elozmeny = pd.read_csv(ELOZMENY_FAJL, parse_dates=["kezdet"])
        elozmeny = elozmeny[elozmeny.ut != nev]
    else:
        elozmeny = pd.DataFrame(columns=["ut", "kezdet", "sorok", "reziduum", "idoszak", "ewma", "also", "felso", "allapot", "referencia"])
 
    aktualis = elozmeny[elozmeny.idoszak == "szerviz utan"].sort_values("kezdet")
    t = len(aktualis) + 1
    z_elozo = float(aktualis.ewma.iloc[-1]) if len(aktualis) else csomag["mu0"]
    z = csomag["lambda"] * rez + (1 - csomag["lambda"]) * z_elozo
    fel = ewma_hatar(t, csomag)
    also, felso = csomag["mu0"] - fel, csomag["mu0"] + fel
    uj = {
        "ut": nev, "kezdet": m.TIME.min(), "sorok": len(m), "reziduum": round(rez, 3),
        "idoszak": "szerviz utan", "ewma": round(z, 3), "also": round(also, 3), "felso": round(felso, 3),
        "allapot": allapot(z, also, felso), "referencia": False,
    }
    elozmeny = pd.concat([elozmeny, pd.DataFrame([uj])], ignore_index=True).sort_values("kezdet")
    elozmeny.to_csv(ELOZMENY_FAJL, index=False)
    try:
        ewma_abra(elozmeny)
    except Exception as hiba:
        print(f"Az EWMA-abra nem frissult: {hiba}")
    return uj
 
 
if __name__ == "__main__":
    adat = se.tisztitas(se.beolvasas())
    csomag, oos = tanitas(adat)
    elozmeny_epitese(adat, csomag)
    fojtoszelep_savok(adat, csomag, oos)
    print("\nKesz.")

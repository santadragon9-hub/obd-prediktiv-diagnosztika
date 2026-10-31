#!/home/pi/szakdolg_env/bin/python3

import os
import sys

VENV_MAPPA = "/home/pi/szakdolg_env"
VENV_PYTHON = os.path.join(VENV_MAPPA, "bin", "python3")

if sys.prefix != VENV_MAPPA and os.path.isfile(VENV_PYTHON) and not os.environ.get("SZAKDOLG_VENV"):
    os.environ["SZAKDOLG_VENV"] = "1"
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALAP_MAPPA = os.path.dirname(os.path.abspath(__file__))
LOG_MAPPA = os.path.join(ALAP_MAPPA, "logs")
ABRA_MAPPA = os.path.join(ALAP_MAPPA, "abrak")

SZERVIZ_DATUM = pd.Timestamp("2026-04-23")

RPM_LIMIT = 5500
SPD_LIMIT = 170
COOLANT_LIMIT = 130
TRIM_LIMIT = 50
RPM_ELTERES = 1500
COOLANT_ELTERES = 8
IAT_ELTERES = 10

TESZT_UTAK_SZAMA = 7

ERVENYES_RPM = 500
ERVENYES_COOLANT = 80

OSZLOPNEVEK = {
    "TIMESTAMP": "TIME",
    "SPEED": "SPD",
    "COOLANT_TEMP": "COOLANT",
    "THROTTLE_POS": "THROTTLE",
    "INTAKE_AIR_TEMP": "IAT",
}
MERT_OSZLOPOK = ["RPM", "SPD", "COOLANT", "THROTTLE", "IAT", "LOAD", "STFT", "LTFT"]
MODELL_BEMENETEK = ["RPM", "SPD", "THROTTLE", "IAT", "COOLANT"]


def beolvasas():
    os.makedirs(ABRA_MAPPA, exist_ok=True)
    fajlok = sorted(glob.glob(os.path.join(LOG_MAPPA, "seat_log_*.csv")))
    if not fajlok:
        raise SystemExit(f"Nem talalhato seat_log_*.csv itt: {LOG_MAPPA}")

    reszek, jelentes = [], []
    for ut in fajlok:
        nev = os.path.basename(ut)
        df = pd.read_csv(ut).rename(columns=OSZLOPNEVEK)

        if df.empty or "TIME" not in df.columns:
            jelentes.append({"fajl": nev, "sorok": 0, "hianyzo_oszlopok": "URES / HIBAS"})
            continue

        hianyzo = [o for o in MERT_OSZLOPOK if o not in df.columns]
        df["TIME"] = pd.to_datetime(df["TIME"], errors="coerce")
        for oszlop in MERT_OSZLOPOK:
            df[oszlop] = pd.to_numeric(df[oszlop], errors="coerce") if oszlop in df.columns else np.nan
        df["ut"] = nev

        reszek.append(df[["TIME", "ut"] + MERT_OSZLOPOK])
        jelentes.append({"fajl": nev, "sorok": len(df), "hianyzo_oszlopok": ", ".join(hianyzo) or "-"})

    adat = pd.concat(reszek, ignore_index=True)
    adat = adat.dropna(subset=["TIME"]).sort_values(["ut", "TIME"]).reset_index(drop=True)

    jelentes = pd.DataFrame(jelentes)
    jelentes.to_csv(os.path.join(ABRA_MAPPA, "fajl_attekintes.csv"), index=False)

    hianyosak = jelentes[jelentes.hianyzo_oszlopok != "-"]
    print("=" * 62)
    print("BEOLVASAS")
    print("=" * 62)
    print(f"Talalt fajlok    : {len(fajlok)} db")
    print(f"Feldolgozott     : {len(reszek)} db")
    print(f"Nyers adathalmaz : {len(adat)} sor")
    print(f"Idoszak          : {adat.TIME.min():%Y-%m-%d} - {adat.TIME.max():%Y-%m-%d}")
    lepesek = adat.groupby("ut").TIME.diff().dt.total_seconds()
    kihagyasok = lepesek[lepesek > 60]
    print(f"Osszes menetido  : {lepesek[lepesek <= 60].sum() / 3600:.1f} ora "
          f"({len(kihagyasok)} db 60 s-nal hosszabb idougras nelkul, osszesen {kihagyasok.sum() / 3600:.1f} ora)")

    if not hianyosak.empty:
        print(f"\nEltero szerkezetu fajlok ({len(hianyosak)} db) - beolvasva, a hianyzo mezok uresek:")
        print(hianyosak.to_string(index=False))
    print("\nReszletes attekintes: abrak/fajl_attekintes.csv")
    return adat


def tisztitas(adat):
    elotte = len(adat)

    hatarsertes = (
        (adat.RPM > RPM_LIMIT)
        | (adat.SPD > SPD_LIMIT)
        | (adat.COOLANT > COOLANT_LIMIT)
        | (adat.STFT.abs() > TRIM_LIMIT)
        | (adat.LTFT.abs() > TRIM_LIMIT)
    )

    def mediantol_elter(oszlop, kuszob):
        kozep = adat.groupby("ut")[oszlop].transform(
            lambda s: s.rolling(7, center=True, min_periods=1).median()
        )
        return (adat[oszlop] - kozep).abs() > kuszob

    kiugras = (
        mediantol_elter("RPM", RPM_ELTERES)
        | mediantol_elter("COOLANT", COOLANT_ELTERES)
        | mediantol_elter("IAT", IAT_ELTERES)
    )

    adat = adat[~hatarsertes & ~kiugras].copy()

    print("\n" + "=" * 62)
    print("ADATTISZTITAS")
    print("=" * 62)
    print(f"Fizikai hatarerteket serto sorok  : {int(hatarsertes.sum())}")
    print(f"Mozgo mediantol kiugro sorok      : {int(kiugras.sum())}")
    eldobva = elotte - len(adat)
    print(f"Osszesen eldobva                  : {eldobva} sor ({eldobva / elotte:.3%})")
    print(f"Megmaradt adathalmaz              : {len(adat)} sor")
    return adat


def ervenyes_tartomany(adat):
    """Jaro, uzemmeleg motor: csak itt van eleg tanitoadat a modellekhez."""
    return adat[(adat.RPM >= ERVENYES_RPM) & (adat.COOLANT >= ERVENYES_COOLANT)]


def leiro_statisztika(adat):
    tabla = adat[MERT_OSZLOPOK].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)
    tabla.to_csv(os.path.join(ABRA_MAPPA, "leiro_statisztika.csv"))

    uzem = pd.Series({
        "alapjarat (SPD<=1)": (adat.SPD <= 1).mean(),
        "varosi (1<SPD<=50)": ((adat.SPD > 1) & (adat.SPD <= 50)).mean(),
        "orszaguti (50<SPD<=90)": ((adat.SPD > 50) & (adat.SPD <= 90)).mean(),
        "autopalya (SPD>90)": (adat.SPD > 90).mean(),
        "hideg motor (COOLANT<60)": (adat.COOLANT < 60).mean(),
    }) * 100
    uzem.round(2).to_csv(os.path.join(ABRA_MAPPA, "uzemallapotok.csv"), header=["szazalek"])

    print("\n" + "=" * 62)
    print("LEIRO STATISZTIKA")
    print("=" * 62)
    print(tabla.to_string())
    print("\nUzemallapotok megoszlasa (%)")
    print(uzem.round(1).to_string())
    ervenyes = len(ervenyes_tartomany(adat)) / len(adat)
    print(f"\nErvenyes tartomany (RPM >= {ERVENYES_RPM}, COOLANT >= {ERVENYES_COOLANT} C): a sorok {ervenyes:.1%}-a")


def idosor_abra(adat):
    leghosszabb = adat.ut.value_counts().idxmax()
    ut = adat[adat.ut == leghosszabb]
    perc = (ut.TIME - ut.TIME.min()).dt.total_seconds() / 60

    fig, tengelyek = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for tengely, (oszlop, cim, szin) in zip(tengelyek, [
        ("RPM", "Motorfordulatszam (1/min)", "tab:red"),
        ("SPD", "Jarmusebesseg (km/h)", "tab:blue"),
        ("COOLANT", "Hutoviz homerseklete (C)", "tab:orange"),
        ("LTFT", "Hosszutavu korrekcio (%)", "tab:green"),
    ]):
        tengely.plot(perc, ut[oszlop], color=szin, linewidth=0.8)
        tengely.set_ylabel(cim, fontsize=9)
        tengely.grid(True, linestyle="--", alpha=0.4)
    tengelyek[-1].set_xlabel("Eltelt ido a meres kezdetetol (perc)")
    fig.suptitle(f"Mintameres idosora - {leghosszabb}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ABRA_MAPPA, "1_idosor.png"), dpi=300)
    plt.close(fig)
    print(f"\nAbra: abrak/1_idosor.png  ({leghosszabb})")


def korrelacio(adat):
    pearson = adat[MERT_OSZLOPOK].corr().round(2)
    spearman = adat[MERT_OSZLOPOK].corr(method="spearman").round(2)
    pearson.to_csv(os.path.join(ABRA_MAPPA, "korrelacio_pearson.csv"))
    spearman.to_csv(os.path.join(ABRA_MAPPA, "korrelacio_spearman.csv"))

    print("\n" + "=" * 62)
    print("KORRELACIOK (Spearman)")
    print("=" * 62)
    print(spearman.to_string())

    fig, tengely = plt.subplots(figsize=(7.5, 6.5))
    kep = tengely.imshow(spearman.values, cmap="RdBu_r", vmin=-1, vmax=1)
    tengely.set_xticks(range(len(MERT_OSZLOPOK)), MERT_OSZLOPOK, rotation=45, ha="right")
    tengely.set_yticks(range(len(MERT_OSZLOPOK)), MERT_OSZLOPOK)
    for i in range(len(MERT_OSZLOPOK)):
        for j in range(len(MERT_OSZLOPOK)):
            tengely.text(j, i, f"{spearman.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(kep, ax=tengely, label="Spearman-korrelacio")
    tengely.set_title("A mert valtozok rangkorrelacioja")
    fig.tight_layout()
    fig.savefig(os.path.join(ABRA_MAPPA, "2_korrelacio.png"), dpi=300)
    plt.close(fig)
    print("\nAbra: abrak/2_korrelacio.png")


def ltft_elemzes(adat):
    m = adat.dropna(subset=MODELL_BEMENETEK + ["LOAD"])
    napi = m.groupby("ut").agg(nap=("TIME", "min"), ltft=("LTFT", "mean")).dropna()
    elott = napi[napi.nap < SZERVIZ_DATUM].ltft.mean()
    utana = napi[napi.nap >= SZERVIZ_DATUM].ltft.mean()

    print("\n" + "=" * 62)
    print(f"HOSSZU TAVU KORREKCIO ES A SZERVIZ ({SZERVIZ_DATUM:%Y-%m-%d})")
    print("=" * 62)
    print(f"Utankenti atlagos LTFT: szerviz elott {elott:.2f}%, utana {utana:.2f}%")

    fig, tengely = plt.subplots(figsize=(10, 4.5))
    tengely.scatter(napi.nap, napi.ltft, color="tab:blue", s=28)
    tengely.axvline(SZERVIZ_DATUM, color="tab:red", linestyle="--", linewidth=1.5, label="szerviz")
    tengely.set_ylabel("Atlagos LTFT (%)")
    tengely.set_title("Hosszutavu uzemanyag-korrekcio utankent")
    tengely.legend()
    tengely.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(ABRA_MAPPA, "3_ltft_szerviz.png"), dpi=300)
    plt.close(fig)
    print("Abra: abrak/3_ltft_szerviz.png")

    ltft_regresszio(m)


def ltft_regresszio(m):
    utankent = m.groupby("ut").agg(
        LTFT=("LTFT", "mean"),
        LOAD=("LOAD", "mean"),
        IAT=("IAT", "mean"),
        kezdet=("TIME", "min"),
    ).dropna()

    if len(utankent) < 8:
        return

    utankent["nap"] = (utankent.kezdet - utankent.kezdet.min()).dt.days
    utankent["szerviz"] = (utankent.kezdet >= SZERVIZ_DATUM).astype(int)

    print("\n  LTFT magyarazo modellek (utankenti atlagokon, linearis regresszio)")
    for nev, valtozok in [
        ("LOAD + IAT + nap", ["LOAD", "IAT", "nap"]),
        ("LOAD + IAT + nap + szerviz", ["LOAD", "IAT", "nap", "szerviz"]),
    ]:
        A = np.c_[np.ones(len(utankent)), utankent[valtozok].values]
        egyutthatok, *_ = np.linalg.lstsq(A, utankent.LTFT.values, rcond=None)
        maradek = utankent.LTFT.values - A @ egyutthatok
        r2 = 1 - (maradek ** 2).sum() / ((utankent.LTFT - utankent.LTFT.mean()) ** 2).sum()
        reszletek = "   ".join(f"{v}={e:+.3f}" for v, e in zip(valtozok, egyutthatok[1:]))
        print(f"    {nev:28} R2 = {r2:.3f}    {reszletek}")


if __name__ == "__main__":
    adat = beolvasas()
    adat = tisztitas(adat)
    leiro_statisztika(adat)
    korrelacio(adat)
    idosor_abra(adat)
    ltft_elemzes(adat)
    print("\nKesz. Az abrak es a tablazatok itt vannak: " + ABRA_MAPPA)

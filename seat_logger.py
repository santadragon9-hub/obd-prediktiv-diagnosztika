#!/home/pi/szakdolg_env/bin/python3

import os
import sys

VENV_MAPPA = "/home/pi/szakdolg_env"
VENV_PYTHON = os.path.join(VENV_MAPPA, "bin", "python3")

if sys.prefix != VENV_MAPPA and os.path.isfile(VENV_PYTHON) and not os.environ.get("SZAKDOLG_VENV"):
    os.environ["SZAKDOLG_VENV"] = "1"
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import csv
import time
from datetime import datetime

import obd

PORT = "/dev/rfcomm0"
ALAP_MAPPA = os.path.dirname(os.path.abspath(__file__))
LOG_MAPPA = os.path.join(ALAP_MAPPA, "logs")
MINTAVETEL_MP = 1
NAN_FIGYELMEZTETES = 10

PARANCSOK = [
    ("RPM", obd.commands.RPM),
    ("SPD", obd.commands.SPEED),
    ("COOLANT", obd.commands.COOLANT_TEMP),
    ("THROTTLE", obd.commands.THROTTLE_POS),
    ("IAT", obd.commands.INTAKE_TEMP),
    ("LOAD", obd.commands.ENGINE_LOAD),
    ("STFT", obd.commands.SHORT_FUEL_TRIM_1),
    ("LTFT", obd.commands.LONG_FUEL_TRIM_1),
]


def ertekelo_betoltese():
    try:
        import seat_ertekelo
        csomag = seat_ertekelo.csomag_betoltese()
        print(f"Modell betoltve: {csomag['algoritmus']} (referencia: {csomag['referencia_kezdete']} utan)")
        return seat_ertekelo, csomag
    except Exception as hiba:
        print(f"Nincs betoltheto modell, csak naplozas tortenik ({hiba.__class__.__name__}).")
        return None, None


def jarmu_adatai(connection, idobelyeg):
    sorok = [f"Idopont: {datetime.now():%Y-%m-%d %H:%M:%S}"]
    try:
        sorok.append(f"Protokoll: {connection.protocol_name()}")
    except Exception as hiba:
        sorok.append(f"Protokoll: nem olvashato ({hiba})")
    try:
        tamogatott = sorted(c.name for c in connection.supported_commands if c.mode == 1)
        sorok.append(f"Tamogatott Mode 01 parancsok ({len(tamogatott)}): {', '.join(tamogatott)}")
    except Exception as hiba:
        sorok.append(f"Tamogatott parancsok: nem olvashato ({hiba})")
    for nev, parancs in [("MIL es hibakodszam", obd.commands.STATUS), ("Tarolt hibakodok (Mode 03)", obd.commands.GET_DTC)]:
        try:
            valasz = obd.OBD.query(connection, parancs, force=True)
            if valasz.is_null():
                sorok.append(f"{nev}: nincs valasz")
            elif parancs is obd.commands.STATUS:
                sorok.append(f"{nev}: MIL {'BE' if valasz.value.MIL else 'KI'}, {valasz.value.DTC_count} kod")
            else:
                kodok = [f"{kod} ({leiras})" for kod, leiras in valasz.value]
                sorok.append(f"{nev}: {', '.join(kodok) if kodok else 'nincs tarolt hibakod'}")
        except Exception as hiba:
            sorok.append(f"{nev}: nem olvashato ({hiba})")

    fajlnev = os.path.join(LOG_MAPPA, f"seat_info_{idobelyeg}.txt")
    with open(fajlnev, "w") as fajl:
        fajl.write("\n".join(sorok) + "\n")
    print("\n".join(sorok))
    print(f"Jarmuadatok mentve: {fajlnev}\n")


def main():
    os.makedirs(LOG_MAPPA, exist_ok=True)

    if not os.path.exists(PORT):
        print(f"A {PORT} port nem letezik. Hozd letre eloszor:")
        print("    sudo rfcomm bind rfcomm0 13:E0:2F:8D:58:E1")
        raise SystemExit(1)

    ertekelo, csomag = ertekelo_betoltese()

    print(f"Csatlakozas az OBD-II adapterhez ({PORT})...")
    connection = obd.Async(PORT)
    if not connection.is_connected():
        print("Nem sikerult csatlakozni. Ellenorizd a gyujtast es az rfcomm bind-ot.")
        raise SystemExit(1)

    idobelyeg = f"{datetime.now():%Y%m%d_%H%M%S}"
    jarmu_adatai(connection, idobelyeg)

    for _, parancs in PARANCSOK:
        connection.watch(parancs)
    connection.start()

    fajlnev = os.path.join(LOG_MAPPA, f"seat_log_{idobelyeg}.csv")
    print(f"Naplozas ide: {fajlnev}")
    print("Leallitas: Ctrl+C\n")

    ures_sorozat = 0
    sorok = 0
    rez_osszeg, rez_db = 0.0, 0

    with open(fajlnev, mode="w", newline="") as fajl:
        iro = csv.writer(fajl)
        iro.writerow(["TIME"] + [nev for nev, _ in PARANCSOK])

        try:
            while True:
                most = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ertekek = []
                for _, parancs in PARANCSOK:
                    valasz = connection.query(parancs)
                    ertekek.append("NaN" if valasz.is_null() else valasz.value.magnitude)

                iro.writerow([most] + ertekek)
                fajl.flush()
                os.fsync(fajl.fileno())
                sorok += 1

                if all(e == "NaN" for e in ertekek):
                    ures_sorozat += 1
                    if ures_sorozat == NAN_FIGYELMEZTETES:
                        print(f"FIGYELEM: {NAN_FIGYELMEZTETES} egymas utani ures valasz. Megszakadt a kapcsolat?")
                else:
                    ures_sorozat = 0

                kiiras = "  ".join(
                    f"{nev}:{ertek if ertek == 'NaN' else round(ertek, 1)}"
                    for (nev, _), ertek in zip(PARANCSOK, ertekek)
                )
                if csomag is not None:
                    try:
                        rez = ertekelo.pillanatnyi_reziduum(csomag, dict(zip([n for n, _ in PARANCSOK], ertekek)))
                    except Exception:
                        rez = None
                    if rez is not None:
                        rez_osszeg += rez
                        rez_db += 1
                        kiiras += f"  | Rez:{rez:+.1f}  Atlag:{rez_osszeg / rez_db:+.2f}"
                print(f"[{most}] {kiiras}")

                time.sleep(MINTAVETEL_MP)

        except KeyboardInterrupt:
            print(f"\nLeallitva. {sorok} sor mentve ide: {fajlnev}")
        finally:
            connection.stop()
            connection.close()

    if csomag is not None:
        try:
            eredmeny = ertekelo.ut_ertekelese(fajlnev, csomag)
            if "ewma" in eredmeny:
                print(f"Ut atlagos reziduuma: {eredmeny['reziduum']:+.2f} szazalekpont, "
                      f"EWMA: {eredmeny['ewma']:+.2f} (hatarok: {eredmeny['also']:+.2f} ... {eredmeny['felso']:+.2f})")
            print(f"Szivorendszer allapota: {eredmeny['allapot']}")
        except Exception as hiba:
            print(f"Az ut kiertekelese nem sikerult: {hiba}")


if __name__ == "__main__":
    main()

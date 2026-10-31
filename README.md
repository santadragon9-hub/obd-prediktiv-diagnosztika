# Prediktív diagnosztikai rendszer OBD-II adatok alapján

Szakdolgozat, Debreceni Egyetem Informatikai Kar, 2026.

Egy Raspberry Pi 5 egy ELM327-alapú Bluetooth-adapteren keresztül másodpercenként rögzíti egy Seat Ibiza 6J (2008, 1,4 l, BXW) általános OBD-II adatait. Egy véletlen erdő megtanulja az egészséges motor számított terhelését az üzemi pontból. Minden út után az átlagos eltérést (reziduumot) egy EWMA-kártya figyeli. A rendszer csak olvas, a motorvezérlőbe nem ír.

## Fájlok

| Fájl | Feladat | Fejezet |
| --- | --- | --- |
| `seat_logger.py` | Adatgyűjtő: másodpercenként naplóz, kijelzi a reziduumot, az út végén kiértékel | 3 |
| `seat_elemzo.py` | A naplók beolvasása, egységesítése, tisztítása és leíró elemzése | 4 |
| `seat_osszehasonlitas.py` | Célváltozók és öt tanuló algoritmus összehasonlítása, a modell kiválasztása | 5 |
| `seat_ertekelo.py` | Referenciamodell, utankénti értékelés, EWMA-kártya | 6 |
| `indito.sh` | A Bluetooth-port létrehozása és az adatgyűjtő indítása a Pi-n | 3 |
| `requirements.txt` | A használt Python-csomagok verziói | 3 |

## Telepítés

Raspberry Pi OS (64 bites), Python 3.13:

```
python3 -m venv /home/pi/szakdolg_env
/home/pi/szakdolg_env/bin/pip install -r requirements.txt
```

A szkriptek bármely mappából futtathatók. Ha a `/home/pi/szakdolg_env` környezet létezik, maguktól átváltanak a Pythonjára; máshol elég, ha a csomagok telepítve vannak.

## Használat

Adatgyűjtés az autóban, az adapter párosítása után:

```
./indito.sh
```

Ez létrehozza a `/dev/rfcomm0` soros portot, és elindítja a `seat_logger.py`-t. A napló a `logs/` mappába kerül, a program Ctrl+C-re áll le. Az `indito.sh` az adapter saját Bluetooth-címét tartalmazza; más adapternél ezt kell átírni.

Elemzés, ebben a sorrendben:

```
python3 seat_elemzo.py
python3 seat_osszehasonlitas.py
python3 seat_ertekelo.py
```

Az ábrák az `abrak/`, a modell és az útelőzmények a `modell/` mappába kerülnek; a mappákat a szkriptek hozzák létre.

## Adatok

A szkriptek a `logs/` mappában lévő `seat_log_ÉÉÉÉHHNN_ÓÓPPMM.csv` naplókat dolgozzák fel. A szakdolgozat 39 naplóból készült (2026. március 8. – szeptember 6.). A véletlen magok rögzítettek (`random_state=42`), így a szkriptek ugyanazokból a naplókból ugyanazokat a számokat adják.

# SimpleTrack — nuScenes

Pipeline śledzenia obiektów 3D dla detekcji BEVFusion, MVP i LCF3D:
walidacja wejść, preprocessing, tracking, eksport, ewaluacja i wizualizacje.
Kod wykorzystywanego trackera i konfiguracje są w `third_party/SimpleTrack`.
Wersja bazowa: `05c96bb7ed98fc179856f327544612a66c839b5e`.
Lokalne poprawki opisuje [SIMPLETRACK_UPSTREAM_CHANGES.md](SIMPLETRACK_UPSTREAM_CHANGES.md).
Licencja trackera: [MIT](third_party/SimpleTrack/LICENSE).

## Instalacja

Wymagany Python 3.8 (środowisko zgodne ze starszym nuScenes i SimpleTrack):

```bash
conda env create -f environment-simpletrack.yml
conda activate simpletrack
bash scripts/install.sh
bash scripts/test.sh
```

Alternatywnie w środowisku Python 3.8: `python -m pip install -r requirements.txt`.
Testy jednostkowe nie wymagają datasetu ani GPU. Wizualizacje GUI wymagają
systemowego Tk oraz środowiska graficznego.

## Model i trening

SimpleTrack używa filtra Kalmana i kojarzenia detekcji. Nie trenuje sieci
neuronowej i nie wymaga plików wag. Pełny kod modelu ruchu znajduje się w
`third_party/SimpleTrack/mot_3d/motion_model/kalman_filter.py`, a konfiguracja
eksperymentu w `third_party/SimpleTrack/configs/nu_configs/giou.yaml`.

Trening dotyczy detektorów dostarczających wejściowe detekcje. Ich kod,
środowiska i procedury są w osobnych prywatnych repozytoriach na tym koncie:

- [LCF3D-nuscenes](https://github.com/mmr0z/LCF3D-nuscenes)
- [bevfusion-nuscenes](https://github.com/mmr0z/bevfusion-nuscenes)
- [mvp-nuscenes](https://github.com/mmr0z/mvp-nuscenes)

## Uruchomienie i walidacja

Pobierz nuScenes osobno. Katalog danych powinien zawierać `v1.0-trainval`,
`samples` i `sweeps`. Wymagane są detekcje w formacie nuScenes Detection JSON.

```bash
python scripts/prepare_simpletrack_nuscenes.py /path/to/nuscenes /path/to/preprocessed
python tools/validate_nuscenes_detections.py \
  --input-json /path/to/results_nusc.json --nuscenes-root /path/to/nuscenes \
  --velocity-policy require --report validation.json
python run_simpletrack_nuscenes.py \
  --det-name lcf3d --detections /path/to/results_nusc.json \
  --nuscenes-root /path/to/nuscenes --data-folder /path/to/preprocessed \
  --output ./tracking_results --process 8
```

Ostatnia komenda uruchamia cały pipeline wraz z ewaluacją na `val`.
`--det-name` obsługuje również `bevfusion` i `mvp`.
Jeżeli detekcje nie zawierają prędkości, zastosuj opisaną w instrukcji
politykę `--velocity-policy ignore` konsekwentnie dla wszystkich metod.

## Testy i pełna instrukcja

`bash scripts/test.sh` uruchamia testy walidatora, porównań, wyboru scen
i wizualizacji. Ewaluacja na nuScenes wymaga lokalnych danych; oficjalny
split `test` nie udostępnia publicznych etykiet. Ten pipeline obsługuje
eksperymenty na `val`, a testy jednostkowe są osobnym sprawdzeniem kodu.

[SIMPLETRACK_NUSCENES.md](SIMPLETRACK_NUSCENES.md) zawiera wszystkie etapy,
porównania metod, smoke testy i programy wizualizacji.
Dane nuScenes, wygenerowane wyniki i lokalne środowisko nie są częścią repozytorium.

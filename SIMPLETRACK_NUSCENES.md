# SimpleTrack dla nuScenes validation (2 Hz)

Ten projekt uruchamia jeden niezmienny tracker dla trzech źródeł detekcji:

```text
BEVFusion ─┐
MVP ───────┼─> oficjalny SimpleTrack 2 Hz ─> nuScenes Tracking JSON ─> oficjalny evaluator
LCF3D ─────┘
```

W każdym pełnym eksperymencie używany jest ten sam upstream, split `val`, zestaw
siedmiu klas i domyślny plik
`third_party/SimpleTrack/configs/nu_configs/giou.yaml`. Pipeline nie wykonuje
dodatkowego thresholdingu, NMS, mapowania klas, filtrowania po odległości ani
kalibracji confidence. Pozostaje jedynie standardowe zachowanie upstream
SimpleTrack i jego configu.

## 1. Instalacja

Zalecane jest osobne środowisko Python 3.8:

```bash
conda env create -f environment-simpletrack.yml
conda activate simpletrack
cd third_party/SimpleTrack
pip install -e .
cd ../..
```

Alternatywnie utwórz własne środowisko i zainstaluj
`requirements-simpletrack.txt`, a następnie wykonaj `pip install -e .` z
katalogu `third_party/SimpleTrack`. Nie trzeba modyfikować środowiska
`mmdetection3d`.

Upstream jest przypięty do commita zapisanego w `experiment_info.json`.
Minimalne lokalne poprawki opisuje `SIMPLETRACK_UPSTREAM_CHANGES.md`.

## 2. Przygotowanie nuScenes 2 Hz

Źródłowy katalog powinien zawierać `v1.0-trainval`, `samples` i `sweeps`.
Uruchamiane są wyłącznie oficjalne preprocessory wariantu `--mode 2hz`:

```bash
./scripts/prepare_simpletrack_nuscenes.sh \
  /path/to/nuscenes \
  /path/to/simpletrack_nuscenes_2hz
```

Powstaną `token_info`, `ts_info`, `calib_info`, `ego_info`, `gt_info` i
`pc/raw_pc`. Gotowe komponenty są pomijane. Domyślny `giou.yaml` ma `pc: true`,
więc chmury punktów są wymagane i mogą zajmować dużo miejsca. Żaden katalog
20 Hz nie jest tworzony.

Smoke test jednej sceny:

```bash
python scripts/prepare_simpletrack_nuscenes.py \
  /path/to/nuscenes /path/to/simpletrack_nuscenes_2hz \
  --debug-scene scene-0003
```

## 3. Wejściowe detekcje

Akceptowany jest standardowy nuScenes Detection JSON. Walidator sprawdza:

- obiekt `results` i listy detekcji;
- przynależność każdego tokenu do validation;
- `translation` (3), `size` (3, kolejność `[width, length, height]`),
  `rotation` (4, kolejność `[w, x, y, z]`);
- `detection_name`, skończone `detection_score` i dodatnie wymiary;
- `velocity` (2), NaN/Inf oraz puste/uszkodzone rekordy.

Samodzielne wywołanie:

```bash
python tools/validate_nuscenes_detections.py \
  --input-json /path/to/results_nusc.json \
  --nuscenes-root /path/to/nuscenes \
  --velocity-policy require \
  --report validation.json
```

`require` jest bezpiecznym ustawieniem domyślnym: wszystkie detekcje muszą mieć
rzeczywiste velocity i upstream zapisuje je przez `--velo`. Jeżeli choć jeden z
trzech detektorów ich nie ma, należy po sprawdzeniu wejść uruchomić **wszystkie
trzy** z `--velocity-policy ignore`. Wtedy velocity nie jest generowane i nie
trafia do plików `.npz`. Nie wolno mieszać polityk; narzędzie porównujące wykrywa
taką rozbieżność.

Dodatkowe klasy nuScenes są raportowane i pozostają bez mapowania. SimpleTrack
uruchamia tylko oficjalne klasy tracking: car, bus, trailer, truck, pedestrian,
bicycle i motorcycle.

## 4. Poszczególne etapy

Preprocessing detekcji (oficjalny `detection.py --mode 2hz`):

```bash
python tools/run_simpletrack_detection_preprocess.py \
  --det-name bevfusion \
  --input-json /path/to/bevfusion_results.json \
  --nuscenes-root /path/to/nuscenes \
  --simpletrack-data /path/to/simpletrack_nuscenes_2hz
```

Tracking (wyłącznie oficjalny `tools/main_nuscenes.py`):

```bash
python tools/run_tracking.py \
  --det-name bevfusion \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --result-folder ./tracking_results \
  --process 8
```

Konwersja uruchamia kolejno oficjalne `nuscenes_result_creation.py` i
`nuscenes_type_merge.py`, nigdy ich warianty 10 Hz:

```bash
python tools/convert_tracking_results.py \
  --name BEVFusion_SimpleTrack_2Hz \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --result-folder ./tracking_results
```

Ewaluacja oficjalnym `nuscenes-devkit`:

```bash
python tools/evaluate_tracking.py \
  tracking_results/BEVFusion_SimpleTrack_2Hz/results/tracking_result.json \
  --nuscenes-root /path/to/nuscenes \
  --split val \
  --output-dir tracking_results/BEVFusion_SimpleTrack_2Hz/evaluation
```

## 5. Jedna komenda na detektor

BEVFusion:

```bash
python run_simpletrack_nuscenes.py \
  --det-name bevfusion \
  --detections /path/to/bevfusion_results.json \
  --nuscenes-root /path/to/nuscenes \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --output ./tracking_results --process 8
```

MVP i LCF3D uruchamia się identycznie, zmieniając tylko `--det-name` oraz plik:

```bash
python run_simpletrack_nuscenes.py --det-name mvp \
  --detections /path/to/mvp_results.json \
  --nuscenes-root /path/to/nuscenes \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --output ./tracking_results --process 8

python run_simpletrack_nuscenes.py --det-name lcf3d \
  --detections /path/to/lcf3d_results.json \
  --nuscenes-root /path/to/nuscenes \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --output ./tracking_results --process 8
```

Pipeline wznawia ukończone etapy z `pipeline_state.json`. W razie zmiany pliku
detekcji, configu lub polityki velocity wymaga innego `--output`, co chroni przed
przypadkowym zmieszaniem eksperymentów.

## 6. Wszystkie trzy metody

```bash
python run_all_simpletrack_nuscenes.py \
  --bevfusion /path/to/bevfusion_results.json \
  --mvp /path/to/mvp_results.json \
  --lcf3d /path/to/lcf3d_results.json \
  --nuscenes-root /path/to/nuscenes \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --output ./tracking_results --process 8 \
  --velocity-policy require
```

Po trzech ewaluacjach skrypt automatycznie tworzy porównanie. Można je również
uruchomić ręcznie:

```bash
python tools/compare_tracking_results.py \
  --input bevfusion=tracking_results/BEVFusion_SimpleTrack_2Hz \
  --input mvp=tracking_results/MVP_SimpleTrack_2Hz \
  --input lcf3d=tracking_results/LCF3D_SimpleTrack_2Hz \
  --output-dir tracking_results
```

Powstaną `tracking_comparison.csv`, `.json`, `.md` oraz
`tracking_comparison_per_class.csv`. Porównanie odmawia połączenia manifestów z
różnym hashem configu, splitem, trybem lub polityką velocity.

## 7. Tryb debug jednej sceny

```bash
python run_simpletrack_nuscenes.py \
  --det-name lcf3d --detections /path/to/lcf3d_results.json \
  --nuscenes-root /path/to/nuscenes \
  --data-folder /path/to/simpletrack_nuscenes_2hz \
  --output ./tracking_results --process 1 \
  --debug-scene scene-0003
```

Tryb tworzy kopię wejścia zawierającą wyłącznie tokeny sceny (bez zmiany
wartości detekcji), uruchamia cały tracking i konwersję, a w
`conversion_report.json` zapisuje liczbę klatek, unikalne ID, ID widziane w
wielu klatkach, kontrolę quaternionów, zakres współrzędnych oraz odległość XY do
najbliższej wejściowej detekcji tej samej klasy. Nie oblicza AMOTA
dla fragmentu splitu. Po tym teście należy uruchomić pełne validation.
Nakładka wizualna chmury punktów pozostaje zalecanym ręcznym potwierdzeniem
układu współrzędnych; pipeline nie transformuje wejściowych globalnych boxów.

## 8. Wyniki i metryki

Dla BEVFusion wynik znajduje się w:

```text
tracking_results/BEVFusion_SimpleTrack_2Hz/
  experiment_info.json
  input_validation.json
  pipeline_state.json
  summary/<class>/<scene>.npz
  results/tracking_result.json
  conversion_report.json
  evaluation/metrics_summary.json
  evaluation/metrics_details.json
  evaluation/metrics_readable.json
  evaluation/metrics_global.csv
  evaluation/metrics_per_class.csv
  evaluation/metrics.txt
```

Analogiczne katalogi powstają dla MVP i LCF3D. `metrics_summary.json` i
`metrics_details.json` są pełnymi wynikami oficjalnego evaluatora; eksporty nie
usuwają dodatkowych metryk.

AMOTA i AMOTP są metrykami benchmarku nuScenes Tracking. SimpleTrack tworzy
trajektorie oraz stabilne `tracking_id`, natomiast oficjalny evaluator nuScenes
oblicza AMOTA/AMOTP i pozostałe wartości. Recall mierzy pokrycie obiektów; MOTAR
i MOTA łączą błędy detekcji i asocjacji; MOTP opisuje błąd lokalizacji; IDS to
zmiany identyfikatora; FP i FN to fałszywie dodatnie i pominięte obiekty.

## 9. Wizualizacja i porównanie trajektorii

Skrypt tworzy trzy osobne rzuty BEV, nałożenie metod oraz wspólną planszę 2×2.
Boxy z globalnego Tracking JSON są przeliczane do układu ego bieżącej klatki,
a punkty `LIDAR_TOP` z układu sensora do ego. W układzie ego `x` wskazuje przód,
`y` lewo, a na obrazie przód pojazdu jest zawsze u góry. Transformacja nie
zmienia plików wynikowych. Domyślny
`--min-score 0` nie odrzuca żadnych wyników. Dla wyników debug jednej sceny:

```bash
python tools/visualize_tracking_comparison.py \
  --scene scene-0003 \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results_debug \
  --frame-index 20 \
  --show-gt
```

W `tracking_visualizations/scene-0003` powstaną osobne PNG dla BEVFusion, MVP
i LCF3D, PNG z nałożeniem, plansza porównawcza oraz manifest parametrów.
Animacja całej sceny w natywnym tempie keyframe'ów 2 Hz:

```bash
python tools/visualize_tracking_comparison.py \
  --scene scene-0003 \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results_debug \
  --show-gt --gif --fps 2
```

Przydatne opcje: `--classes car,pedestrian`, `--color-by id`, `--labels`,
`--history 12`, `--view-range 60`, `--no-lidar` oraz `--all-frames`. Filtry
`--classes` i `--min-score` są wyłącznie wizualne i nie modyfikują ani nie
przeliczają eksperymentu. Dla pełnych wyników 150 scen wystarczy zmienić
`--results-root` na `tracking_results` i podać dowolną scenę validation.

Widok z przedniej kamery z oficjalnie rzutowanymi punktami `LIDAR_TOP`, boxami
3D i tym samym układem trzech paneli oraz nałożenia:

```bash
python tools/visualize_tracking_camera.py \
  --scene scene-0003 \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results_debug \
  --camera-channel CAM_FRONT \
  --frame-index 20 \
  --show-gt --gif --fps 2
```

Skrypt obsługuje również `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`,
`CAM_BACK_LEFT` i `CAM_BACK_RIGHT`. Punkty LiDAR są rzutowane funkcją
`nuscenes-devkit map_pointcloud_to_image`, która uwzględnia osobne timestampy,
pozy ego, extrinsics oraz intrinsics kamery. `--no-lidar` pozostawia sam obraz
i boxy, a `--max-depth` steruje skalą koloru głębokości punktów.

Automatyczny wybór 12 zróżnicowanych scen na podstawie opisów nuScenes,
lokalizacji, zagęszczenia klas oraz geometrycznej rozbieżności metod:

```bash
python tools/select_tracking_scenes.py \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results \
  --count 12
```

Raporty `scene_ranking.csv`, `scene_ranking.json`, `selected_scenes.json`,
`selected_scenes.txt` i `selected_scenes.md` trafią do
`tracking_visualizations/scene_selection`. Jest to ranking diagnostyczny, nie
oficjalna metryka nuScenes. Rozbieżność korzysta z dopasowania Hungarian boxów
tej samej klasy w odległości globalnej XY do 2 m. Aby po selekcji wygenerować
wyłącznie GIF-y BEV i `CAM_FRONT`, bez pomocniczych PNG, dodaj
`--generate-gifs`. Oba renderery przyjmują też samodzielnie `--gif-only`.

Dookólny układ wszystkich sześciu kamer, zapisany osobno dla każdego modelu
(bez nakładania BEVFusion/MVP/LCF3D):

```bash
python tools/visualize_tracking_surround.py \
  --scene scene-0012 --scene scene-1060 --scene scene-0798 \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results \
  --show-gt --gif-only --fps 2
```

Każdy GIF ma układ `FRONT_LEFT | FRONT | FRONT_RIGHT` w pierwszym wierszu oraz
`BACK_LEFT | BACK | BACK_RIGHT` w drugim. Wyniki trafiają do
`tracking_visualizations/surround/<scene>/<method>/`.

Trójwymiarowy widok samych trajektorii, osobno dla modeli:

```bash
python tools/visualize_tracking_trajectories_3d.py \
  --scene scene-0012 --scene scene-1060 --scene scene-0798 \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results \
  --z-axis time --color-by id --gif-only
```

Domyślne `--z-axis time` tworzy wykres czasoprzestrzenny X/Y/czas, który dobrze
pokazuje długość, fragmentację i ciągłość `tracking_id`. `--z-axis height`
przełącza wykres na dosłowne współrzędne przestrzenne XYZ. Przerwy między
klatkami nie są sztucznie łączone. `--min-track-length` i `--min-score` są
wyłącznie filtrami wizualnymi i nie zmieniają Tracking JSON ani metryk.

Osobne GIF-y BEV i `CAM_FRONT` dla każdego wybranego modelu (bez łączenia
widoków, bez nakładania metod i bez LCF3D):

```bash
python tools/visualize_tracking_individual_models.py \
  --scene-file tracking_visualizations/scene_selection/selected_scenes.txt \
  --nuscenes-root /path/to/nuscenes \
  --results-root tracking_results \
  --method bevfusion --method mvp \
  --show-gt --gif-only --fps 2
```

Wyniki trafiają do `tracking_visualizations/individual_models/<scene>/<method>/`.
Skrypt ładuje wyłącznie wskazane Tracking JSON-y, więc powyższe wywołanie nie
korzysta z wyników LCF3D.

### Aplikacja okienkowa do eksportu PNG/GIF

Uruchom aplikację w środowisku SimpleTrack:

```bash
.venv-simpletrack/bin/python tools/tracking_visualization_gui.py
```

Okno pokazuje wyłącznie 150 scen splitu `val`. Można zaznaczyć wiele scen,
modeli oraz niezależnych widoków: BEV i sześć kamer nuScenes. PNG jest
domyślnym formatem; pole `Klatka` wybiera indeks keyframe'u zapisywanego do
PNG. Opcjonalne zaznaczenie GIF zapisuje dodatkowo animację całej sceny.
Opcja `Oznacz TP / FP / FN` dodaje do obrazu informację o obiektach
dopasowanych, nadmiarowych predykcjach i niewykrytych obiektach GT, także z
podziałem na klasy. Jest to diagnostyczne, klasowe dopasowanie centrów XY z
domyślnym progiem 2 m; nie zastępuje oficjalnego evaluatora i nie zmienia
Tracking JSON ani metryk.
Szczegółowe listy dopasowanych par, FP i FN są dodatkowo zapisywane w
`<scene>/<method>/detection_status.json`; liczniki dotyczą całego keyframe'u
360°, natomiast na widoku kamery rysowane są tylko obiekty widoczne w danej
kamerze.
Każda kombinacja scena/model/widok trafia do osobnego pliku w katalogu
`tracking_visualizations/gui_exports/<scene>/<method>/` (lub w katalogu
wskazanym w oknie). Modele ani widoki nie są łączone na jednym panelu.

## 10. Reprodukowalność i znane problemy

`experiment_info.json` zapisuje detektor, wejście, hash configu, commit
SimpleTrack, Python, wersje zależności (w tym `nuscenes-devkit`), datę, split,
2 Hz, procesy, politykę velocity i wszystkie argumenty.

- Upstream `giou.yaml` zawiera `running.has_velo: true`, ale implementacja
  `MOTModel.has_velo` wyznacza użycie velocity z `motion_model`; dla domyślnego
  `kf` velocity detektora nie jest wstrzykiwane do filtra. Nie zmieniono tego,
  aby zachować oficjalny algorytm.
- Upstream `setup.py` instaluje `mot_3d`, lecz nie pakuje katalogu `data_loader`,
  który importuje `main_nuscenes.py`. Wrapper dodaje checkout SimpleTrack do
  `PYTHONPATH`; nie zmienia to kodu ani danych trackera.
- Oficjalny konwerter SimpleTrack wpisuje `[0.0, 0.0]` do obowiązkowego pola
  velocity wynikowego Tracking JSON. Jest to zachowanie upstream, nie estymacja
  dodana przez wrapper.
- `nuscenes-devkit==1.1.11` oczekuje wewnętrznego API `motmetrics==1.1.3`.
  `motmetrics 1.4` zmienia format bufora zdarzeń i nie jest z nim zgodny.
  Na Pythonie 3.10 lokalny launcher przywraca jedynie przeniesiony alias
  `collections.Iterable`; obliczenia nadal wykonuje oficjalny evaluator.
- Pełny evaluator `val` wymaga dokładnie wszystkich sample tokenów validation;
  nie należy używać go do pojedynczej sceny.
- Boxy wejściowe muszą już być w globalnym układzie nuScenes. Wrapper nie zamienia
  wymiarów, quaternionów ani współrzędnych.

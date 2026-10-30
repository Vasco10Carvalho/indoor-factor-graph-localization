# Localização no hospital — farmácia / run1

Este ensaio compara seis trajetórias no mapa fixo da farmácia:

- Factor Graph 2D;
- AMCL no YAML/PGM fixo;
- SLAM Toolbox em modo de localização, com pose graph congelado;
- Cartographer em pure localization, com `pbstream` congelado;
- KISS-ICP + Cartographer em pure localization;
- KISS-ICP isolado, apenas como baseline de odometria.

O bag de teste é `${HOSPITAL_DATA_ROOT}/pharmacy/run1`. O mapa ROS é
`farmacia_map_min1p60.yaml`, que referencia o PGM pedido, define resolução de
0,05 m/pixel e origem `[-58.15, -50.5, 0]`. Os nós ROS devem receber o YAML,
não o PGM isolado.

## Mapas nativos usados

O SLAM Toolbox e o Cartographer não conseguem carregar diretamente o
YAML/PGM como estado de localização. Usam estados nativos já existentes e
independentes do bag de teste:

```text
SLAM Toolbox:
$SLAM_STATE_BASE.posegraph
$SLAM_STATE_BASE.data

Cartographer:
$CARTOGRAPHER_STATE
```

Não se usa `pharmacy/run2` para gerar estes estados: a inspeção espacial
mostrou que cobre apenas cerca de 31% do percurso de `run1` a uma distância de
1 m, pelo que não constitui um mapa de referência completo para este teste.

## Pose e instante iniciais

O teste começa aos 90 s do bag, depois do período inicial em que o robô está
praticamente parado. A pose no mapa fixo é:

```text
x = -17.984100 m
y = -37.044571 m
yaw = 107.677028 graus
```

Devido à diferença de referencial, a pose equivalente no `pbstream` do
Cartographer é:

```text
x = 14.352065 m
y = -11.626481 m
yaw = 138.548619 graus
```

Esta semente é obrigatória neste mapa: sem ela, o Cartographer pode escolher
um corredor geometricamente semelhante, produzindo uma localização errada.
O estado do SLAM Toolbox já tem a origem no arranque pretendido, por isso usa
`[0, 0, 0]` no seu referencial nativo.

## 1. Preparar o ambiente

```bash
source /opt/ros/jazzy/setup.bash
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
MAPPING_WS="${MAPPING_WS:-$HOME/mapping_ws}"
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
PHARMACY_MAP_YAML="/caminho/absoluto/para/farmacia_map_min1p60.yaml"
SLAM_STATE_BASE="/caminho/absoluto/para/my_map_final"
CARTOGRAPHER_STATE="/caminho/absoluto/para/pharma_run3_carto.pbstream"
REFERENCE_ROOT="${REFERENCE_ROOT:-$HOME/localization_reference_maps}"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
FG_RUN_ROOT="${FG_RUN_ROOT:-$HOME/indoor_factor_graph_runs}"

source "$ROS_WS/install/setup.bash"
source "$MAPPING_WS/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTDDS_BUILTIN_TRANSPORTS=UDPv4

cd "$REPO_ROOT/trajectory_comparison"
mkdir -p "$REFERENCE_ROOT/pharmacy_independent"
```

## 2. Executar o Factor Graph 2D

```bash
./run_factor_graph_pose2_pharmacy.sh \
  "$HOSPITAL_DATA_ROOT/pharmacy/run1" \
  "$PHARMACY_MAP_YAML" \
  "$FG_RUN_ROOT/pharmacy_run1_factor_graph2d" \
  --initial-pose -17.984100 -37.044571 107.677028 \
  --start-offset 90 \
  --rate 0.5 \
  --domain-id 97
```

O wrapper seleciona automaticamente `configs/pharmacy_pose2.yaml`. O script
usa `/ouster/points` para criar o `LaserScan`,
`/odometry/filtered` como odometria e `/ouster/imu` como IMU disponível no
pipeline Pose2. O Factor Graph Humble/GTSAM corre no contentor existente; o
resto da preparação e a reprodução do bag correm em Jazzy.

## 3. Executar os outros cinco métodos

```bash
./run_all_localization_methods.sh \
  "$HOSPITAL_DATA_ROOT/pharmacy/run1" \
  "$PHARMACY_MAP_YAML" \
  "$REFERENCE_ROOT/pharmacy_independent" \
  "$RUN_ROOT/pharmacy_run1_localization" \
  --start-offset 90 \
  --rate 0.5 \
  --initial-x -17.984100 \
  --initial-y -37.044571 \
  --initial-yaw-deg 107.677028 \
  --slam-state-base "$SLAM_STATE_BASE" \
  --cartographer-state "$CARTOGRAPHER_STATE" \
  --cartographer-start-x 14.352065 \
  --cartographer-start-y -11.626481 \
  --cartographer-start-yaw-deg 138.548619
```

Os cinco métodos são executados sequencialmente e cada um reproduz o trecho
restante do bag. Juntando o Factor Graph, o ensaio completo deve demorar cerca
de 1 h 45 min a `--rate 0.5`, mais a inicialização dos nós.

Para repetir resultados existentes, adicionar `--overwrite` aos dois comandos.

## 4. Gerar a figura, CSV alinhado e métricas

```bash
./build_localization_comparison.sh \
  "$PHARMACY_MAP_YAML" \
  "$FG_RUN_ROOT/pharmacy_run1_factor_graph2d/trajectory.csv" \
  "$RUN_ROOT/pharmacy_run1_localization" \
  "$RUN_ROOT/pharmacy_run1_localization/comparison" \
  "Comparação de trajetórias — hospital / farmácia / run1"
```

O alinhamento de cada resultado externo é uma única transformação SE(2)
pela primeira pose e não altera escala nem corrige drift. As diferenças para o
Factor Graph medem concordância entre métodos, não erro absoluto, porque este
bag não tem ground truth externo certificado.

## 5. Publicar tudo no RViz

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=98
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
PHARMACY_MAP_YAML="/caminho/absoluto/para/farmacia_map_min1p60.yaml"
cd "$REPO_ROOT/trajectory_comparison"

python3 publish_comparison.py \
  --map-yaml "$PHARMACY_MAP_YAML" \
  --trajectories "$RUN_ROOT/pharmacy_run1_localization/comparison/trajectories_aligned.csv"
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=98
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
rviz2 -d "$REPO_ROOT/trajectory_comparison/trajectory_comparison.rviz"
```

## Cores

- azul-escuro: Factor Graph 2D;
- roxo: AMCL;
- ciano: SLAM Toolbox Localization;
- verde: Cartographer Pure Localization;
- vermelho: KISS-ICP + Cartographer Pure Localization;
- laranja: KISS-ICP Odometry.

No RViz, cada método é um display `Path` independente e pode ser
ligado/desligado através do respetivo checkbox no painel `Displays`.

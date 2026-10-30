# Comparação apenas de localização

Esta variante separa o benchmark de localização do ensaio anterior de SLAM.
O bag `urgency/run2` é usado exclusivamente para construir os mapas nativos e
o bag `urgency/run1` é reservado para teste.

Métodos comparados:

- Factor Graph 2D num occupancy map fixo;
- AMCL no mesmo YAML/PGM fixo;
- SLAM Toolbox em `mode: localization`, sobre pose graph congelado;
- Cartographer em pure localization, sobre `pbstream` congelado;
- KISS-ICP + Cartographer em pure localization;
- KISS-ICP como baseline de odometria sem mapa.

O Factor Graph e o AMCL usam diretamente
`urgencia_map_min1p60_edit.yaml`. SLAM Toolbox e Cartographer precisam dos seus
formatos nativos, criados com o `run2`; esses mapas não são atualizados durante
o teste. No pós-processamento, todas as trajetórias recebem somente um
alinhamento SE(2) pela primeira pose para serem publicadas no mapa editado.

O argumento passado aos nós ROS deve ser o ficheiro YAML (não o PGM isolado):
o YAML já referencia `urgencia_map_min1p60_edit.pgm` e contém a resolução e a
origem do mapa. Na gravação das localizações por TF, as amostras só começam
0,5 s de tempo ROS depois do primeiro `/scan`, evitando TFs transitórias durante
o carregamento dos mapas nativos.

## 1. Ambiente

```bash
source /opt/ros/jazzy/setup.bash
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
MAPPING_WS="${MAPPING_WS:-$HOME/mapping_ws}"
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
REFERENCE_ROOT="${REFERENCE_ROOT:-$HOME/localization_reference_maps}"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
FG_RUN_ROOT="${FG_RUN_ROOT:-$HOME/indoor_factor_graph_runs}"
URGENCY_MAP_YAML="/caminho/absoluto/para/urgencia_map_min1p60_edit.yaml"

source "$ROS_WS/install/setup.bash"
source "$MAPPING_WS/install/setup.bash"
cd "$REPO_ROOT/trajectory_comparison"
```

## 2. Criar os mapas nativos com `run2`

Teste curto da infraestrutura:

```bash
./prepare_all_reference_maps.sh \
  "$HOSPITAL_DATA_ROOT/urgency/run2" \
  "$REFERENCE_ROOT/urgency_run2_smoke" \
  --duration 30 --rate 0.5
```

Preparação completa para a comparação:

```bash
./prepare_all_reference_maps.sh \
  "$HOSPITAL_DATA_ROOT/urgency/run2" \
  "$REFERENCE_ROOT/urgency_run2" \
  --rate 0.5
```

Outputs principais:

```text
urgency_run2/slam_toolbox/urgency_reference.posegraph
urgency_run2/slam_toolbox/urgency_reference.data
urgency_run2/cartographer/urgency_reference.pbstream
```

## 3. Localizar no `run1`

```bash
./run_all_localization_methods.sh \
  "$HOSPITAL_DATA_ROOT/urgency/run1" \
  "$URGENCY_MAP_YAML" \
  "$REFERENCE_ROOT/urgency_run2" \
  "$RUN_ROOT/urgency_run1_localization" \
  --start-offset 90 --rate 0.5
```

Os valores iniciais usados por omissão são a primeira pose do resultado do
Factor Graph: `(5.330157484, 45.834921258, -96.878981676 deg)`. Podem ser
alterados com `--initial-x`, `--initial-y` e `--initial-yaw-deg`.

## 4. Gerar a comparação

```bash
./build_localization_comparison.sh \
  "$URGENCY_MAP_YAML" \
  "$FG_RUN_ROOT/urgency_run1/trajectory.csv" \
  "$RUN_ROOT/urgency_run1_localization" \
  "$RUN_ROOT/urgency_run1_localization/comparison"
```

## 5. Ver no RViz

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=98
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
URGENCY_MAP_YAML="/caminho/absoluto/para/urgencia_map_min1p60_edit.yaml"
cd "$REPO_ROOT/trajectory_comparison"

python3 publish_comparison.py \
  --map-yaml "$URGENCY_MAP_YAML" \
  --trajectories "$RUN_ROOT/urgency_run1_localization/comparison/trajectories_aligned.csv"
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=98
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
rviz2 -d "$REPO_ROOT/trajectory_comparison/trajectory_comparison.rviz"
```

## Cores

- azul: Factor Graph 2D;
- roxo: AMCL;
- ciano: SLAM Toolbox Localization;
- verde: Cartographer Pure Localization;
- vermelho: KISS-ICP + Cartographer Pure Localization;
- laranja: KISS-ICP Odometry.

## Interpretação

O KISS-ICP isolado não é um localizador e não publica `map -> base_link`; é
mantido como controlo de drift odométrico. As diferenças para o Factor Graph
são métricas de concordância entre métodos e não erro absoluto, porque não há
ground truth externo associado ao bag.

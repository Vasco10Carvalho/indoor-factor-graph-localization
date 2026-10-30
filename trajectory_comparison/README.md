# Comparação de trajetórias indoor

Para a comparação em que todos os métodos usam mapas fixos, consultar
[`README_LOCALIZATION.md`](README_LOCALIZATION.md).

Para repetir essa comparação no mapa e no bag da farmácia, incluindo as
poses iniciais e os estados nativos corretos, consultar
[`README_PHARMACY_LOCALIZATION.md`](README_PHARMACY_LOCALIZATION.md).

Esta pasta permite repetir no mesmo bag e desenhar no mesmo mapa:

- Factor Graph 2D (resultado de referência já existente);
- SLAM Toolbox, alimentado pelo `LaserScan` e pelo TF `odom -> base_link` do bag;
- KISS-ICP, alimentado pela point cloud Ouster 3D;
- Cartographer 2D, alimentado pelo mesmo `LaserScan` e por
  `/odometry/filtered` do bag;
- KISS-ICP + Cartographer, em que a odometria do Cartographer vem do KISS-ICP.

Os quatro novos ensaios usam a mesma gravação, o mesmo instante inicial e a
mesma geometria `base_link -> os_sensor -> os_lidar`. Os métodos 2D usam o
mesmo corte de altura da point cloud. Por omissão, este corte é `0.60–0.80 m`,
igual ao usado no ensaio do Factor Graph 2D do hospital.

## O que significa o alinhamento

Cada trajetória externa começa num referencial local arbitrário. Para a colocar
no mapa do hospital, `compare_trajectories.py` aplica apenas uma transformação
rígida SE(2): a primeira pose do método é alinhada com a pose do Factor Graph
interpolada no mesmo timestamp. Não há correção de escala nem ajuste usando a
trajetória completa. Assim, o drift continua visível.

As diferenças para o Factor Graph são métricas de concordância entre métodos;
não são erro absoluto, porque o Factor Graph não é ground truth.

## Preparação no ROS 2 Jazzy

```bash
source /opt/ros/jazzy/setup.bash
ROS_WS="${ROS_WS:-$HOME/ros2_ws}"
MAPPING_WS="${MAPPING_WS:-$HOME/mapping_ws}"
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
FG_RUN_ROOT="${FG_RUN_ROOT:-$HOME/indoor_factor_graph_runs}"
URGENCY_MAP_YAML="/caminho/absoluto/para/urgencia_map.yaml"

source "$ROS_WS/install/setup.bash"
source "$MAPPING_WS/install/setup.bash"

cd "$REPO_ROOT/trajectory_comparison"
```

Os scripts não contêm estes caminhos e podem ser copiados para outro PC. Nesse
caso, basta fazer `source` dos workspaces equivalentes antes de os executar.

## Teste curto dos quatro métodos

Este comando processa 30 segundos a partir dos 90 segundos do bag:

```bash
./run_all_methods.sh \
  "$HOSPITAL_DATA_ROOT/urgency/run1" \
  "$RUN_ROOT/urgency_run1_smoke" \
  --start-offset 90 --duration 30 --rate 1.0
```

## Ensaio completo

Cada método tem de observar o bag completo, pelo que o comando faz quatro
reproduções sequenciais. Com `--rate 1.0`, demora aproximadamente quatro vezes
a duração restante do bag.

```bash
./run_all_methods.sh \
  "$HOSPITAL_DATA_ROOT/urgency/run1" \
  "$RUN_ROOT/urgency_run1_full" \
  --start-offset 90 --rate 1.0
```

Para repetir e substituir resultados existentes, acrescente `--overwrite`.
Também é possível executar um método de cada vez:

```bash
./run_method.sh kiss_icp \
  "$HOSPITAL_DATA_ROOT/urgency/run1" \
  "$RUN_ROOT/urgency_run1_full/kiss_icp" \
  --start-offset 90 --rate 1.0
```

Cada diretório contém `trajectory.csv`, `run_summary.txt`, o comando exato de
reprodução do bag e os logs de todos os nós.

## Criar a figura e as métricas

```bash
./build_comparison.sh \
  "$URGENCY_MAP_YAML" \
  "$FG_RUN_ROOT/urgency_run1/trajectory.csv" \
  "$RUN_ROOT/urgency_run1_full" \
  "$RUN_ROOT/urgency_run1_full/comparison"
```

Outputs:

- `trajectories_aligned.csv`: todas as poses já no referencial do mapa;
- `trajectory_comparison_all_methods.png`: figura sobre o mapa 2D;
- `trajectory_comparison_metrics.json`: comprimento, fecho do percurso e
  diferenças temporais relativamente ao Factor Graph. Para não favorecer nem
  penalizar métodos que publicam a frequências diferentes, `path_length_m` e os
  erros são calculados nos timestamps do Factor Graph; `raw_path_length_m`
  conserva o comprimento à frequência original de cada método.

## Ver tudo no RViz

Terminal 1:

```bash
source /opt/ros/jazzy/setup.bash
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
RUN_ROOT="${RUN_ROOT:-$HOME/trajectory_comparison_runs}"
URGENCY_MAP_YAML="/caminho/absoluto/para/urgencia_map.yaml"
cd "$REPO_ROOT/trajectory_comparison"

python3 publish_comparison.py \
  --map-yaml "$URGENCY_MAP_YAML" \
  --trajectories "$RUN_ROOT/urgency_run1_full/comparison/trajectories_aligned.csv"
```

Terminal 2:

```bash
source /opt/ros/jazzy/setup.bash
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
rviz2 -d "$REPO_ROOT/trajectory_comparison/trajectory_comparison.rviz"
```

O publisher disponibiliza um `nav_msgs/Path` separado em
`/trajectory_comparison/<metodo>/path`. A configuração RViz apresenta seis
displays independentes, com checkbox e cor fixa para cada método; não usa um
`MarkerArray` agregado.

## Limitações conhecidas

- Esta primeira comparação regista a estimativa online publicada por cada
  método. Não extrai retroativamente todos os nós já corrigidos do pose graph
  final do Cartographer ou do SLAM Toolbox.
- A trajetória do Factor Graph é usada como referência temporal e de pose
  inicial, não como ground truth certificado.
- Os parâmetros dos métodos são os do pipeline existente e um perfil base do
  SLAM Toolbox; ainda não foram afinados especificamente para este percurso.
- A IMU do Ouster não é usada nesta comparação 2D. O KISS-ICP usa a point cloud
  3D e o seu deskew; Cartographer e SLAM Toolbox usam o corte `LaserScan`.

# Factor Graph 2D na farmácia

O matcher Pose2 principal inclui um pequeno prior de movimento no custo do
`scan-to-map`. Este prior foi validado na farmácia e na urgência e desempata
correspondências quase equivalentes em corredores compridos.

## Construção

```bash
cd "$HOME/indoor-factor-graph-localization"
./pipeline.sh build
./pipeline.sh check
```

Os pesos ficam no perfil YAML. Colocar
`scan_translation_prior_weight: 0.0` e
`scan_rotation_prior_weight: 0.0` reproduz o scorer antigo sem regularização.

## Farmácia, run1

```bash
source /opt/ros/jazzy/setup.bash
REPO_ROOT="$HOME/indoor-factor-graph-localization"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
BAG_DIR="$HOSPITAL_DATA_ROOT/pharmacy/run1"
MAP_YAML="$HOME/maps/mapas_hospital_novos/farmacia/farmacia_map.yaml"
OUTPUT_DIR="$HOME/indoor_factor_graph_runs/pharmacy_run1_factor_graph2d_motion_prior_full"
cd "$REPO_ROOT/trajectory_comparison"

./run_factor_graph_pose2_pharmacy.sh \
  "$BAG_DIR" \
  "$MAP_YAML" \
  "$OUTPUT_DIR" \
  --initial-pose -17.984100 -37.044571 107.677028 \
  --start-offset 90 \
  --rate 1.0 \
  --domain-id 117 \
  --min-height 0.60 \
  --max-height 0.80
```

O lançador de conveniência seleciona automaticamente
`configs/pharmacy_pose2.yaml`. É possível usar outro perfil passando
explicitamente `--config CAMINHO`.

Para repetir numa pasta onde já existe `trajectory.csv`, acrescentar
`--overwrite`. O `domain-id` pode ser alterado se esse domínio ROS já estiver
em uso.

## Resultado validado

O ensaio completo do `run1` produziu 1251 poses. Relativamente à trajetória
AMCL usada apenas como referência de comparação, obteve erro de posição
mediano de 0,209 m, percentil 95 de 1,333 m e erro final de 0,195 m. Foram
aceites 1246 correspondências `scan-to-map`, rejeitadas 3 e aceites 2 loop
closures.

O resultado encontra-se em:

```text
$HOME/indoor_factor_graph_runs/pharmacy_run1_factor_graph2d_motion_prior_full
```

## Comparação no RViz

Depois de gerar `trajectories_aligned.csv` com
`build_localization_comparison.sh`, publicar o mapa e as trajetórias:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=118
REPO_ROOT="$HOME/indoor-factor-graph-localization"
MAP_YAML="$HOME/maps/mapas_hospital_novos/farmacia/farmacia_map.yaml"
COMPARISON_DIR="$HOME/trajectory_comparison_runs/pharmacy_run1_localization/comparison_factor_graph_pharmacy_v2"

python3 "$REPO_ROOT/trajectory_comparison/publish_comparison.py" \
  --map-yaml "$MAP_YAML" \
  --trajectories "$COMPARISON_DIR/trajectories_aligned.csv"
```

Noutro terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=118
rviz2 -d "$HOME/indoor-factor-graph-localization/trajectory_comparison/trajectory_comparison.rviz"
```

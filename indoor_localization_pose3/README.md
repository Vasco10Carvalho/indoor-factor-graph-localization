# Factor Graph Indoor Pose3/6D

Esta implementação é uma extensão independente da versão Pose2 existente em
`indoor_localization/`. Não altera o node 2D e permite comparar os dois
resultados.

Cada keyframe é `Xi = map_T_base_link` em `SE(3)`:

```text
Prior Pose3 -> X0 -- odometria Pose3 -- X1 -- odometria Pose3 -- X2 ...
                 |                     |                         |
          ICP 3D robusto        ICP 3D robusto            ICP 3D robusto
                 \____________ loop Pose3/DCS ______________/
```

Não existe qualquer dependência de GNSS ou AMCL. Também não é adicionado um
fator IMU raw nesta fase, porque `/odometry/filtered` já contém orientação 3D e
não foi encontrada a configuração original de `robot_localization` que permita
garantir independência estatística.

## Processamento

Para cada PointCloud2 selecionado como keyframe, o node:

1. lê XYZ e o tempo `t` de cada ponto;
2. interpola a odometria em SE(3) e faz deskew para o início do scan;
3. filtra NaN, alcance e Z, faz voxel downsampling e limita os pontos;
4. prevê a pose com o delta de odometria;
5. recorta um submapa local do PLY com uma KD-tree 2D e limites Z;
6. executa ICP point-to-plane coarse e fine;
7. rejeita o resultado por fitness, RMSE, correspondências ou correção 6D;
8. cria `PriorFactorPose3` com Huber e sigmas adaptativos separados;
9. valida revisitas antigas com ICP 3D e cria `BetweenFactorPose3` com DCS;
10. atualiza iSAM2 incrementalmente.

As sigmas seguem a convenção GTSAM
`[roll, pitch, yaw, x, y, z]`.

## Auditoria do novo bag da Urgência

Bag validado: `${HOSPITAL_DATA_ROOT}/urgency/run1`, em que
`HOSPITAL_DATA_ROOT` deve apontar para a raiz local dos dados do hospital.

- duração: 1106,825 s;
- `/ouster/points`: 9598 mensagens, frame `os_lidar`, 64x1024 pontos;
- cada nuvem tem `t` em nanossegundos, aproximadamente de 0 a 100 ms;
- `/odometry/filtered`: 55342 mensagens, `odom -> base_link`, cerca de 50 Hz;
- `/ouster/imu`: sem orientação válida (`orientation_covariance[0] = -1`);
- existem `/tf` e `/tf_static`, mas não existe `/scan` nem GNSS;
- o bag contém `os_sensor -> os_lidar`: Z=0,03618 m e yaw=180 graus;
- falta `base_link -> os_sensor`.

O valor usado abaixo para `base_link_T_lidar.z` é provisório: 0,565 m de uma
configuração hospitalar anterior mais 0,03618 m gravados no bag. Deve ser
substituído por uma medição/calibração do robô.

O primeiro scan estacionário foi alinhado ao PLY a partir do ponto marcado no
mapa. O melhor resultado foi:

```text
base_link = [5,250, 45,863, 0,068 m, -0,22°, 0,62°, -94,60°]
fitness   = 0,837
RMSE      = 0,087 m
```

Os limites do PLY e do occupancy map 2D também são compatíveis. Isto constitui
evidência geométrica de alinhamento, mas não uma transformação formal declarada
nos metadados.

## Preparação reproduzível do mapa LiScan

O script [`tools/prepare_urgencia_icp_map.py`](../tools/prepare_urgencia_icp_map.py)
reproduz a preparação usada neste ensaio. O PLY original é aberto apenas para
leitura; todos os resultados são criados no diretório de saída e nenhum ficheiro
da origem é movido ou apagado.

O processamento recentra as coordenadas projetadas num frame local, aplica
voxel de 3 cm e um filtro estatístico conservador, recorta o interior, desloca
o piso de `Z=2,3175 m` para aproximadamente zero e preserva piso, paredes e teto.
O teto é importante para restringir `Z`, roll e pitch no ICP 3D. São gerados um
mapa de 5 cm, uma variante de 10 cm, JSON com a transformação e uma imagem de
verificação.

Exemplo, escrevendo para uma pasta nova:

```bash
SOURCE_PLY="/caminho/absoluto/para/urgencia_original.ply"
ICP_MAP_ROOT="${ICP_MAP_ROOT:-$HOME/indoor_factor_graph_maps}"

python3 tools/prepare_urgencia_icp_map.py \
  "$SOURCE_PLY" \
  "$ICP_MAP_ROOT/urgencia"
```

O mapa de 5 cm validado neste computador tem 1.467.089 pontos e limites
`X=[-53,473; 52,519]`, `Y=[12,000; 69,993]` e
`Z=[-0,167; 2,633] m`.

## Comandos para `urgency/run1`

Definir primeiro os caminhos locais; nenhum deles fica codificado no
repositório. Depois, construir uma vez:

```bash
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
URGENCY_MAP_PLY="/caminho/absoluto/para/urgencia_3d_icp_map_0p05.ply"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/indoor_factor_graph_runs}"
cd "$REPO_ROOT"

./pose3_pipeline.sh build
./pose3_pipeline.sh check
```

Terminal 1, iniciar o factor graph:

```bash
./pose3_pipeline.sh run \
  --map "$URGENCY_MAP_PLY" \
  --output "$OUTPUT_ROOT/urgency_run1_pose3_6d" \
  --config configs/hospital_pose3.yaml \
  --initial-pose 5.250 45.863 0.068 -0.22 0.62 -94.60 \
  --extrinsic 0 0 0.60118 0 0 180 \
  --domain-id 76
```

Terminal 2, RViz:

```bash
./pose3_pipeline.sh rviz --domain-id 76
```

Terminal 3, diagnóstico:

```bash
./pose3_pipeline.sh monitor --domain-id 76
```

Terminal 4, primeiro teste de 60 segundos:

```bash
./pose3_pipeline.sh play \
  --bag "$HOSPITAL_DATA_ROOT/urgency/run1" \
  --duration 60 \
  --domain-id 76
```

Depois de parar o node com `Ctrl+C`:

```bash
./pose3_pipeline.sh evaluate \
  --output "$OUTPUT_ROOT/urgency_run1_pose3_6d"
```

Para reproduzir o bag completo, repetir sem `--duration 60`.

## Validação medida em ROS 2 Humble

O `run1` completo foi reproduzido à velocidade original (`--rate 1.0`). O teste
processou 1.817 keyframes ao longo de 1.104,90 s e conservou a parte final em
que o robô regressa ao local inicial.

- 1.816 medições ICP aceites e uma rejeitada: 99,945% de aceitação;
- fitness mediana: 0,884 e percentil 5: 0,795;
- RMSE mediana: 0,086 m e percentil 95: 0,103 m;
- correção aceite máxima: 0,152 m e 7,39 graus;
- 15 loop closures Pose3 aceites;
- trajetória otimizada: 244,84 m, com salto máximo de 0,373 m;
- `Z`: -0,039 a 0,142 m;
- roll absoluto máximo: 2,45 graus;
- pitch absoluto máximo: 4,28 graus;
- ponto de retorno mais próximo do início: 0,002 m;
- pose final a 0,638 m da pose inicial.

Não houve keyframes sem submapa disponível nesta execução. A odometria raw
terminou a 38,43 m do início, enquanto a combinação ICP + grafo recuperou o
regresso e fechou a trajetória. Sem ground truth independente, os valores de
retorno e drift são indicadores de consistência, não erro absoluto.

## Trajetórias publicadas

- **vermelho**, `/indoor_pose3/path`: estados finais otimizados pelo factor graph;
- **laranja**, `/indoor_pose3/odometry_path`: odometria raw alinhada ao prior inicial;
- **cinzento**, `/indoor_pose3/predicted_path`: previsão do grafo antes da correção ICP;
- **azul**, `/indoor_pose3/scan_to_map_path`: medições ICP aceites, ainda antes da fusão com os restantes fatores;
- **esferas vermelhas**, `/indoor_pose3/rejected_scan_markers`: candidatos ICP calculados mas rejeitados pelos gates;
- **ciano**, `/indoor_pose3/pose2_aid_path`: auxílio Pose2 usado, vazio quando o Pose3 funciona sozinho;
- `/indoor/factor_graph/path`: versão Pose2, opcional no mesmo RViz;
- **magenta**, `/indoor_pose3/loops`: loop closures aceites;
- `/indoor_pose3/map_points`: cópia visual do PLY limitada por
  `map_publish_z_min/max`. No perfil do hospital é publicada só até `Z=2,15 m`
  para retirar o teto. O ICP e o factor graph continuam a usar o mapa carregado
  até `map_z_max=3,20 m`, incluindo o teto;
- `/indoor_pose3/axes`: eixos XYZ da base;
- `/indoor_pose3/status`: diagnóstico JSON.

O vermelho e o azul podem ficar quase sobrepostos quando o ICP é bom: o azul é
a medição scan-to-map e o vermelho é o resultado depois de essa medição ser
fundida com odometria, prior e eventuais loops. Não representam segmentos bons
e maus da mesma trajetória.

Depois de terminar o bag, todas as trajetórias guardadas podem ser republicadas
em conjunto, sem voltar a executar o factor graph:

```bash
./pose3_pipeline.sh publish \
  --map "$URGENCY_MAP_PLY" \
  --output "$OUTPUT_ROOT/urgency_run1_pose3_full_rate1" \
  --domain-id 76
```

Noutro terminal, o RViz offline deve usar tempo de parede:

```bash
./pose3_pipeline.sh rviz --domain-id 76 --live
```

O publicador lê o PLY e o CSV em modo read-only e volta a publicar o mapa sem
teto, a trajetória otimizada, a odometria alinhada, as previsões, os ICP aceites,
os candidatos rejeitados, o auxílio Pose2 disponível e os loop closures.

O CSV guarda os mesmos componentes para comparação offline. O avaliador gera
JSON e PNG com aceitação ICP, fitness/RMSE, correspondências, correções, p50/p95,
saltos, Z/roll/pitch e indicadores de retorno. Sem ground truth, estes últimos
são métricas de consistência, não erro absoluto.

## Farmácia: `pharmacy/run1` + `pharmacy/run2`

O mapa da farmácia foi criado a partir de um PLY LiScan original, fornecido ao
script através do argumento `SOURCE_PLY`.
O PLY original não é alterado nem removido. A cópia fixa recomendada para ICP é:

```text
$ICP_MAP_ROOT/pharmacy/pharmacy_3d_icp_map_0p05.ply
```

O perfil [`configs/pharmacy_pose3.yaml`](../configs/pharmacy_pose3.yaml) usa o
mapa 3D completo até `Z=4,20 m` no ICP, incluindo o teto. A publicação para o
RViz termina em `Z=3,10 m`, retirando o teto da visualização na zona da
farmácia sem alterar o mapa usado pelo factor graph.

O alinhamento dos primeiros scans com o PLY mediu a pose inicial
`[-14,019; -85,224; 1,014; 0,008; 0,056; 104,961 graus]`, com fitness `0,909`
e RMSE `0,073 m`. O resultado e a imagem de validação são escritos em
`$OUTPUT_ROOT/pharmacy_alignment_6d/`.

Terminal 1, iniciar uma única instância do factor graph para os dois bags:

```bash
REPO_ROOT="${REPO_ROOT:-$HOME/indoor-factor-graph-localization}"
HOSPITAL_DATA_ROOT="/caminho/absoluto/para/hospital"
ICP_MAP_ROOT="${ICP_MAP_ROOT:-$HOME/indoor_factor_graph_maps}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/indoor_factor_graph_runs}"
PHARMACY_MAP_PLY="$ICP_MAP_ROOT/pharmacy/pharmacy_3d_icp_map_0p05.ply"
cd "$REPO_ROOT"

./pose3_pipeline.sh run \
  --map "$PHARMACY_MAP_PLY" \
  --output "$OUTPUT_ROOT/pharmacy_pose3_6d" \
  --config configs/pharmacy_pose3.yaml \
  --initial-pose -14.019 -85.224 1.014 0.008 0.056 104.961 \
  --extrinsic 0 0 0.60118 0 0 180 \
  --domain-id 77
```

Terminal 2, abrir o RViz:

```bash
./pose3_pipeline.sh rviz --domain-id 77
```

Terminal 3, reproduzir primeiro `run1` completo e, quando terminar, `run2` no
mesmo node. Não usar `--start-offset`, porque a pose inicial corresponde ao
primeiro scan real de `run1`:

```bash
./pose3_pipeline.sh play \
  --bag "$HOSPITAL_DATA_ROOT/pharmacy/run1" \
  --rate 1.0 \
  --domain-id 77

./pose3_pipeline.sh play \
  --bag "$HOSPITAL_DATA_ROOT/pharmacy/run2" \
  --rate 1.0 \
  --domain-id 77
```

`run2` continua a odometria de `run1`, portanto o factor graph não deve ser
reiniciado entre os dois comandos. No final, parar o node do Terminal 1 com
`Ctrl+C` e avaliar:

```bash
./pose3_pipeline.sh evaluate \
  --output "$OUTPUT_ROOT/pharmacy_pose3_6d"
```

Para voltar a publicar o resultado guardado, sem executar os bags:

```bash
./pose3_pipeline.sh publish \
  --map "$PHARMACY_MAP_PLY" \
  --output "$OUTPUT_ROOT/pharmacy_pose3_6d" \
  --map-z-max 3.10 \
  --domain-id 77

./pose3_pipeline.sh rviz --domain-id 77 --live
```

Foi validado um ensaio desde o início de `run1` até à saída da farmácia: 152
keyframes, 100% dos ICP aceites, fitness mediana `0,878`, RMSE mediana
`0,076 m`, correção máxima `0,047 m` e nenhuma zona sem submapa. Esta validação
curta confirma o arranque e o alinhamento; as métricas finais devem ser geradas
depois da reprodução integral dos dois bags.

## Portabilidade e Jazzy

Não são usados `.env` nem caminhos fixos no código. Caminhos e poses são
argumentos; tópicos e tuning ficam em `configs/hospital_pose3.yaml`.

O Docker Humble é atualmente o caminho validado e reprodutível. O computador
tem ROS 2 Jazzy nativo e quase todas as dependências, mas ainda lhe falta GTSAM
no Python do ROS. A execução nativa pode ser acrescentada depois de instalar e
testar `python3-gtsam`; não é necessária para este ensaio.

## Limitações atuais

- o extrínseco base-LiDAR ainda é uma aproximação;
- não existe ground truth independente para esta corrida;
- o loop closure só pode ser validado num trecho que regresse a uma zona antiga;
- os thresholds são conservadores e devem ser revistos após 60 s e depois após
  uma secção com movimento;
- a comparação Pose2 simultânea requer gerar `/scan`, pois o novo bag só traz
  PointCloud2.

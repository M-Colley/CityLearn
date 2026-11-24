# Datasets de preços e carbono

`citylearn.price_carbon` escreve `pricing.csv` e `carbon_intensity.csv` para um intervalo de tempo, na resolução do dataset. Há três fontes: preços do mercado diário e mix de geração do ENTSO-E para uma zona de oferta, a API pública Carbon Intensity para a Grã-Bretanha e um modelo de despacho PyPSA para um cenário futuro. O `CityLearnEnv` não importa o módulo, pelo que não há custo na simulação. O módulo escreve ficheiros; aponte as entradas `pricing` e `carbon_intensity` de um esquema para eles.

```console
pip install "citylearn[price_carbon]"   # entsoe-py, PyPSA, HiGHS
```

## Saída

| Ficheiro | Colunas | Unidade |
|---|---|---|
| `pricing.csv` | `electricity_pricing`, `electricity_pricing_predicted_1/2/3` | moeda/kWh |
| `carbon_intensity.csv` | `carbon_intensity` | kgCO2/kWh |
| `price_carbon_metadata.json` | fonte, unidades, fatores, primeiro e último passo (UTC), contagens | |

`carbon_intensity.csv` não tem colunas previstas porque o CityLearn lê apenas `carbon_intensity`.

As três colunas de preço previsto contêm o preço real `--horizons` horas à frente (por omissão 1, 2 e 3 horas): uma previsão perfeita. Os últimos passos repetem o último preço. Substitua-as para avaliar controladores com previsões imperfeitas.

A grelha é contínua no tempo absoluto de `--start` (interpretado em `--tz`, salvo se tiver desvio) até `--end` (exclusivo). Um dia local com mudança de hora tem por isso 23 ou 25 passos horários. Os dados nativos de 15, 30 ou 60 minutos são promediados ou mantidos na grelha. Até `--max-gap-steps` (por omissão 3) passos em falta são interpolados; uma lacuna maior é um erro. Os preços podem ser negativos e mantêm-se.

## ENTSO-E (zonas de oferta da UE)

Peça uma chave gratuita em transparency.entsoe.eu e exporte-a. A ferramenta lê-a apenas de `ENTSOE_API_KEY` e nunca a regista nem a escreve.

```console
export ENTSOE_API_KEY=...
python -m citylearn.price_carbon entsoe --zone DE_LU --start 2024-01-01 --end 2025-01-01 \
    --tz Europe/Berlin --resolution-minutes 60 --out data/my_dataset
```

Os preços são EUR/MWh a dividir por 1000. A intensidade de carbono é `sum(geração_i * fator_i) / sum(geração_i)` sobre os tipos de geração com fator, com as medianas de ciclo de vida do IPCC AR5 WG3 Anexo III (Tabela A.III.2, gCO2eq/kWh): carvão 820, gás 490, biomassa 230, geotérmica 38, hídrica 24, nuclear 12, solar fotovoltaica (utility) 48, eólica em terra 11, eólica no mar 12. A produção de bombagem conta como hídrica e o consumo líquido é limitado a zero. Petróleo, resíduos, turfa, marinha e tipos "outros" não têm mediana AR5 e ficam de fora; os metadados listam-nos com a sua quota, e a ferramenta pára quando esta excede 10 % da geração. O resultado é uma média baseada na produção, que ignora importações e não é uma intensidade marginal.

## Grã-Bretanha (sem chave)

```console
python -m citylearn.price_carbon gb --start 2024-01-01 --end 2025-01-01 --resolution-minutes 30 --out data/gb
```

A ferramenta chama `api.carbonintensity.org.uk/intensity` em blocos de 13 dias. `--field actual` (por omissão) recorre à previsão quando o valor real é nulo. Os valores são médias nacionais em gCO2/kWh, convertidas para kgCO2/kWh. O comando escreve apenas `carbon_intensity.csv`.

## Cenários futuros (PyPSA)

Um ficheiro YAML define instantes, barramentos, uma carga, geradores com custo marginal (moeda/MWh) e `emission_factor` (gCO2/kWh), unidades de armazenamento e ligações opcionais. O HiGHS resolve o despacho de custo mínimo. O preço é o preço marginal do balanço do barramento. A intensidade de carbono é as emissões despachadas a dividir pela energia despachada (`intensity_basis: system`, ou `bus` para os geradores do barramento de preço). Um gerador de corte de carga mantém o modelo admissível; a ferramenta avisa quando o usa.

```console
python -m citylearn.price_carbon scenario --config examples/price_carbon/scenario.yaml --out data/scenario_2035
```

Veja `examples/price_carbon/scenario.yaml` e a docstring do módulo `citylearn/price_carbon/scenario.py` para as chaves. Os perfis vêm de um CSV (`file`, `column`), de uma lista repetida (`values`) ou de uma série sintética com semente (`seed`, `mean`, `sd`). O preço marginal de um sistema de um só nó é o custo do gerador marginal, pelo que preços de escassez só surgem quando o custo de corte ou uma capacidade saturada os fixa.

## Python

```python
from citylearn.price_carbon.entsoe import entsoe_datasets
from citylearn.price_carbon.common import write_outputs

pricing, carbon, grid, meta = entsoe_datasets('DE_LU', '2024-01-01', '2025-01-01', 60, 'Europe/Berlin')
write_outputs('data/my_dataset', pricing, carbon, meta, grid)
```

# Envelhecimento da bateria com BLAST-Lite

Perda de capacidade semiempírica opcional para baterias de edifícios e de VE. Substitui o modelo linear
`capacity_loss_coefficient` de uma bateria por um modelo de célula do [BLAST-Lite](https://github.com/NatLabRockies/BLAST-Lite)
que depende do SOC, da temperatura, da profundidade de descarga, da taxa C e do tempo. Sem a definição nada muda: os KPIs e as
saídas da simulação são idênticos bit a bit aos de uma execução sem esta funcionalidade.

Versão em inglês: [../battery_degradation.md](../battery_degradation.md).

## Instalar

```console
python -m pip install "citylearn[degradation]"
```

O BLAST-Lite fixa `numpy<2`, pelo que o extra restringe o NumPy em todo o ambiente.

## Configurar

Adicione `degradation` aos atributos de `electrical_storage` ou da `battery` de um VE no schema:

```json
"electrical_storage": {
  "type": "citylearn.energy_model.Battery",
  "attributes": {
    "capacity": 6.4,
    "nominal_power": 5.0,
    "degradation": {
      "model": "blast_lite",
      "cell": "Lfp_Gr_250AhPrismatic",
      "update_interval_steps": 24,
      "temperature": "outdoor_dry_bulb"
    }
  }
}
```

| Chave | Omissão | Significado |
| --- | --- | --- |
| `model` | `blast_lite` | O único modelo. |
| `cell` | `Nmc111_Gr_Kokam75Ah_Battery` | Nome da classe em `blast.models`. `citylearn.degradation.available_cells()` lista-as. LFP: `Lfp_Gr_250AhPrismatic`, `Lfp_Gr_SonyMurata3Ah_Battery`. NMC: `Nmc111_Gr_Kokam75Ah_Battery`, `NMC_Gr_75Ah_A`, `Nmc622_Gr_DENSO50Ah_Battery`, `Nmc811_GrSi_LGM50_5Ah_Battery`. |
| `update_interval_steps` | 24 | Passos entre atualizações do estado do BLAST-Lite. |
| `temperature` | 25,0 | Constante em [C], ou `outdoor_dry_bulb` para a meteorologia do edifício. A bateria de um VE só aceita uma constante. |
| `degradation_scalar` | 1,0 | Multiplicador da taxa de degradação do modelo. |

Chaves desconhecidas, células desconhecidas e `update_interval_steps < 1` levantam `ValueError`.

## O que a bateria faz

- Regista o SOC no fim de cada passo, como fração da capacidade atual, com a temperatura e o tempo.
- A cada `update_interval_steps` passos entrega o lote a `update_battery_state`. A última amostra inicia o lote seguinte, pelo que nenhum intervalo se perde.
- A capacidade é `capacity * q`, com `q` do BLAST-Lite. Mantém-se constante entre atualizações, e `degraded_capacity`, `capacity_history` e `capacity_before_step` leem-na como antes.
- O CityLearn não tem modelo de resistência, pelo que o aumento de resistência do BLAST-Lite não é aplicado.
- Os passos depois do último intervalo completo de um episódio não entram na capacidade. `reset()` inicia um novo modelo BLAST-Lite.
- Entre amostras o BLAST-Lite interpola o SOC linearmente. Uma bateria de VE que não é carregada num passo não contribui com amostra para esse passo.

Use `update_interval_steps` igual ou superior a 2. O BLAST-Lite 1.1.3 calcula a taxa C de um lote com `np.trapz` sobre as taxas
dos intervalos sem o primeiro intervalo, pelo que um lote de um passo tem taxa C 0 e a perda depende do intervalo.
Está proposta uma correção a montante.

## Custo

`scripts/manual/benchmark_battery_charge.py` mede `Battery.charge` ao longo de 8760 passos. Numa máquina Windows
carregada, melhor de três, Python 3.12:

| Modelo | us por passo |
| --- | --- |
| linear (omissão) | 115 |
| BLAST-Lite, intervalo 168 | 95 |
| BLAST-Lite, intervalo 24 | 130 |
| BLAST-Lite, intervalo 1 | 1144 |

Uma atualização do BLAST-Lite custa cerca de 0,7 a 0,8 ms por chamada, pelo que o custo por passo é esse valor dividido pelo intervalo.

## Verificação com o PyBaMM

`scripts/manual/validate_degradation_pybamm.py` compara o BLAST-Lite com uma simulação SPMe do PyBaMM com crescimento da
SEI limitado pela difusão do solvente, para um ciclo de funcionamento. É offline e não faz parte do caminho de execução:

```console
python -m pip install pybamm
python scripts/manual/validate_degradation_pybamm.py --cycles 100 --c-rate 0.5 --temperature 25
```

Para a célula LG M50 (`Nmc811_GrSi_LGM50_5Ah_Battery` e `Chen2020` do PyBaMM), 100 ciclos completos a C/2 e 25 C com uma hora a plena carga:

| Modelo | Perda de capacidade |
| --- | --- |
| PyBaMM SPMe + SEI | 0,24 % |
| BLAST-Lite | 4,60 % |

A diferença é de 4,4 pontos percentuais. Os parâmetros da SEI do PyBaMM não estão ajustados a dados de envelhecimento, enquanto o
BLAST-Lite está ajustado a envelhecimento medido de células, pelo que o script indica a ordem de grandeza e não calibra nenhum dos modelos.

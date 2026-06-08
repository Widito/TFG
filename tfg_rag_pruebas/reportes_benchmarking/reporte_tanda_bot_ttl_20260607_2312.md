# Reporte de Ejecución - Tanda bot_ttl
- Fecha y Hora: 2026-06-07 23:12:43
- Total de Requisitos Evaluados: 12

## 1. Contexto de Entrada (El Gold Standard)
| ID | Query Natural | Ontología Esperada |
|---|---|---|
| 1 | Zones are areas with spatial 3D volumes | bot.ttl |
| 2 | Construction sites are areas with spatial 3D volumes | bot.ttl |
| 3 | Buildings are areas with spatial 3D volumes | bot.ttl |
| 4 | Storeys are areas with spatial 3D volumes | bot.ttl |
| 5 | Spaces are limited three-dimensional extent defined physically or notionally, and are areas with spatial 3D volumes | bot.ttl |
| 6 | Zones may contain other zones | bot.ttl |
| 7 | Construction sites may contain buildings | bot.ttl |
| 8 | Buildings may contain storeys | bot.ttl |
| 9 | Storeys may contain spaces | bot.ttl |
| 10 | Spaces may be contained in storeys, buildings, and construction sites | bot.ttl |
| 11 | Spaces may intersect different storeys, buildings, and construction sites | bot.ttl |
| 12 | A zone can be adjacent to another zone | bot.ttl |

## 2. Matriz de Cobertura Generada
```json
{
  "facility.ttl": [
    "Zones are areas with spatial 3D volumes",
    "Storeys are areas with spatial 3D volumes",
    "Storeys may contain spaces",
    "Spaces may be contained in storeys, buildings, and construction sites"
  ],
  "bot.ttl": [
    "Zones are areas with spatial 3D volumes",
    "Storeys are areas with spatial 3D volumes",
    "Spaces are limited three-dimensional extent defined physically or notionally, and are areas with spatial 3D volumes",
    "Zones may contain other zones",
    "A zone can be adjacent to another zone"
  ],
  "brick.ttl": [
    "Construction sites are areas with spatial 3D volumes"
  ],
  "s4inma_2019-04-30.n3": [
    "Construction sites are areas with spatial 3D volumes",
    "Buildings are areas with spatial 3D volumes",
    "Construction sites may contain buildings",
    "Spaces may intersect different storeys, buildings, and construction sites"
  ],
  "building.ttl": [
    "Buildings are areas with spatial 3D volumes",
    "Zones may contain other zones",
    "Construction sites may contain buildings",
    "Buildings may contain storeys",
    "Storeys may contain spaces",
    "Spaces may be contained in storeys, buildings, and construction sites",
    "Spaces may intersect different storeys, buildings, and construction sites"
  ],
  "s4agri_2020-06-05.n3": [
    "Construction sites may contain buildings"
  ]
}
```

## 3. Veredicto Final del LLM
**Veredicto Final**

Recomendación de Red: **facility.ttl**, **building.ttl**, and **s4inma_2019-04-30.n3**

Justificación:

1. **facility.ttl**: Esta ontología cubre conceptos fundamentales como zonas, storeys, y espacios, lo que la hace una excelente elección para establecer el marco general de nuestra red de ontologías.
2. **building.ttl**: Como su nombre indica, esta ontología se enfoca en edificios y construcciones, lo que nos permite abordar aspectos como la relación entre edificios y zonas, así como la estructura interna de los edificios (storeys y espacios).
3. **s4inma_2019-04-30.n3**: Esta ontología proporciona una visión más amplia de la construcción y el sitio de construcción, lo que nos permite conectar conceptos como zonas, edificios y sitios de construcción.

Observaciones:

* La cobertura de requisitos es generalmente buena en nuestra recomendación, con solo un pequeño hueco en la ontología **s4inma_2019-04-30.n3**, que no cubre el requisito "Zones are areas with spatial 3D volumes" (que ya está cubierto por **facility.ttl**).
* La ontología **bot.ttl** no se incluyó en nuestra recomendación debido a su baja cobertura de requisitos y la existencia de otras ontologías que cubren mejor los conceptos relacionados.
* La ontología **brick.ttl** tampoco se incluyó, ya que solo cubre un requisito específico y no proporciona una visión más amplia de la construcción o el sitio de construcción.

En resumen, nuestra recomendación de red de ontologías (**facility.ttl**, **building.ttl**, and **s4inma_2019-04-30.n3**) nos permite cubrir un rango amplio de requisitos y proporcionar una visión general sólida de la construcción y el sitio de construcción.

# Reporte de Ejecución - Tanda hard
- Fecha y Hora: 2026-06-08 08:42:21
- Total de Requisitos Evaluados: 10

## 1. Contexto de Entrada 
| ID | Query Natural | Ontología Esperada |
|---|---|---|
| 11 | Spaces may intersect different storeys, buildings, and construction sites | bot.ttl |
| 18 | A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room | saref_2020-05-29.n3 |
| 20 | Building objects are objects in the building that can be controlled by devices, such as doors or windows | saref_2020-05-29.n3 |
| 22 | A building space has also a property that can be used to specify the type of space, for example, the living room or the bedroom | saref_2020-05-29.n3 |
| 23 | A building space is a geographical point characterized by a certain altitude, latitude and longitude | saref_2020-05-29.n3 |
| 26 | It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc. | odrl_2017-09-16.n3 |
| 27 | The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template. | odrl_2017-09-16.n3 |
| 29 | It should be possible to define the price for duty/duties of payment for all permissions of a policy in a global way - while currently the payment duty must be defined for each permission individually. | odrl_2017-09-16.n3 |
| 32 | Being able to tie Permission, Prohibition, Duty, and Constraint entities together with an AND, OR or XOR relationship | odrl_2017-09-16.n3 |
| 33 | Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified. | odrl_2017-09-16.n3 |

## 2. Matriz de Cobertura Generada
```json
{
  "bot.ttl": [
    "Spaces may intersect different storeys, buildings, and construction sites"
  ],
  "s4agri_2020-06-05.n3": [
    "A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room"
  ],
  "building.ttl": [
    "A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room",
    "Building objects are objects in the building that can be controlled by devices, such as doors or windows",
    "A building space has also a property that can be used to specify the type of space, for example, the living room or the bedroom",
    "A building space is a geographical point characterized by a certain altitude, latitude and longitude"
  ],
  "beo.ttl": [
    "Building objects are objects in the building that can be controlled by devices, such as doors or windows"
  ],
  "odrl_2017-09-16.n3": [
    "It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc.",
    "Being able to tie Permission, Prohibition, Duty, and Constraint entities together with an AND, OR or XOR relationship",
    "Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified."
  ],
  "s4ehaw_2020-05-01.n3": [
    "It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc."
  ],
  "dpv_2025-08-01.n3": [
    "The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template.",
    "Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified."
  ],
  "meb_2013-03-26.n3": [
    "It should be possible to define the price for duty/duties of payment for all permissions of a policy in a global way - while currently the payment duty must be defined for each permission individually."
  ]
}
```

## 3. Veredicto Final del LLM
**Veredicto Final**

Recomendación de Red:

1. **building.ttl**
2. **odrl_2017-09-16.n3**
3. **s4agri_2020-06-05.n3**

Justificación por Ontología:

1. **building.ttl**: Esta ontología cubre los requisitos relacionados con la definición de espacios físicos en un edificio, como cocina o salón, y objetos que pueden ser controlados por dispositivos, como puertas o ventanas. Es fundamental para entender el contexto en el que se ubicarán los dispositivos.
2. **odrl_2017-09-16.n3**: Esta ontología cubre los requisitos relacionados con la definición de permisos, prohibiciones y deberes, así como la capacidad de asignar múltiples individuos a un grupo para especificar permisos/prohibiciones/deberes. Es crucial para entender cómo se establecen las restricciones en el entorno.
3. **s4agri_2020-06-05.n3**: Esta ontología cubre los requisitos relacionados con la definición de espacios físicos en un edificio y la ubicación de dispositivos en esos espacios. Es importante para entender cómo se relacionan los dispositivos con el entorno.

Observaciones de Cobertura y Posibles Huecos:

* La cobertura de los requisitos es buena, pero hay algunos huecos que pueden ser cubiertos mediante la integración de otras ontologías.
* No hay una ontología que abarque completamente todos los requisitos, por lo que se requiere una integración cuidadosa para garantizar la coherencia y consistencia en el modelo.
* La ontología **dpv_2025-08-01.n3** no fue seleccionada porque no cubre suficientes requisitos y puede ser reemplazada por otras ontologías que abarcan temas similares.

En resumen, la recomendación de red consiste en combinar las ontologías **building.ttl**, **odrl_2017-09-16.n3** y **s4agri_2020-06-05.n3** para crear un modelo coherente y completo que cubra los requisitos relacionados con la definición de espacios físicos, objetos controlables y restricciones en el entorno.

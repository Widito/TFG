# Reporte de Ejecución - Tanda todos
- Fecha y Hora: 2026-06-08 08:35:40
- Total de Requisitos Evaluados: 36

## 1. Contexto de Entrada 
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
| 13 | A device performs one or more functions | saref_2020-05-29.n3 |
| 14 | Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine | saref_2020-05-29.n3 |
| 15 | A device shall have a model property | saref_2020-05-29.n3 |
| 16 | A device shall have a manufacturer property | saref_2020-05-29.n3 |
| 17 | A device can optionally have a description | saref_2020-05-29.n3 |
| 18 | A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room | saref_2020-05-29.n3 |
| 19 | A building space contains devices or building objects | saref_2020-05-29.n3 |
| 20 | Building objects are objects in the building that can be controlled by devices, such as doors or windows | saref_2020-05-29.n3 |
| 21 | A building object can be opened or closed by an actuator | saref_2020-05-29.n3 |
| 22 | A building space has also a property that can be used to specify the type of space, for example, the living room or the bedroom | saref_2020-05-29.n3 |
| 23 | A building space is a geographical point characterized by a certain altitude, latitude and longitude | saref_2020-05-29.n3 |
| 24 | The devices can be classified into categories: FunctionRelated, EnergyRelated and BuildingRelated | saref_2020-05-29.n3 |
| 25 | A single option for applying a constraint to a party (of a specific role) should be defined. | odrl_2017-09-16.n3 |
| 26 | It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc. | odrl_2017-09-16.n3 |
| 27 | The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template. | odrl_2017-09-16.n3 |
| 28 | In addition to the existing identifiers of a policy means for expressing a version of this policy should specified. | odrl_2017-09-16.n3 |
| 29 | It should be possible to define the price for duty/duties of payment for all permissions of a policy in a global way - while currently the payment duty must be defined for each permission individually. | odrl_2017-09-16.n3 |
| 30 | Extend the temporal constraints by setting a time reference, any period in the rightOperand refers to this point in time. | odrl_2017-09-16.n3 |
| 31 | For a relativeTimePeriod constraint the rightOperand has to provide a time period as value. | odrl_2017-09-16.n3 |
| 32 | Being able to tie Permission, Prohibition, Duty, and Constraint entities together with an AND, OR or XOR relationship | odrl_2017-09-16.n3 |
| 33 | Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified. | odrl_2017-09-16.n3 |
| 34 | Ability to link from a Policy or a Permission to the original (text-based) license. | odrl_2017-09-16.n3 |
| 35 | It should be possible to define policies of type Assertion. | odrl_2017-09-16.n3 |
| 36 | An Assertion policy does not grant any permissions/prohibitions, but reflects policy terms that a party believes to have. | odrl_2017-09-16.n3 |

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
    "Construction sites are areas with spatial 3D volumes",
    "Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine",
    "A building object can be opened or closed by an actuator",
    "The devices can be classified into categories: FunctionRelated, EnergyRelated and BuildingRelated"
  ],
  "building.ttl": [
    "Construction sites are areas with spatial 3D volumes",
    "Buildings are areas with spatial 3D volumes",
    "Zones may contain other zones",
    "Construction sites may contain buildings",
    "Buildings may contain storeys",
    "Storeys may contain spaces",
    "Spaces may be contained in storeys, buildings, and construction sites",
    "Spaces may intersect different storeys, buildings, and construction sites",
    "A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room",
    "A building space contains devices or building objects",
    "Building objects are objects in the building that can be controlled by devices, such as doors or windows",
    "A building space has also a property that can be used to specify the type of space, for example, the living room or the bedroom",
    "A building space is a geographical point characterized by a certain altitude, latitude and longitude"
  ],
  "beo.ttl": [
    "Construction sites are areas with spatial 3D volumes",
    "Building objects are objects in the building that can be controlled by devices, such as doors or windows"
  ],
  "th_building.owl": [
    "Buildings are areas with spatial 3D volumes"
  ],
  "s4agri_2020-06-05.n3": [
    "Construction sites may contain buildings",
    "A building space defines the physical spaces of the building where a device is located, such as a kitchen or a living room",
    "A building space contains devices or building objects"
  ],
  "s4inma_2019-04-30.n3": [
    "Construction sites may contain buildings",
    "Spaces may intersect different storeys, buildings, and construction sites",
    "A device performs one or more functions",
    "A device shall have a model property",
    "A building space contains devices or building objects",
    "The devices can be classified into categories: FunctionRelated, EnergyRelated and BuildingRelated"
  ],
  "saref_2020-05-29.n3": [
    "A device performs one or more functions",
    "Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine",
    "A device shall have a model property"
  ],
  "s4watr_2020-06-03.n3": [
    "Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine"
  ],
  "mep.ttl": [
    "Examples of devices are a light switch, a temperature sensor, an energy meter, a washing machine",
    "A building object can be opened or closed by an actuator"
  ],
  "frapo_2014-01-31.n3": [
    "A device shall have a model property"
  ],
  "s4ehaw_2020-05-01.n3": [
    "A device shall have a manufacturer property"
  ],
  "iot-lite_2015-06-01.n3": [
    "A device can optionally have a description"
  ],
  "ifc4.ttl": [
    "A device can optionally have a description",
    "It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc.",
    "For a relativeTimePeriod constraint the rightOperand has to provide a time period as value."
  ],
  "odrl_2017-09-16.n3": [
    "A single option for applying a constraint to a party (of a specific role) should be defined.",
    "It should be possible to define the target of a constraint inside a Permission, Prohibition or Duty: e.g. a constraint of human age applies to the Assignee, a constraint of play time applies to Assets (of type audio or video), etc.",
    "Being able to tie Permission, Prohibition, Duty, and Constraint entities together with an AND, OR or XOR relationship",
    "Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified.",
    "It should be possible to define policies of type Assertion.",
    "An Assertion policy does not grant any permissions/prohibitions, but reflects policy terms that a party believes to have."
  ],
  "m4i_2025-03-10.n3": [
    "The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template.",
    "Extend the temporal constraints by setting a time reference, any period in the rightOperand refers to this point in time."
  ],
  "gc_2009-08-24.n3": [
    "The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template."
  ],
  "modsci_2022-12-18.n3": [
    "The current data model assumes a policy instance includes all required data explicitly. This should be extended to policy instances which include explicit data and variables for values which are defined by parameters provided by an access to this template."
  ],
  "th_sharedvocalulary.owl": [
    "In addition to the existing identifiers of a policy means for expressing a version of this policy should specified."
  ],
  "seo_2019-04-10.n3": [
    "In addition to the existing identifiers of a policy means for expressing a version of this policy should specified.",
    "Ability to link from a Policy or a Permission to the original (text-based) license."
  ],
  "meb_2013-03-26.n3": [
    "It should be possible to define the price for duty/duties of payment for all permissions of a policy in a global way - while currently the payment duty must be defined for each permission individually."
  ],
  "ei2a_2021-10-21.n3": [
    "Extend the temporal constraints by setting a time reference, any period in the rightOperand refers to this point in time.",
    "For a relativeTimePeriod constraint the rightOperand has to provide a time period as value."
  ],
  "s4city_2020-06-05.n3": [
    "Extend the temporal constraints by setting a time reference, any period in the rightOperand refers to this point in time."
  ],
  "swc_2009-05-11.n3": [
    "Being able to assign multiple individuals of type Party to a Group for which permissions/prohibitions/duties can be specified."
  ],
  "ccrel.rdf": [
    "Ability to link from a Policy or a Permission to the original (text-based) license."
  ],
  "dpv_2025-08-01.n3": [
    "An Assertion policy does not grant any permissions/prohibitions, but reflects policy terms that a party believes to have."
  ]
}
```

## 3. Veredicto Final del LLM
**Veredicto Final**

Recomendación de Red:

1. **facility.ttl**
2. **building.ttl**
3. **mep.ttl**

Justificación:

* **facility.ttl**: Esta ontología cubre conceptos relacionados con la infraestructura y los espacios físicos, como zonas, estancias y construcciones. Es fundamental para entender las relaciones entre los objetos y espacios en el contexto de la web semántica.
* **building.ttl**: Esta ontología proporciona una visión más detallada de los edificios y sus componentes, incluyendo espacios físicos, dispositivos y objetos que pueden ser controlados. Es crucial para entender las relaciones entre los objetos y espacios en el contexto de la web semántica.
* **mep.ttl**: Esta ontología se enfoca en los dispositivos y objetos que pueden ser controlados, como interruptores, sensores y medidores de energía. Es fundamental para entender cómo los dispositivos interactúan con los espacios físicos y las construcciones.

Observaciones:

* La cobertura de requisitos es generalmente buena en estas tres ontologías, especialmente en lo que respecta a la infraestructura y los objetos físicos.
* Sin embargo, hay algunos requisitos no cubiertos, como el concepto de "party" (parte) en la ontología **odrl_2017-09-16.n3**, que podría ser relevante para entender las relaciones entre los objetos y espacios en el contexto de la web semántica.
* La cobertura de requisitos puede ser mejorada mediante la integración de otras ontologías, como **bot.ttl** o **s4inma_2019-04-30.n3**, que se enfocan en los dispositivos y objetos que pueden ser controlados.

En resumen, la recomendación de red consiste en combinar las ontologías **facility.ttl**, **building.ttl** y **mep.ttl** para obtener una cobertura más amplia de requisitos relacionados con la infraestructura, los objetos físicos y los dispositivos que pueden ser controlados.

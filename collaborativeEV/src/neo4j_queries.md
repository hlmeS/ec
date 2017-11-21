# Creating Sample Data Set #

It's best to run all at once to avoid the need for ```MATCH``` statements.

```java

CREATE (p1:Person {name: 'Sally', tel: '202-1234'})
CREATE (p2:Person {name: 'John', tel: '345-8989'})

CREATE (s1:Space {type: '20 footer', volume:33.2, lat: '23.456 N', lon: '156.65 W'})
CREATE (s2:Space {type: '20 footer', volume:33.2, lat: '23.456 N', lon: '156.65 W'})
CREATE (s3:Space {type: '40 footer', volume:67.7, lat: '22.456 N', lon: '156.65 W'})
CREATE (s4:Space {type: 'office', volume: 75.3})
CREATE (s5:Space {type: 'trailer', volume: 4.02})

CREATE (a1:AC {brand: 'LEZETi', solar: 'yes', ducted: 'no', power: 885, btu: 11500})
CREATE (a2:AC {brand: 'LEZETi', solar: 'yes', ducted: 'yes', power: 885, btu: 11500})
CREATE (a3:AC {brand: 'Hotspot', solar: 'yes', ducted: 'yes', power: 885, btu: 11500})
CREATE (a4:AC {brand: 'Hotspot', solar: 'no',ducted: 'no',  power: 885, btu: 11500})
CREATE (a5:AC {brand: 'Hessaire', solar: 'no', ducted: 'no', btu: 12000})

CREATE (i1:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'})
CREATE (i2:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'})
CREATE (i3:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'})
CREATE (i4:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'})
CREATE (i5:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'})

CREATE (o1:Object {type: 'papaya', setpoint: 54})
CREATE (o2:Object {type: 'flowers', setpoint: 40})
CREATE (o3:Object {type: 'wine', setpoint: 55})
CREATE (o4:Object {type: 'produce', setpoint: 48})

CREATE (p1) - [:OWNS] -> (s1)
CREATE (p1) - [:OWNS] -> (s2)
CREATE (p2) - [:OWNS] -> (s3)
CREATE (p2) - [:OWNS] -> (s4)
CREATE (p2) - [:OWNS] -> (s5)

CREATE (a1) - [:COOLS] -> (s1)
CREATE (a2) - [:COOLS] -> (s2)
CREATE (a3) - [:COOLS] -> (s3)
CREATE (a4) - [:COOLS] -> (s4)
CREATE (a5) - [:COOLS] -> (s5)

CREATE (o1) - [:COOLED_IN] -> (s1)
CREATE (o4) - [:COOLED_IN] -> (s2)
CREATE (o3) - [:COOLED_IN] -> (s4)
CREATE (o2) - [:COOLED_IN] -> (s3)
CREATE (o2) - [:COOLED_IN] -> (s5)

CREATE (i1) - [:CONTROLS] -> (a1)
CREATE (i2) - [:CONTROLS] -> (a2)
CREATE (i3) - [:CONTROLS] -> (a3)
CREATE (i4) - [:CONTROLS] -> (a4)
CREATE (i5) - [:CONTROLS] -> (a5)

CREATE (i1) - [:SENSES] -> (a1)
CREATE (i2) - [:SENSES] -> (a2)
CREATE (i3) - [:SENSES] -> (a3)
CREATE (i4) - [:SENSES] -> (a4)
CREATE (i5) - [:SENSES] -> (a5)
```

```java
match (:Person {name: 'John'}) - [:OWNS] -> (s) <- [] - (:Object {type: 'papaya'})
CREATE (a1:AC {brand: 'LEZETi', solar: 'yes', ducted: 'yes', power: 885, btu: 11500}) - [:COOLS] -> (s)
CREATE (i:CU {type: 'Cooling', control: ['AC units'], sensing: ['AC power', 'DC power', 'temperature'], coms: 'MQTT', mac: 'b827ebe0f622'}) - [:CONTROLS] -> (a1)
create (i) - [:SENSES] -> (a1)
```

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

```java
match (s) where ID(s) = 12 set s.mac = 'b827ebe0f633' return s
match (s) where ID(s) = 13 set s.mac = 'b827ebe0f645' return s
match (s) where ID(s) = 14 set s.mac = 'b827ebe0d633' return s
match (s) where ID(s) = 15 set s.mac = 'b827ebe0f593' return s
match (s) where ID(s) = 16 set s.mac = 'b827ebe0f599' return s

match (s) where ID(s) = 23 return s

```
```java
match (i:CU {mac: 'b827ebe0f622'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 2.0, c.ki = 0.02, c.kd = 259, c.sp = 54, c.ts = timestamp() - (1000*60*60*60*4), s.c = 46, s.pv = 54.8, s.P = 300, s.ts = timestamp() - (1000*60*60*60*4)
return i, s, c



match (i:CU {mac: 'b827ebe0f633'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 1.5, c.ki = 0.01, c.kd = 180, c.sp = 54, c.ts = timestamp() - (1000*60*60*60*4), s.c = 46, s.pv = 56.4, s.P = 400, s.ts = timestamp() - (1000*60*60*60*4)
return i, s, c


match (i:CU {mac: 'b827ebe0f645'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 4.0, c.ki = 0.15, c.kd = 290, c.sp = 48, c.ts = timestamp() - (1000*60*60*60*2), s.c = 85, s.pv = 56.4, s.P = 700, s.ts = timestamp() - (1000*60*60*60*2)
return i, s, c


match (i:CU {mac: 'b827ebe0d633'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 4.0, c.ki = 0.015, c.kd = 300, c.sp = 40, c.ts = timestamp() - (1000*60*60*60*2), s.c = 55, s.pv = 43, s.P = 800, s.ts = timestamp() - (1000*60*60*60*2)
return i, s, c


match (i:CU {mac: 'b827ebe0f599'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 10.0, c.ki = 0.8, c.kd = 300, c.sp = 40, c.ts = timestamp() - (1000*60*60*60*2), s.c = 120, s.pv = 49, s.P = 700, s.ts = timestamp() - (1000*60*60*60*2)
return i, s, c

match (i:CU {mac: 'b827ebe0f593'}) - [c:CONTROLS] -> (:AC)
match (i) - [s] -> (:AC) set c.kp = 5.0, c.ki = 0.2, c.kd = 120, c.sp = 55, c.ts = timestamp() - (1000*60*60*60*3), s.c = 30, s.pv = 53, s.P = 50, s.ts = timestamp() - (1000*60*60*60*3)
return i, s, c


```

```java
match (i1:CU {mac: 'b827ebe0f622'}) - [c1:CONTROLS] - () - [] - () - [] - (o1:Object) - [] - () - [] - () - [c:CONTROLS] - (i:CU) return c.kp, c.ki, c.kd, c.sp

match (i:CU) - [c:CONTROLS] - () - [] - (s:Space) where i.mac <> 'b827ebe0f622' and abs(s.volume -33.5) < 5 return c.kp, c.ki, c.kd, c.sp


```

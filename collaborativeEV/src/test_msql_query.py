import datetime
import mysql.connector

config = {
  'user': 'matsut',
  'password': 'H8wy8aqi*',
  'host': 'deep.outtter.space',
  'port': 53306,
  'database': 'meow_cooler',
  'raise_on_warnings': True,
}

print config

cnx = mysql.connector.connect(**config)
cursor = cnx.cursor()

#query = ("SELECT first_name, last_name, hire_date FROM employees "
#         "WHERE hire_date BETWEEN %s AND %s")
#query = ("SELECT * FROM measurement LIMIT 10 ")
query = ("Select timestamp, tempSet, tempActual FROM measurement "
         "WHERE deviceID=3 and containerID = 4 and timestamp > (NOW() - INTERVAL 1 HOUR) "
         "ORDER BY timestamp DESC LIMIT 30" )
#hire_start = datetime.date(1999, 1, 1)
#hire_end = datetime.date(1999, 12, 31)

#cursor.execute(query, (hire_start, hire_end))
cursor.execute(query)
results = cursor.fetchall()

temp = np.zeros((len(results),2))
time = []

for index, (timestamp, tempSet, tempActual) in enumerate(results):
    time.append(np.datetime64(timestamp, 's'))
    temp[index, 0] = tempSet
    temp[index, 1] = tempActual

cursor.close()
cnx.close()

import datetime
import mysql.connector
from mysql.connector import MySQLConnection, Error
from ConfigParser import ConfigParser

def read_db_config(filename='config.ini', section='mysql'):
    """ Read database configuration file and return a dictionary object
    :param filename: name of the configuration file
    :param section: section of database configuration
    :return: a dictionary of database parameters
    """

    # create parser and read ini configuration file
    config = ConfigParser()
    config.read(filename)

    print config
    # get section, default to mysql
    db = {}
    if config.has_section(section):
        items = config.items(section)
        for item in items:
            db[item[0]] = item[1]
    else:
        raise Exception('{0} not found in the {1} file'.format(section, filename))

    db['port'] = int(db['port'])
    db['raise_on_warnings'] = True

    return db

def db_connect():
    """ Connect to MySQL database """

    db_config = read_db_config()

    try:
        print('Connecting to MySQL database...')
        conn = MySQLConnection(**db_config)

        if conn.is_connected():
            print('connection established.')
        else:
            print('connection failed.')

        return conn

    except Error as error:
        print(error)

def db_insert(conn, insertData):
    """
    Given an existing DB connection,
    insert control settings.

    ToDo:
    Add option to submit custome insert
    as **kwargs opition.
    """
    #time = str(datetime.datetime.now().replace(microsecond=0))

    add_setting = ("INSERT INTO controlSetting "
                   "(deviceid, containerid, timestamp, status, setpoint, Kp, valveLower, valveUpper, valveCenter, dcmultiplier) "
                   "VALUES (%(deviceid)s, %(containerid)s, %(timestamp)s, %(status)s, %(setpoint)s, %(Kp)s, %(valveLower)s, %(valveUpper)s, %(valveCenter)s, %(dcmultiplier)s)")

    try:
        cursor = conn.cursor()
        cursor.execute(add_setting, insertData)
        conn.commit()
        cursor.close()
        return 1

    except Error as error:
        print(error)
        return 0

def deb_query(conn, query):
    """
    Query the DB
    """

    cursor = cnx.cursor()

    #query = ("Select timestamp, tempSet, tempActual FROM measurement "
    #         "WHERE deviceID=3 and containerID = 4 "
    #         "ORDER BY timestamp DESC LIMIT 30" )

    cursor.execute(query)

    for (timestamp, tempSet, tempActual) in cursor:
        #print("{}, {} was hired on {:%d %b %Y}".format(
        #last_name, first_name, hire_date))
        print (str(timestamp), tempSet, tempActual)
    type(cursor)
    print cursor

    cursor.close()
    return data


if __name__ == '__main__':
    conn = db_connect()

    insert  = {
      'deviceid': 2,
      'containerid': 3,
      'timestamp': str(datetime.datetime.now().replace(microsecond=0)),
      'status': 1,
      'setpoint': 54,
      'Kp': 7.7,
      'valveLower': 10,
      'valveUpper': 255,
      'valveCenter': 35,
      'dcmultiplier': 2
    }

    db_insert(conn, insert)

    conn.close
    print "connection closed."

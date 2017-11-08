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

def connect():
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

if __name__ == '__main__':
    conn = connect()
    conn.close
    print "connection closed."

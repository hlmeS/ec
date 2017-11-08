#! /usr/bin/env python2

"""

Title: sqlCon.py
Author: Holm Smidt
Version: 1.0
Date: 11-04-2017

Overview:
Class wrapper for interfacing with MySQL database
as well as web server (http requests).

"""

import numpy as np
import datetime
import mysql.connector
from mysql.connector import MySQLConnection, Error
from ConfigParser import ConfigParser
import requests
import urllib

class sqlCon:

    def __init__(self):

        self.db = {}


    def read_db_config(self, filename='config.ini', section='mysql'):
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

        if config.has_section(section):
            items = config.items(section)
            for item in items:
                self.db[item[0]] = item[1]
        else:
            raise Exception('{0} not found in the {1} file'.format(section, filename))

        self.db['port'] = int(self.db['port'])
        self.db['raise_on_warnings'] = True


    def db_connect(self):
        """ Connect to MySQL database """

        self.read_db_config()

        try:
            print('Connecting to MySQL database...')
            conn = MySQLConnection(**self.db)

            if conn.is_connected():
                print('connection established.')
            else:
                print('connection failed.')

            return conn

        except Error as error:
            print(error)

    def db_insert(self, conn, insertData):

        cursor = conn.cursor()

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

    def db_query(self, conn, query):
        """
        Query the DB
        """
        cursor = conn.cursor()

        # OLD query
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
        conn.close()
        return temp

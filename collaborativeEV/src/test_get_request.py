import requests
import urllib

params = urllib.urlencode({'tempset': 53, 'Kp': 7.7, 'valve_lower_limit': 15, 'valve_upper_limit': 255, 'valve_center': 35, 'dcmultiplier':2})
response = requests.get("http://deep.outtter.space:51880/kahuku_farms/PID_container2/?%s" % params, auth=('user', 'Mat5uda$$'))
print response

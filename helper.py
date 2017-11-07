import numpy as np
from datetime import datetime

def get_time_of_day(time):
    '''
    Method that converts a time of day (given as a string) into a one-hot vector with 4 components such that 
    component 0 is 1 iff the time of day is between 6 AM and 11:59 AM ("morning")
    component 1 is 1 iff the time of day is between 12 PM and 5:59 PM ("afternoon")
    component 2 is 1 iff the time of day is between 6 PM and 11:59 PM ("evening")
    component 3 is 1 iff the time of day is between 12 AM and 5:59 AM ("late night")
    '''
    result = np.zeros((1,4))
    parts = time.split(":")
    if int(parts[0]) >= 6 and int(parts[0]) < 12:
        if parts[-1].endswith("PM"):
            result[0,2] = 1
        else: #has "AM"
            result[0,0] = 1
    else: #12 - 5
        if parts[-1].endswith("PM"):
            result[0,1] = 1
        else: #has "AM"
            result[0,3] = 1
            
    return result

def get_day_of_week(date):
    '''
    Method that converts a date into a one-hot vector with 7 components such that
    component 0 is 1 iff the day of the week corresponding to the date is *Monday*
    component 1 is 1 iff the day of the week corresponding to the date is Tuesday, etc.

    See below:
    https://stackoverflow.com/questions/9847213/how-do-i-get-the-day-of-week-given-a-date-in-python
    '''
    result = np.zeros((1,7))
    parts = date.split("/")
    dt = datetime(int(parts[2]), int(parts[0]), int(parts[1]))
    result[0,dt.weekday()] = 1
    return result
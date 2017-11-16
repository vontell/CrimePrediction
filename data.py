from helper import *
import numpy as np
import pickle

class CrimeLoader:

    def __init__(self):
        """
        Creates a loader for managing data.
        """
        pass


def chicago_from(row, params=None):
    '''
    Method which takes a row from the Chicago data and returns a Crime representing that row
    '''
    city = "CHICAGO"
    date_intermediate = row[3].split(" ")
    date = date_intermediate[0]
    day = get_day_of_week(date)
    time = date_intermediate[1] + date_intermediate[2]
    time_of_day = get_time_of_day(time)
    crime = row[6]
    location = (float(row[-2][2:]), float(row[-1][0:-2]))
    return Crime(city, date, day, time, time_of_day, crime, None, None, None, location)


config = {
    "root_data": "data/",
    "LA_crime": "crime-in-los-angeles/Crime_Data_2010_2017.csv",
    "CH_crime": "crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",
    "CH_schema": chicago_from,
    "CH_condensed_crime_encoding": {
        'KIDNAPPING': 0,
        'HUMAN TRAFFICKING': 0,
        'OFFENSE INVOLVING CHILDREN': 0,
        'ROBBERY': 1,
        'BURGLARY': 1,
        'THEFT': 1,
        'MOTOR VEHICLE THEFT': 1,
        'BATTERY': 2,
        'ASSAULT': 2,
        'HOMICIDE': 2,
        'CRIM SEXUAL ASSAULT': 2,
        'SEX OFFENSE': 2,
        'NARCOTICS': 3,
        'OTHER NARCOTIC VIOLATION': 3,
        'PUBLIC PEACE VIOLATION': 4,
        'INTERFERENCE WITH PUBLIC OFFICER': 4,
        'OBSCENITY': 4,
        'PUBLIC INDECENCY': 4,
        'INTIMIDATION': 4,
        'STALKING': 4,
        'CRIMINAL TRESPASS': 4,
        'CRIMINAL DAMAGE': 5,
        'ARSON': 5,
        'NON - CRIMINAL': 6,
        'NON-CRIMINAL': 6,
        'NON-CRIMINAL (SUBJECT SPECIFIED)': 6,
        'OTHER OFFENSE': 6,
        'DECEPTIVE PRACTICE': 6,
        'CONCEALED CARRY LICENSE VIOLATION': 7,
        'WEAPONS VIOLATION': 7,
        'PROSTITUTION': 8,
        'LIQUOR LAW VIOLATION': 8,
        'GAMBLING': 8
    },
    "CH_condensed_crime_decoding": [
        "KIDNAPPING / CHILDREN",
        "ROBBERY/BURGLARY/THEFT",
        "ASSAULT/VIOLENCE",
        "NARCOTICS",
        "PUBLIC-RELATED CRIME",
        "DAMAGE/ARSON",
        "OTHER/NON-CRIMINAL",
        "WEAPON-RELATED",
        "PROHIBITIVE CRIME"
    ],
}


def convert_crime_class_to_condensed_integer_CH(c):
    index = config["CH_condensed_crime_encoding"].get(c, 6)
    result = np.zeros(9)
    result[index] = 1
    return result


def convert_one_hot_encoding_to_crime_class_CH(one_hot_encoding):
    index = np.argmax(one_hot_encoding > 0)
    return config["CH_condensed_crime_decoding"][index]


def save_full_crime_encoding(encoding):
    config["CH_full_encoding"] = encoding


def convert_crime_class_to_full_integer_CH(c):
    index = config["CH_full_encoding"].index(c)
    result = np.zeros(len(config["CH_full_encoding"]))
    result[index] = 1
    return result


def convert_one_hot_full_encoding_to_crime_class_CH(one_hot_encoding):
    index = np.argmax(one_hot_encoding > 0)
    return config["CH_full_encoding"][index]


def get_decoder_from(comp_list):
    def decode_result(result):
        inc = 0
        results = []
        if "day" in comp_list:
            day = np.argmax(result[inc:inc + 7] > 0)
            results.append(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"][day])
            inc += 7
        if "time" in comp_list:
            time = np.argmax(result[inc:inc + 4] > 0)
            results.append(["MORNING", "AFTERNOON", "EVENING", "LATE NIGHT"][time])
            inc += 4
        if "time min" in comp_list:
            mins = result[inc:inc + 1]
            results.append(time_from_min(int(mins[0])))
            inc += 1
        if "location" in comp_list:
            latlong = result[inc:inc + 2]
            results.append(latlong)
            inc += 2
        if "crime condensed" in comp_list:
            crime = result[inc:inc + 9]
            results.append(convert_one_hot_encoding_to_crime_class_CH(crime))
            inc += 9
        if "crime full" in comp_list:
            crime = result[inc:inc + len(config["CH_crime_encoding"])]
            results.append(convert_one_hot_full_encoding_to_crime_class_CH(crime))
            inc += len(config["CH_crime_encoding"])

        return results

    return decode_result

class Crime:
    def __init__(self, city, date, day, raw_time, time_of_day, crime, victim_age, victim_sex, weapon, location):
        self.city = city
        self.date = date
        self.day = day
        self.raw_time = raw_time
        self.time_of_day = time_of_day
        self.crime = crime
        self.victim_age = victim_age
        self.victim_sex = victim_sex
        self.weapon = weapon
        self.location = location

    def get_specified_vector(self, comp_list):
        '''
        Returns a feature vector representing this crime.
        '''
        feature = np.array([])
        if "day" in comp_list:
            feature = np.concatenate((feature, self.day[0]))
        if "time" in comp_list:
            feature = np.append(feature, self.time_of_day[0])
        if "time min" in comp_list:
            time = np.array(min_from_time(self.raw_time))
            feature = np.concatenate((feature, [time]))
        if "location" in comp_list:
            others = np.array([self.location[0], self.location[1]])
            feature = np.concatenate((feature, others))
        if "crime condensed" in comp_list:
            feature = np.concatenate((feature, convert_crime_class_to_condensed_integer_CH(self.crime)))
        if "crime full" in comp_list:
            feature = np.concatenate((feature, convert_crime_class_to_full_integer_CH(self.crime)))

        return feature

    def __str__(self):
        return self.crime + " in the " + str(self.time_of_day) + " on " + str(self.day)


def load_data(force_refresh=False):

    if not force_refresh:
        try:
            result = pickle.load(open("crime_data.p", "rb"))
            if result:
                print("Loaded crime data from pickle file")
                return result
        except:
            print("Creating checkpoint for crimes")
            pass

    print("Loading crime data")
    all_data = {"CH": []}

    for city in all_data.keys():
        data_file = config["root_data"] + config[city + "_crime"]
        parse_from = config[city + "_schema"]
        with open(data_file) as csvfile:
            content = csvfile.readlines()
            content = [x.strip() for x in content]
            del content[0]  # Remove the header
            all_data[city].append([])

            # Save the number of possible rows available
            all_data[city].append(len(content))

            count = 0
            for row in content[0:10]:
                try:
                    d = row.split(",")
                    new_crime = parse_from(d)
                    all_data[city][0].append(new_crime)
                except:
                    # print("Unexpected error:", sys.exc_info()[0])
                    # raise
                    count = count + 1

            # Save the number of errored rows
            all_data[city].append(count)

    print("Saving checkpoint")
    pickle.dump(all_data, open("crime_data.p", "wb"))
    print("Finished loading data")
    return all_data

def get_workable_data(data, X_comps, Y_comps):
    '''
    Returns a matrix where rows are feature vectors of
    [1-hot day vector, 1-hot time vector, crime, crime encoding (9 DIGITS LONG), latitude, longitude,]
    Returned as strings due to the crime being a string. Age, sex, and weapon are omitted for now
    since the Chicago data set does not include those.
    '''
    X_matrix = np.array([i.get_specified_vector(X_comps) for i in data])
    Y_matrix = np.array([i.get_specified_vector(Y_comps) for i in data])
    X_decoder = get_decoder_from(X_comps)
    Y_decoder = get_decoder_from(Y_comps)
    return (X_matrix, Y_matrix, X_decoder, Y_decoder)

if __name__ == "__main__":
    results = load_data(force_refresh=False)
    test = results["CH"][0][0:10]

    save_full_crime_encoding(list(set([i.crime for i in results["CH"][0]])))

    X_features = ["day", "time min", "time", "location"]
    Y_features = ["crime condensed"]
    X, Y, X_decoder, Y_decoder = get_workable_data(results["CH"][0], X_features, Y_features)
    print("Featurization achieved")
    print(X_decoder(X[0]))
    print(Y_decoder(Y[0]))
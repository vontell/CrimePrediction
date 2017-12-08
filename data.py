from helper import *
import numpy as np
import pickle

class CrimeLoader:

    def __init__(self):
        """
        Creates a loader for managing data.
        """

        self.results = None
        self.config = {
            "root_data": "data/",
            "LA_crime": "crime-in-los-angeles/Crime_Data_2010_2017.csv",
            "CH_crime": "crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv",
            "CH_social": "chicago_poverty_and_crime.csv",
            "CH_schema": self._chicago_from,
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

    def _chicago_from(self, row, params=None):
        """
        Method which takes a row from the Chicago data and returns a Crime representing that row
        """
        city = "CHICAGO"
        date_intermediate = row[3].split(" ")
        date = date_intermediate[0]
        day = get_day_of_week(date)
        time = date_intermediate[1] + date_intermediate[2]
        time_of_day = get_time_of_day(time)
        crime = row[6]
        location = (float(row[-2][2:]), float(row[-1][0:-2]))
        return Crime(city, date, day, time, time_of_day, crime, None, None, None, location, self)

    def _convert_crime_class_to_condensed_integer_CH(self, c):
        index = self.config["CH_condensed_crime_encoding"].get(c, 6)
        result = np.zeros(9)
        result[index] = 1
        return result

    def _convert_one_hot_encoding_to_crime_class_CH(self, one_hot_encoding):
        index = np.argmax(one_hot_encoding > 0)
        return self.config["CH_condensed_crime_decoding"][index]

    def _save_full_crime_encoding(self, encoding):
        self.config["CH_crime_encoding"] = encoding

    def _save_location_norm_info(self, locations):

        ll_lat = float("inf")
        ll_long = float("inf")
        ur_lat = float("-inf")
        ur_long = float("-inf")

        for location in locations:
            lat = location[0]
            long = location[1]

            if lat < ll_lat:
                ll_lat = lat
            elif lat > ur_lat:
                ur_lat = lat

            if long < ll_long:
                ll_long = long
            elif long > ur_long:
                ur_long = long

        self.config["location norm info"] = (ll_lat, ll_long, ur_lat, ur_long)

    def _get_normed_location_from(self, location):
        lat = location[0]
        long = location[1]

        comps = self.config["location norm info"]

        lat = lat - comps[0]
        long = long - comps[1]
        lat = lat / (comps[2] - comps[0])
        long = long / (comps[3] - comps[1])
        return [lat, long]

    def _get_location_from_norm(self, normed_location):
        lat = normed_location[0]
        long = normed_location[1]

        comps = self.config["location norm info"]

        lat = lat * (comps[2] - comps[0])
        long = long * (comps[3] - comps[1])

        lat = lat + comps[0]
        long = long + comps[1]
        return [lat, long]

    def _get_closest_neighborhood_index(self, location):

        # Take note that this uses a flat earth assumption O.O

        neighs = self.results["CH_social"][0]
        distance = float("inf")
        neigh = None
        for i in range(len(neighs)):
            n = neighs[i]
            dist = ((location[0] - float(n[-2]))**2 + (location[1] - float(n[-1]))**2 )**(1.0/2.0)
            if neigh is None or dist < distance:
                distance = dist
                neigh = i

        return neigh

    def _convert_crime_class_to_full_integer_CH(self, c):
        index = self.config["CH_crime_encoding"].index(c)
        result = np.zeros(len(self.config["CH_crime_encoding"]))
        result[index] = 1
        return result

    def _convert_one_hot_full_encoding_to_crime_class_CH(self, one_hot_encoding):
        index = np.argmax(one_hot_encoding > 0)
        return self.config["CH_crime_encoding"][index]

    def _get_decoder_from(self, comp_list):

        if "all" in comp_list:
            comp_list = ["day", "time", "time min", "hour", "location normalized", "crime condensed", "crime full", "below poverty count"]

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
            if "hour" in comp_list:
                hour = result[inc:inc + 1]
                results.append(time_from_min(int(hour[0])*60))
                inc += 1
            if "location" in comp_list:
                latlong = result[inc:inc + 2]
                results.append(latlong)
                inc += 2
            if "location normalized" in comp_list:
                latlong = result[inc:inc + 2]
                results.append(self._get_location_from_norm(latlong))
                inc += 2
            if "crime condensed" in comp_list:
                crime = result[inc:inc + 9]
                results.append(self._convert_one_hot_encoding_to_crime_class_CH(crime))
                inc += 9
            if "crime full" in comp_list:
                crime = result[inc:inc + len(self.config["CH_crime_encoding"])]
                results.append(self._convert_one_hot_full_encoding_to_crime_class_CH(crime))
                inc += len(self.config["CH_crime_encoding"])
            if "neighborhood" in comp_list:
                neigh = result[inc:inc + self.results["CH_social"][1]]
                index = np.argmax(neigh > 0)
                results.append(self.results["CH_social"][0][index][1])
                inc += self.results["CH_social"][1]
            if "below poverty count" in comp_list:
                b_pov = result[inc:inc + 1]
                results.append(b_pov[0])
                inc += 1
            if "crowded" in comp_list:
                crowd = result[inc:inc + 1]
                results.append(crowd[0])
                inc += 1
            if "no diploma" in comp_list:
                dip = result[inc:inc + 1]
                results.append(dip[0])
                inc += 1
            if "income" in comp_list:
                ind = result[inc:inc + 1]
                results.append(ind[0])
                inc += 1
            if "unemployment" in comp_list:
                une = result[inc:inc + 1]
                results.append(une[0])
                inc += 1

            return results

        return decode_result

    def load_data(self, force_refresh=False, force_save=True, data_limit=None, randomize=True):

        if not force_refresh:
            try:
                print("Loading existing pickled crimes")
                result = pickle.load(open("crime_data.p", "rb"))
                if result:
                    print("Loaded crime data from pickle file")
                    self.results = result
                    self._save_full_crime_encoding(list(set([i.crime for i in self.results["CH"][0]])))
                    self._save_location_norm_info([i.location for i in self.results["CH"][0]])
                    return
            except:
                print("Creating checkpoint for crimes")
                pass

        print("Loading crime data")
        all_data = {"CH": []}

        for city in all_data.keys():
            data_file = self.config["root_data"] + self.config[city + "_crime"]
            parse_from = self.config[city + "_schema"]
            with open(data_file) as csvfile:
                content = csvfile.readlines()
                content = [x.strip() for x in content]
                del content[0]  # Remove the header
                all_data[city].append([])

                # Save the number of possible rows available
                all_data[city].append(len(content))

                # Randomize if requested
                if randomize:
                    np.random.shuffle(content)

                # Shrink if requested
                if data_limit is not None:
                    content = content[0:data_limit]

                count = 0
                for row in content:
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

        self.results = all_data
        self._save_full_crime_encoding(list(set([i.crime for i in self.results["CH"][0]])))
        self._save_location_norm_info([i.location for i in self.results["CH"][0]])
        print("Finished loading crime data")

        print("Now loading social / economics data")
        print("WARNING: THIS IS HARDCODED TO WORK WITH CHICAGO ONLY")

        data_file = self.config["root_data"] + self.config["CH_social"]
        with open(data_file) as csvfile:
            content = csvfile.readlines()
            content = [x.strip() for x in content]
            del content[0]  # Remove the header
            all_data["CH_social"] = []
            all_data["CH_social"].append([])

            # Save the number of possible rows available
            all_data["CH_social"].append(len(content))

            for row in content:
                d = row.split(",")
                all_data["CH_social"][0].append(d)

        self.results = all_data

        print("Finished loading Chicago social data")

        if force_save:
            print("Saving checkpoint")
            pickle.dump(all_data, open("crime_data.p", "wb"))
            print("Finished saving!")

        return

    def get_workable_data(self, X_comps, Y_comps):
        '''
        Returns a matrix where rows are feature vectors of
        [1-hot day vector, 1-hot time vector, crime, crime encoding (9 DIGITS LONG), latitude, longitude,]
        Returned as strings due to the crime being a string. Age, sex, and weapon are omitted for now
        since the Chicago data set does not include those.
        '''
        print("Creating X feature matrix")
        X_matrix = np.array([i.get_specified_vector(X_comps) for i in self.results["CH"][0]])
        print("Creating Y feature matrix")
        Y_matrix = np.array([i.get_specified_vector(Y_comps) for i in self.results["CH"][0]])
        X_decoder = self._get_decoder_from(X_comps)
        Y_decoder = self._get_decoder_from(Y_comps)
        print("Featurization complete")
        return (X_matrix, Y_matrix, X_decoder, Y_decoder)


class Crime:

    def __init__(self, city, date, day, raw_time, time_of_day, crime, victim_age, victim_sex, weapon, location, context):
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
        self.context = context

    def get_specified_vector(self, comp_list):
        '''
        Returns a feature vector representing this crime.
        '''

        if "all" in comp_list:
            comp_list = ["day", "time", "time min", "hour", "location", "crime condensed", "crime full", "below poverty count"]

        feature = np.array([])
        if "day" in comp_list:
            feature = np.concatenate((feature, self.day[0]))
        if "time" in comp_list:
            feature = np.append(feature, self.time_of_day[0])
        if "time min" in comp_list:
            time = np.array(min_from_time(self.raw_time))
            feature = np.concatenate((feature, [time]))
        if "hour" in comp_list:
            hour = np.array(int(int(min_from_time(self.raw_time)) / 60))
            feature = np.concatenate((feature, [hour]))
        if "location" in comp_list:
            others = np.array([self.location[0], self.location[1]])
            feature = np.concatenate((feature, others))
        if "location normalized" in comp_list:
            feature = np.concatenate((feature, self.context._get_normed_location_from(self.location)))
        if "crime condensed" in comp_list:
            feature = np.concatenate((feature, self.context._convert_crime_class_to_condensed_integer_CH(self.crime)))
        if "crime full" in comp_list:
            feature = np.concatenate((feature, self.context._convert_crime_class_to_full_integer_CH(self.crime)))
        if "neighborhood" in comp_list:
            neigh_array = np.array([0]*self.context.results["CH_social"][1])
            neigh_array[self.context._get_closest_neighborhood_index(self.location)] = 1
            feature = np.concatenate((feature, neigh_array))
        if "below poverty count" in comp_list:
            index = self.context._get_closest_neighborhood_index(self.location)
            pov_below = float(self.context.results["CH_social"][0][index][4])
            feature = np.concatenate((feature, np.array([pov_below])))
        if "crowded" in comp_list:
            index = self.context._get_closest_neighborhood_index(self.location)
            crowd = float(self.context.results["CH_social"][0][index][5])
            feature = np.concatenate((feature, np.array([crowd])))
        if "no diploma" in comp_list:
            index = self.context._get_closest_neighborhood_index(self.location)
            dip = float(self.context.results["CH_social"][0][index][7])
            feature = np.concatenate((feature, np.array([dip])))
        if "income" in comp_list:
            index = self.context._get_closest_neighborhood_index(self.location)
            dip = float(self.context.results["CH_social"][0][index][11])
            feature = np.concatenate((feature, np.array([dip])))
        if "unemployment" in comp_list:
            index = self.context._get_closest_neighborhood_index(self.location)
            dip = float(self.context.results["CH_social"][0][index][9])
            feature = np.concatenate((feature, np.array([dip])))

        return feature

    def __str__(self):
        return self.crime + " in the " + str(self.time_of_day) + " on " + str(self.day)


if __name__ == "__main__":

    data = CrimeLoader()
    data.load_data(force_refresh=True, data_limit=10)

    X_features = ["no diploma", "unemployment", "income"]
    Y_features = ["below poverty count"]
    X, Y, X_decoder, Y_decoder = data.get_workable_data(X_features, Y_features)
    print("Featurization achieved")
    print(X[0])
    print(X_decoder(X[0]))
    print(Y_decoder(Y[0]))

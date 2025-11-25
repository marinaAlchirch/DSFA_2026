"""
California Housing dataset
--------------------------

Data Set Characteristics:

Number of Instances: 20640

Number of Attributes: 8 numeric, predictive attributes and the target

Attribute Information:
   #### - MedInc        median income in block group
   #### - HouseAge      median house age in block group
   #### - AveRooms      average number of rooms per household
   #### - AveBedrms     average number of bedrooms per household
   #### - Population    block group population
   #### - AveOccup      average number of household members
   #### - Latitude      block group latitude
   #### - Longitude     block group longitude

###:Missing Attribute Values: None

///////////////////////////////////////////////////////////////////////////

##Info from : https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
"""



"""# Let's find examples where feature values are:

## HouseAge >= 100
## AveRooms >= 150
## AveBedrms >= 100
## Population >= 30000
## AveOccup >= 100

"""


def find_illogical_examples(data, features_of_interest):

  indexes_of_illogical_examples = []
  countAveBeds = 0
  countAveOccup = 0
  countHouseAge = 0
  countAveRooms = 0
  countPopulation = 0

  for feature in features_of_interest:
    for i in range(len(data.frame[feature])):
      if feature == 'HouseAge' and data.frame[feature][i] >= 100:
        indexes_of_illogical_examples.append(i)
        countHouseAge += 1
      elif feature == 'AveRooms' and data.frame[feature][i] >= 150:
        indexes_of_illogical_examples.append(i)
        countAveRooms += 1
      elif feature == 'AveBedrms' and data.frame[feature][i] >= 100:
        indexes_of_illogical_examples.append(i)
        countAveBeds += 1
      elif feature == 'Population' and data.frame[feature][i] >= 30000:
        indexes_of_illogical_examples.append(i)
        countPopulation += 1
      elif feature == 'AveOccup' and data.frame[feature][i] >= 100:
        indexes_of_illogical_examples.append(i)
        countAveOccup += 1
  return indexes_of_illogical_examples, countHouseAge, countAveRooms, countAveBeds, countPopulation, countAveOccup

def clean_data(data, features_of_interest):

  indexes_of_illogical_examples, countHouseAge, countAveRooms, countAveBeds, countPopulation, countAveOccup = find_illogical_examples(data, features_of_interest)
  print("Number of illogical examples: ", len(indexes_of_illogical_examples))
  print("Illogical examples : " + str(indexes_of_illogical_examples))
  print("Per feature the illogical examples are :")
  print("HouseAge: ", countHouseAge)
  print("AveRooms: ", countAveRooms)
  print("AveBeds: ", countAveBeds)
  print("Population: ", countPopulation)
  print("AveOccup: ", countAveOccup)
  print("Dropping those examples from data")
  data.frame = data.frame.drop(indexes_of_illogical_examples)
  print("Data shape after dropping illogical examples: ", data.frame.shape)
  data_X, data_y = data.data.drop(indexes_of_illogical_examples), data.target.drop(indexes_of_illogical_examples)
  data = (data_X, data_y)
  return data




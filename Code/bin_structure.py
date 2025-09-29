""" Class Bin : in this class we form the fields and functions that each of our bins need to have. Those are:

    A) Fields :
    1) bin_id : unique identifier of a bin
    2) lower_bound : the min value of label that a bin can take
    3) upper_bound : the max value of label that a bin can take
    4) labels_list : List of labels per bin
    5) weight : the weight calculated by LDS that a bin will have used later for training our model

    B) Functions :
    1) addLabel : appends a label to the list of labels that a bin has
    2) printBinInfo : prints out the information of a bin (i.e., id, lower/upper bound, list of labels and weight of bin)

"""

class Bin:

    def __init__(self, bin_id, lower_bound, upper_bound, weight=0.0):
        self.bin_id = bin_id
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.labels_list = []
        self.weight = weight

    def addLabel(self, label):
        self.labels_list.append(label)

    def printBinInfo(self):
        return print("Bin : " + str(self.bin_id) + ", with range from : " + str(self.lower_bound) + " to : " + str(self.upper_bound) + ", that has " + str(len(self.labels_list)) + " labels, and weight : " + str(self.weight))


""" Class CollectionOfBins : in this class we form the fields and functions that each collection of bins needs to have. Those are:

    A) Fields :
    1) bins_list : a list that contains bin objects
    2) range_of_bin : the range of each of our bins, i.e., a value that indicates the range of 
    values of labels that can be put inside a bin. In addition, the upper bound of a bin is dependent on the range.
    That is, upper_bound = lower_bound + range_of_bin.  
    

    B) Functions :
    1) organiseLabelsIntoBins : A mapping of labels to bins, returns a mapped bins list
    2) findBinOflabel : finds the bin that a label has been mapped to

"""


class CollectionOfBins:
    def __init__(self, range_of_bin=1.0):
        self.bins_list = []
        self.range_of_bin = range_of_bin


    def organiseLabelsIntoBins(self, labels):
        min_label = min(labels) # find the minimum label value from the label dataset
        max_label = max(labels) # find the maximum label value from the label dataset
        num_bins = int((max_label - min_label) /self.range_of_bin) + 1

        # Initialize an empty bins list, by determining the lower and upper bound of bin based on
        # the user specified range of bin and the iven label dataset
        self.bins_list = []
        for i in range(num_bins):
            lower = min_label + i*self.range_of_bin
            upper = lower + self.range_of_bin
            self.bins_list.append(Bin(bin_id=i, lower_bound=lower, upper_bound=upper))

        # Assign labels to bins
        for label in labels: # for each label in the labels dataset
            # find to which bin each label corresponds to
            bin_of_label = int((label - min_label) / self.range_of_bin)
            self.bins_list[bin_of_label].addLabel(label) # add label to the bin
        # return the mapped bins list
        return self.bins_list

    # Search in bins list for a specific label and return the bin that the label belongs to
    def findBinOflabel(self, label, bins_list):
        for bin in bins_list:
            for entry in bin.labels_list:
                if entry == label:
                    return bin


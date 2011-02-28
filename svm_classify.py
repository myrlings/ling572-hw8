import sys
import math

#### functions
def get_test_vectors(test_filename):
    test_file = open(test_filename, 'r')
    vectors = {}
    count = 0
    # go through test file and add vectors to data structure
    for line in test_file:
        line_array = line.split()
        label = line_array[0]
        line_array.remove(label)
        vectors[count] = {}
        vectors[count]["_label_"] = label
        for element in line_array:
            # in file feature value pairs stored as f1:v1
            fv = element.split(":")
            vectors[count][fv[0]] = fv[1]
        count += 1
    return vectors

def get_model(model_filename):
    model_file = open(model_filename, 'r')
    model = {}
    svm_type = model_file.readline() # don't need this
    kernel_type = model_file.readline().split()[1]
    gamma = model_file.readline().split()[1]
    coef = model_file.readline().split()[1]
    nr_class = model_file.readline().split()[1] # should always be 2
    total_sv = model_file.readline().split()[1]
    rho = model_file.readline().split()[1]
    labels = model_file.readline().split()[1:2] # shd always be 0 1
    nr_sv = model_file.readline().split() # list w/ # sv for each label
    nr_sv.remove("nr_sv")
    sv = model_file.readline() # should always be SV

    count = 0
    for line in model_file:
        line_array = line.split()
        weight = line_array[0]
        line_array.remove(weight)
        model[count] = {}
        model[count]["_weight_"] = weight
        for element in line_array:
            fv = element.split(":")
            model[count][fv[0]] = fv[1]
        count += 1
    return [model, kernel_type, gamma, coef, total_sv, rho, nr_sv]


#### main
if len(sys.argv) < 3:
    print "Please give arguments: test_data model_file sys_output"
    sys.exit()

test_vectors = get_test_vectors(sys.argv[1])
model_list = get_model(sys.argv[2])
print model_list

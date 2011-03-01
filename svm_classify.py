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
            vectors[count][fv[0]] = int(fv[1])
        count += 1
    return vectors

def get_model(model_filename):
    model_file = open(model_filename, 'r')
    model = {}
    svm_type = model_file.readline() # don't need this
    kernel_type = None
    degree = 0.0
    gamma = 0.0
    nr_class = None
    coef = 0.0
    total_sv = None
    rho = 0.0
    nr_sv = None
    
    line = model_file.readline().split()
    variable = line[0]
    while  variable != "SV":   #should exit test sequence as early as possible every time
        #print "variable="+variable
        if variable == "svm_type":
            svm_type = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "kernel_type":
            kernel_type = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "degree":
            line = model_file.readline().split()
            variable = line[0]
            degree = line[1]
        if variable == "gamma":
            gamma = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "nr_class":
            nr_class = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "coef0":
            coef = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "total_sv":
            total_sv = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "rho":
            rho = line[1]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "label":
            label = line[1:len(line)]
            line = model_file.readline().split()
            variable = line[0]
        if variable == "nr_sv":
            nr_sv = line[1]
            line = model_file.readline().split()
            variable = line[0]
    sv = line[0]    
#     kernel_type = model_file.readline().split()[1]
# #    degree = model_file.readline().split()[]
#     gamma = model_file.readline().split()[1]
#     coef = model_file.readline().split()[1]
#     nr_class = model_file.readline().split()[1] # should always be 2
#     total_sv = model_file.readline().split()[1]
#     rho = model_file.readline().split()[1]
#     labels = model_file.readline().split()[1:2] # shd always be 0 1
#     nr_sv = model_file.readline().split() # list w/ # sv for each label
#     #nr_sv.remove("nr_sv")
#     sv = model_file.readline() # should always be SV

    count = 0
    for line in model_file:
        line_array = line.split()
        weight = line_array[0]
        line_array.remove(weight)
        model[count] = {}
        model[count]["_weight_"] = float(weight)
        for element in line_array:
            fv = element.split(":")
            model[count][fv[0]] = int(fv[1])
        count += 1
    return [model, kernel_type, degree, gamma, nr_class, coef, total_sv, rho, nr_sv]

def predict(test_vectors, model_list):
    sys_data = {}
    kernel_function_option = model_list[1]
    model = model_list[0]
    kernel_function = ""
    if kernel_function_option == "linear":
        kernel_function = get_linear
    elif kernel_function_option == "polynomial":
        kernel_function = get_poly
    elif kernel_function_option == "rbf":
        kernel_function = get_rbf
    elif kernel_function_option == "sigmoid":
        kernel_function = get_sigmoid

    for index in test_vectors:
        vector = test_vectors[index]
        sv_sum = 0
        true_label = vector["_label_"]
        for sv in model:
            weight = model[sv]["_weight_"]
            support_vector = model[sv]
            del support_vector["_weight_"] # remove extra feature so can calc fn
            del vector["_label_"] # remove extra feature so can calc fn
            k = kernel_function(vector, support_vector, float(model_list[2]),\
            float(model_list[3]), float(model_list[5])) 
            support_vector["_weight_"] = weight
            vector["_label_"] = true_label
            k = k * weight
            sv_sum += k
        sv_sum = sv_sum - float(model_list[7])
        expected = ""
        if sv_sum >= 0:
            expected = 0
        else:
            expected = 1
        sys_data[index] = [true_label, str(expected), sv_sum]
    return sys_data

# return dot product of the instance vector and the support vector
# ignore degree, gamma, and coef
# u'*v
def get_linear(instance_vector, support_vector, degree, gamma, coef):
    summation = 0
    for f in instance_vector:
        if f in support_vector: # only care about non-zero in both vectors
            summation += instance_vector[f] * support_vector[f]
    return summation

# return result of polynomial function on the vectors
# (gamma*u'*v + coef0)^degree
def get_poly(instance_vector, support_vector, degree, gamma, coef):
    summation = 0
    for f in instance_vector:
        if f in support_vector: 
            summation += instance_vector[f] * support_vector[f]
    num = gamma * summation
    num = num + coef
    num = num ** degree
    return num

# tanh(gamma*u'*v + coef0)
def get_sigmoid(instance_vector, support_vector, degree, gamma, coef):
  summation = 0
  for f in instance_vector:
      if f in support_vector: # only care about non-zero in both vectors
          summation += instance_vector[f] * support_vector[f]
  num = gamma * summation
  num = num + coef 
  num = math.tanh(num)
  return num

def get_rbf(instance_vector, support_vector, degree, gamma, coef):
  summation = 0
  all_vectors = []
  #concatenating key sets, casting to a set to remove duplicates
  all_vectors.append(instance_vector.keys().extend(support_vector.keys())) 
  all_vectors = set(all_vectors)
  for vector in all_vectors:
    if vector in instance_vector:
        u = instance_vector[vector]
    else:
        u = 0
    if vector in support_vector:
        v = support_vector[vector]
    else:
        v = 0
    summation += math.pow((u - v),2)
  num = (-1*gamma * summation)
  num = (math.exp(num))
  return num# exp(-gamma*|u-v|^2) -- this one looks different, please tell me if i interpreted it wrongly

def print_sys(sys_data, sys_filename):
    sys_file = open(sys_filename, 'w')

    for vector in sorted(sys_data.keys()):
        sys_file.write("--index:" + str(vector) + "-- ")
        sys_file.write(sys_data[vector][0] + " ")
        sys_file.write(sys_data[vector][1] + " ")
        sys_file.write(str(sys_data[vector][2]) + "\n")

# a little hack-y since we know there are only two labels, 0 and 1
def print_acc(sys_data):
    print "\nConfusion matrix for the testing data:"
    print "row is the truth, column is the system output\n"
    sys.stdout.write("\t0\t1\n")
    counts = {}
    counts['0'] = {}
    counts['0']['0'] = 0
    counts['0']['1'] = 0
    counts['1'] = {}
    counts['1']['0'] = 0
    counts['1']['1'] = 0
    num_right = 0
    for vector in sys_data:
        counts[sys_data[vector][0]][sys_data[vector][1]] += 1        
        if sys_data[vector][0] == sys_data[vector][1]:
            num_right += 1

    sys.stdout.write("0\t")
    sys.stdout.write(str(counts['0']['0']) + "\t")
    sys.stdout.write(str(counts['0']['1']) + "\n")
    sys.stdout.write("1\t")
    sys.stdout.write(str(counts['1']['0']) + "\t")
    sys.stdout.write(str(counts['1']['1']) + "\n")

    acc = float(num_right) / len(sys_data)
    print "Testing accuracy:", acc

    


#### main
if len(sys.argv) < 3:
    print "Please give arguments: test_data model_file sys_output"
    sys.exit()

test_vectors = get_test_vectors(sys.argv[1])
model_list = get_model(sys.argv[2])
sys_data = predict(test_vectors, model_list)
print_sys(sys_data, sys.argv[3])
print_acc(sys_data)

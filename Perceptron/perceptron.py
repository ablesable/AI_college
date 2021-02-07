#215881 - Bartosz Soból - Perceptron
from random import randint
from random import uniform
import matplotlib.pyplot as plt


def points_generation(quantity, value_range, eq_param): #equation_parameters are: a, b, c
    points_list = []
    for i in range(quantity):
        x1 = randint(value_range[0], value_range[1])
        x2 = randint(value_range[2], value_range[3])
        if (eq_param[0]*x1 + eq_param[1]*x2 + eq_param[2]) >= 0:
            category = 1
        else:
            category = 0
        point = (x1, x2, category)
        points_list.append(point)
    return points_list

class Perceptron:

    def __init__(self, epochs): 
        self.weights = []
        for i in range(3):
            self.weights.append(uniform(-100, 100))
        self.epochs = epochs
        
        self.train_success = 0
        self.train_false = 0
        
        self.learning_rate = uniform(0.001, 0.999)
        
        self.current_epoch = 0
        self.test_success = 0
        self.test_false = 0

        self.weight_correction = 0 
            

    def training(self, train_points):
        x1 = train_points[0]
        x2 = train_points[1]
        category = train_points[2]
        result = 0

        train_sum = self.weights[0]*x1 + self.weights[1]*x2 + self.weights[2]
        if train_sum >= 0 :
            result = 1
        else:
            result = 0

        if(result != category):
            error = category - result #getting an error
            
            self.train_false += 1
            self.correction(train_points, error)
            self.weight_correction += 1
            print(f"{self.weight_correction} changing of weights, {self.weights} - now it look like this")
        else:
            self.train_success += 1

    def correction(self, train_points, error):
        self.weights[0] = self.weights[0] + error * train_points[0] * (self.learning_rate * (self.epochs - self.current_epoch)) #self epochs and current_epochs for better results
        self.weights[1] = self.weights[1] + error * train_points[1] * (self.learning_rate * (self.epochs - self.current_epoch))
        self.weights[2] = self.weights[2] + error * (self.learning_rate * (self.epochs - self.current_epoch))

    def test(self, test_points):
        x1 = test_points[0]
        x2 = test_points[1]
        category = test_points[2]

        test_sum = self.weights[0]*x1 + self.weights[1]*x2 + self.weights[2]
        
        if test_sum >= 0 :
            result = 1
        else:
            result = 0

        if(result != category): #comparing category of point to result
            self.test_false += 1
        else:
            self.test_success += 1
        

#main
eq_param = []
value_range = (-5, 5, -5, 5)
print("Please type parameters of the function:")
for i in range(3):
    param = int(input())  #A = 4, B = 2, C = 2
    eq_param.append(param)

number_of_training = 80
number_of_test = 20

training_points = points_generation(number_of_training, value_range, eq_param) #80 elements training set
testing_points = points_generation(number_of_test, value_range, eq_param) #20 elements test set
repeats = int(input("Please type quantity of repeats in training process:")) #providing desired epochs in training process

perceptron = Perceptron(repeats) #perceptron initialization

array_show_train = [] #getting data for plots
array_show_test = []
repeats_array= [] 
for repeat in range(repeats): #repeats - epochs
    perceptron.current_epoch += 1
    for point in training_points:
        perceptron.training(point)
    print(f"Train Results in {repeat+1} epoch:")
    learning_percentage = (perceptron.train_success/len(training_points))*100
    print(f"{learning_percentage} %")
    perceptron.train_false = 0
    perceptron.train_success = 0
    array_show_train.append(learning_percentage)

    for point in testing_points:
        perceptron.test(point)
    print(f"Test Results in {repeat+1} epoch:")
    testing_percentage = (perceptron.test_success/len(testing_points))*100
    print(f"{testing_percentage} %")
    perceptron.test_false = 0
    perceptron.test_success = 0
    array_show_test.append(testing_percentage)
    repeats_array.append(repeat+1)
    print("----------------------------------")

print("That's End")
print(f"Perceptron changed weights: {perceptron.weight_correction} times, before learned itself.")
perceptron.weight_correction = 0

plt.plot(repeats_array, array_show_train) # training plot
plt.show()
plt.plot(repeats_array, array_show_test) # testing plot
plt.show()


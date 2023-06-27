import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from utils import *
import copy
import math


# Create the main window
window = tk.Tk()

# Set the initial value of the variable
prediction_value = 3.5

# Function to update the prediction value
def update_prediction():
    global prediction_value
    prediction_value = float(entry.get())

# Function to calculate the predicted price
def calculate_price():
    global prediction_value, w, b
    predict = prediction_value * w + b
    output_label.config(text='Predicted price: $%.2f' % (predict * 1000))


#Creating the X and y training data
#X_train is the input variable (size in 1000 square feet)
#y_train is the target/output variable (Price in 1000s of dolalrs)
x_train = np.array([1.0, 1.2, 1.25, 1.5, 2.0, 2.1, 2.2, 2.5, 3.1, 3.3, 3.7, 4.0])
y_train = np.array([300.0, 330.0, 370.0, 400.0, 480.0, 523.0, 525.0, 575.0, 650.0, 670.0, 685.0, 740.0])


#Function to calculate the cost
def compute_cost(x, y, w, b):
   
   # number of training examples
   m = x.shape[0] 
   total_cost = 0

   # Variable to keep track of sum of cost from each example
   cost_sum = 0

   # Loop over training examples
   for i in range(m):
       
       # Uinsg the Linear Regression Function to get prediction f_wb for the ith example
       f_wb = w * x[i] + b

       # Code to get the cost associated with the ith example
       cost = (f_wb - y[i]) ** 2

       # Add to sum of cost for each example
       cost_sum = cost_sum + cost 

   # Get the total cost as the sum divided by (2*m)
   total_cost = (1 / (2 * m)) * cost_sum

   return total_cost


#Function to calculate the gradient of the cost function
def compute_gradient(x, y, w, b): 

    # Number of training examples
    m = x.shape[0]
    
    #Starting values for w and b parameters
    dj_dw = 0
    dj_db = 0
    
    #Loop w and b through the gradient descent algorithm till convergence
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
 
    return dj_dw, dj_db


#Function to implemenet Gradient Descent using learning rate Alpha to find and update the optimal values of w and b
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):     
   
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing


# Initialize fitting parameters at a starting point (0)
initial_w = 0.
initial_b = 0.

#Runs gradient descent 1500 times with learning rate Alpha at 0.01
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)


#Calculates predictions for entire x_train dataset
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b


#Plotting Linear Regression Best-Fit line
plt.plot(x_train, predicted, c = "b")
plt.scatter(x_train, y_train, marker='x', c='r') 
plt.title("Price vs Area of House in sqft")
plt.ylabel('Price in $1,000s')
plt.xlabel('Area of house in 1000 sqft')
plt.show()


#Creating predictions for new values
#Change prediction_value to your desired house area in sqft, divided by 1000
predict1 = prediction_value  * w + b

print('For a house with an area of ' + str(round(prediction_value * 1000)) + ' sqft' + ', we predict a price of $%.2f' % (predict1*1000))


# Create a label and entry for user input
input_label = tk.Label(window, text='House area in 1000 sqft:')
input_label.pack()
entry = tk.Entry(window)
entry.pack()

# Create a button to update the prediction value
update_button = tk.Button(window, text='Update', command=update_prediction)
update_button.pack()

# Create a button to calculate the predicted price
calculate_button = tk.Button(window, text='Calculate Price', command=calculate_price)
calculate_button.pack()

# Create a label to display the predicted price
output_label = tk.Label(window, text='Predicted price: $0.00')
output_label.pack()

# Start the main event loop
window.mainloop()
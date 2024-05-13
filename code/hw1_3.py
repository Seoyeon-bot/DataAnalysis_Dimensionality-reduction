# for part 3 dimensionality reduction 
import numpy as np
from timeit import default_timer as timer # to measure taken time for calculate covar matrix
import linecache 
import os 
import matplotlib.pyplot as plt

# if Database.txt exist then delete, so we can start from the new.
def delete_numerical_Data_file(): 
    if os.path.exists("Database.txt"):
        os.remove("Database.txt")
        print("Database.txt exist so clear first")
    else: 
        print("We can create Database.txt")

delete_numerical_Data_file()
    
def extract_numerical_data(s_line, e_line, filename, outputfile_name):
    lines = []; 
    # open .txt file to  read 
    with open(filename, 'r') as f:
        print("Open ", filename)
        for count, line in enumerate(f,1):
            # check count 
            if s_line <= count <= e_line:
                lines.append(line.strip())
        lines.append('\n')
    # write extracted data into output file 
    with open(outputfile_name,'a') as outputFile : 
       outputFile.write('\n'.join(lines))
        
    print("Data extracted and stored in output file : numericalData.txt")

# center point 
def center(filename):
    # Read data from filename 
    try:
        # https://www.educative.io/answers/what-is-the-genfromtxt-function-in-numpy
        data = np.genfromtxt(filename, delimiter=None) # read row by row 
        print("load data : \n", data , " \n") # check data 
        
    except FileNotFoundError: # handle error 
        print("Error: ", filename, " not found.")
        return None

    if len(data)==0: # handle error if data is empty. 
        print("Error: No data points found in " , filename , ".")
        return None
    
    # let axis = 0 so we can calculte mean per colum . axis =1 mean per row. 
    mean = np.mean(data, axis=0)  # Calculate mean for each column
    print("mean : \n", mean, " \n"); # represented as scientific notaion. 
    centered_data = data - mean # Center each column
    
    rounded_centered_data = np.round(centered_data, 3 ) # round up to 3 decimal points 
    np.savetxt("Z_vector.txt", rounded_centered_data, delimiter=" , ", fmt="%.3f")
    return mean, rounded_centered_data

# Load data and center it - make sure to clear Database.txt before running it since it will append values. 
filename1 = "Database.txt"
extract_numerical_data(54, 1077, "hw1Data.txt", filename1) 
extract_numerical_data(1083, 2106, "hw1Data.txt", filename1) 
mean, Z_vector = center(filename1)

# get total number of data points  : # of row 
def calculate_n (centeredData_vector):
    if centeredData_vector is not None:
        n = centeredData_vector.shape[0]  # Total number of data points, row 
        print("\ntotal number of n: \n", n , " \n")
        # print("Z_vector : ", centeredData_vector)
        return n; 
    else:
        print("Error: Unable to n.")

Database1_n = calculate_n(Z_vector)


# part b calculate covariance matrix (sigma) in 3 different way. 
# sigma : E[(x-mu)^transpo * (x-mu)] definition of sample covariance
def cov1(n, centerd_dataPoints):  # pass Z 
    s_time  = timer(); # start timeer 
    Z_transpo = centerd_dataPoints.T;
    covariance_matrix = np.dot( Z_transpo, centerd_dataPoints) / (n-1) 
    e_time = timer(); 
    elapsedT = e_time - s_time; 
    return  covariance_matrix, elapsedT; 
    
    
# SampleCovarianceMatrix:Inner Product 
# sigma = 1/n * Z^T * Z  a matrix product
def cov2(n, centerd_dataPoints): 
    s_time  = timer(); # start timeer 
    Z_transpo = centerd_dataPoints.T; 
    covariance_matrix = np.dot(Z_transpo, centerd_dataPoints)/n;
    e_time = timer(); 
    elapsedT = e_time - s_time;
    return  covariance_matrix, elapsedT; 
    
# as a sum of outer products
# 1/n ( sigma i=1 to n, Zi * Zi^T )
def cov3(n, centered_dataPoints):
    sum = 0
    s_time  = timer(); # start timeer 
    for z in centered_dataPoints: 
        round_z = np.round(z, decimals=3)
        print("Z: \n", round_z)
        product = z * z.T
        sum += product 
    covariance_matrix = 1/n * (sum)
    e_time = timer(); 
    elapsedT = e_time - s_time;
    return covariance_matrix, elapsedT


# calculate covariance matrixs using three different methods 
covariance_matrix1, t1 = cov1(Database1_n,Z_vector); 
covariance_matrix2,t2 = cov2(Database1_n,Z_vector); 
covariance_matrix3 ,t3= cov3(Database1_n,Z_vector); 

# print. 
print("Covariance Matrix : with Database1 and cov1() function : \n", covariance_matrix1, "\nElapsed Time : ", t1 , "\n");
print("Covariance Matrix : with Database1 and cov2() function : \n", covariance_matrix2 , "\nElapsed Time : ", t2 , "\n");
print("Covariance Matrix : with Database1 and cov3() function : \n", covariance_matrix3,  "\nElapsed Time : ", t3 , "\n");


#Discuss the differences in terms of algorithm complexity
# and explain the difference in measured time

# use calculated covariance matrix to get eigenvector and value.
# used resource
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix1) # return array 1 * D dimention  lambs in eigenvalues

# Print the eigenvectors and eigenvalues
print("Eigenvectors:")
print(eigenvectors)
print("\nEigenvalues:")
print(eigenvalues) 

#  r that will ensure 90% retained variance.
def compute_fraction_of_total (eigenvalues, eigenvectors, alpha) : 
    # if f(r) >= alpha then create reduced basis U1, u2 ,,, Ur and get new coordinate 
    # calculate f(r) for all r = 1, 2, ,,, updo d-dimention 
    var_d = np.sum(eigenvalues) # var(D) , denominator. 
    print("\nCalculate total variance for denominator part var_d : " , var_d)
    
    for i in range(1, len(eigenvalues)+1): # 1,2,,,10
        var_r = 0; 
        for j in range(i):  # Iterate from 0 to i-1
            var_r += eigenvalues[j]
        print("calcualted var_r for r=",i, var_r)
        fraction_f_r = var_r / var_d; 
        
        if fraction_f_r >= alpha:  # check if smallest f(r) >= alpha then yes else no 
            new_basis_vector_U = []; 
            print("Case f(r) >= alpha| f(", i, ")", fraction_f_r, " alpha: ", alpha);
            reduced_dimention = i; 
            
            for k in range (reduced_dimention):  # add u1 ,u2 ,, upto ui 
                Ui = eigenvectors[k]
                new_basis_vector_U.append(Ui) # check 
                print("\nDimension: ", k+1)
                print("New Basis vector: ", Ui)
            
            # Convert new_basis_vector_U to a NumPy array
            new_basis_vector_U = np.array(new_basis_vector_U)
            
            # calculate new coordinate vector A for new basis vector U1, U2 ,,, Ur 
            U_transpo = new_basis_vector_U.T 
            A = []

            # Read original data points from "Datapoints.txt"
            with open("Database.txt", "r") as file:
                x = np.loadtxt(file) # datapoints are seperated by space
            
            # Calculate new coordinates A for new basis vectors U1, U2, ..., Ur
            A = np.dot(x, U_transpo)
            return  reduced_dimention, new_basis_vector_U , A
    print("Unable to reduce demention. ")
    return None   
  
# Assuming mean, centered data points, covariance matrix, eigenvalues, and eigenvectors are defined
# alpha : minimum fraction 
r, new_basis_vector_U , A = compute_fraction_of_total(eigenvalues, eigenvectors, 0.90) # 90% of the variance
print("\nOutput after PCA demention reduced to -> ", r, "\n")
print(A)
print(A.shape)

# task e : plot first two componenet x : dimention, y: that dimention's magnitude 
X_values = [] # store dimention 
Y_values = [] # store magnitude

for i in range(r):  # Extract the first two components u1 u2 within reduced demesion range r 
    col = A[:, i]  # select all row from ith colum
    # calculate magnitude for Y axis 
    magnitude = np.linalg.norm(col) 
    print("magnitude : ", magnitude, " dimention : ", i+1)
    X_values.append(i+1)
    Y_values.append(magnitude)
   
# Plot the components
plt.bar(X_values, Y_values);

plt.xlabel('Dimensions')
plt.ylabel('Magnitude in each dimensions')
plt.title('First Two Components and its magnitude')
plt.xticks(X_values)
plt.show()


# Save the first two components to a text file
with open('Components.txt', 'w') as file:
    for i in range(r):  # 1 2 in our case 
        Ui = new_basis_vector_U[i] # u1 u2 
        components = map(str, Ui) # convert Ui into string type 
        line = ' , '.join(components) # create comma seperated text type 
        file.write(line + "\n") # write on the file
# task f 


"""
Compute the reduced dimension data matrix A with two dimensions by projection on the first two PCs.
Plot the points using a scatter plot
Are there clusters of nearby points? 
What is the retained variance for r = 2?
"""

# Read the original data points X 
with open("Database.txt", "r") as file:
    x = np.loadtxt(file)  # read from file 

# calculate new coordinate point A after PCA dimentional reduction. new_basis_vector_U.t has U1 and U2 
A = np.dot(x, new_basis_vector_U.T)

# Scatter plot
plt.figure(figsize=(10, 8))
first_pc = A[:, 0]  # get first principal component 
second_pc = A[:, 1]  # get second principal compenet 
plt.scatter(first_pc, second_pc, c='b', marker='o', alpha=0.5) # # https://www.geeksforgeeks.org/matplotlib-pyplot-scatter-in-python/
plt.xlabel('First principal component (PC1)')
plt.ylabel('Second principal component (PC2)')
plt.title('Scatter Plot after dimensional reduction on d=2')
plt.grid(True)
plt.show()
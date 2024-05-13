# ICSI431 Homework1 Spring 2024
# Student : Seoyeon Choi
# python --version : Python 3.11.4
# Question 1 -a PMF and CDF 
import numpy as np
import matplotlib.pyplot as plt

# PMF 
inital_student_number = 1000; 
staying_rate = 0.85; 
def pmf(x) :

    if x==1 : 
        pmf_value = (1/1000) * (inital_student_number * (staying_rate))
    else: 
        pmf_value = (1/1000) * (inital_student_number * (staying_rate)**x)
    return pmf_value

              
print("check pmf value")
print(pmf(1))
print(pmf(2))
print(pmf(3))
print(pmf(4)) 
     
# CDF 
def CDF(x):
    if x==1 : 
        cmf_value = pmf(1)
    elif x ==2 : 
        cmf_value = pmf(1) + pmf(2)
    elif x ==3 : 
        cmf_value = pmf(1) + pmf(2) + pmf(3)
    elif x == 4:
        cmf_value = pmf(1) + pmf(2) + pmf(3) + pmf(4)
    return cmf_value;

print("check cmf : ")
print(CDF(1)); 
print(CDF(2)); 
print(CDF(3)); 
print(CDF(4)); 

# store pmf and cdf values to plot 
pmf_dict = {};  # create empty dictionary 
cmf_dict = {} ; 
def store_pmf(): 
    for i in range(1,5): # X : {1,2,3,4}
        pmf_dict[i] = pmf(i); 
    return pmf_dict; 

def store_cmf():
    for i in range(1,5): 
        cmf_dict[i] = CDF(i); 
    return cmf_dict; 

print("\nPrint pmf values:\n" , store_pmf()); 
print("\nPrint cmf values : \n" , store_cmf());

def calculate_colA_mean(count):
   mean = 0; 
   for i in range(1, count+1):  # iterate 1,2,3,4,,, count 
       value = i * (pmf(i))
       mean +=value; 
   return mean; 

# store college C's pmf values f^(x)
colC_pmf_values = {}; 
    
def cal_colC_mean():
    mean = 0; 
    TotalNumber = 1190; # 1000 for iniitial col A student + col B has 50, 40, 100 student for year 6,7,8
    n1 = 1000; # initial total studnet in col A 
    passing_rate = 0.85; 

    for i in range(1, 9): 
        value = 0
        if i  == 1: 
            pmf = (n1/TotalNumber); colC_pmf_values[i] = pmf;
            value = i * pmf; 
            mean+=value; 
        elif i >1 & i <=5: 
            pmf = (n1 * passing_rate ** (i-1))/ TotalNumber;  colC_pmf_values[i] = pmf; 
            value = i * pmf
            mean+=value; 
        elif i == 6: 
            pmf = (n1 * passing_rate ** (i-1) + 40) / TotalNumber; colC_pmf_values[i] = pmf; 
            value = i * pmf;
            mean+=value; 
        elif i ==7: 
            pmf = (n1 * passing_rate ** (i-1) + 50 )/ TotalNumber; colC_pmf_values[i] = pmf; 
            value = i * pmf;
            mean+=value; 
        elif i == 8 : 
            pmf = (n1 * passing_rate ** (i-1) + 100)  / TotalNumber; colC_pmf_values[i] = pmf; 
            mean+= value; 
    return mean;
            
colA_mean = calculate_colA_mean(4); 
colC_mean = cal_colC_mean(); 

       
print("\nMean(College A X=4)", colA_mean) 
print("\nMean(College C X=8)", colC_mean)   

for key, value in  colC_pmf_values.items(): 
    print(key,"th : pmf: " , value) 

def cal_variance(count): 
    var = 0
    if count == 4: 
        mean = calculate_colA_mean(4)
        for i in range(1, count + 1): 
            pmf_value = pmf(i)  # Assuming pmf is defined elsewhere
            value = ((i - mean) ** 2) * pmf_value
            var += value
    else: 
        mean = cal_colC_mean()
        for i in range(1, 9): 
            pmf_value = colC_pmf_values.get(i)  # Access PMF value for year i
            if pmf_value is not None:  # Check if PMF value exists
                value = ((i - mean) ** 2) * pmf_value
                var += value
    
    return var 

colA_var =  cal_variance(4);
colC_var =  cal_variance(8);  

print("\nVariance(CollegeA X=4) ", colA_var) 
print("\nVariance(College C X=8)" , colC_var)

# Task1-e create box plot for years in college A and college C 
colA_student_dict = {} ; 
colC_student_dict = {} ; 

def cal_col_in_college(numberOfyear) : 
    n1 = 1000; # student in college in first year 
    for i in range(1, numberOfyear+1) : # 1,2,3 ,,, number of year 
        if i == 1: # handle first year 
            colA_student_dict[i] = n1; 
            colC_student_dict[i] = n1; 
        else: 
            if i < 5: 
                total_student = n1 * (pmf(i-1))
                colA_student_dict[i] = total_student; 
                colC_student_dict[i] = total_student; 
                # return colA_student_dict; 
            else: # i = 5,6,7,8,
                if i == 5: 
                    total_student = n1 * (pmf(i-1))
                    colC_student_dict[i] = total_student; 
                elif i == 6: 
                    total_student = (n1* (pmf(i-1))) + 40; 
                    colC_student_dict[i] = total_student; 
                elif i == 7: 
                    total_student = (n1* (pmf(i-1))) + 50; 
                    colC_student_dict[i] = total_student; 
                elif i == 8: 
                    total_student = (n1* (pmf(i-1))) + 100;
                    colC_student_dict[i] = total_student; 
                     
    if numberOfyear >=5 : 
        return colC_student_dict
    else: 
        return colA_student_dict

print("\nColA people in college per year : \n" , cal_col_in_college(4)); 
print("\nColC people in college per year : \n" , cal_col_in_college(8)); 

# range : Maximum number of student - minimum number of student. 
max_student_A = 1000; 
max_student_C = 1000;  # max number of student 

def get_minimum_student(dictionary): 
    min_student = 1000; 
    for value in dictionary.values(): 
        if value < min_student: 
            min_student = value; 
    return min_student; 
# get minimum number of student in college A and C 
min_student_A = get_minimum_student(colA_student_dict)
min_student_C = get_minimum_student(colC_student_dict)

range_A = max_student_A - min_student_A; 
range_C = max_student_C - min_student_C; 


# check 
print("\nMinimum number of student in college A: ", min_student_A, " range : " , range_A); 
print("\nMinimum number of student in college C: ", min_student_C, " range: ", range_C); 

# plot pmf values 
plt.subplot(1,2,1) # create subplot with 1 row 2 col
plt.bar(pmf_dict.keys(), pmf_dict.values())
plt.xlabel('Year in College A : X value')
plt.ylabel('PMF values')
plt.title('PMF')

# plot cmf values 
plt.subplot(1,2,2) # create 2nd plot
plt.bar(cmf_dict.keys(), cmf_dict.values(), color="red")
plt.xlabel('Year in College A : X value')
plt.ylabel('CMF values')
plt.title('CMF')

# print plots 
plt.subplots_adjust(wspace=0.3) # give width space between coordinates
plt.show()


# plot box block for college A and C. 

# create data for box plots 
colA_Data = [colA_mean, colA_var , min_student_A, max_student_A , range_A]
colC_Data = [colC_mean, colC_var, min_student_C, max_student_C, range_C]

# create box block 
plt.boxplot([colA_Data, colC_Data], labels=['Xa', 'Xc'])

# lable and title 
plt.xlabel('Year in College')
plt.ylabel('Number of students in college per year')
plt.title('Comparing college A and C')

plt.show()


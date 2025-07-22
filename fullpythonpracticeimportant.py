# Python Practice Questions
# Pandas (10 Questions)
# 1. Load a CSV file and display the first 5 rows.
# 2. Check for null values in a DataFrame.
# 3. Replace all null values with the mean of the column.
# 4. Group data by a column and calculate the average.
# 5. Filter rows where column 'age' > 25.
# 6. Sort the DataFrame based on column 'salary'.
# 7. Rename columns in a DataFrame.
# 8. Drop duplicate rows from a DataFrame.
# 9. Merge two DataFrames on a common column.
# 10. Create a pivot table showing mean sales by region.
# NumPy (10 Questions)
# 1. Create a NumPy array of numbers from 1 to 10.
# 2. Reshape a 1D array into 2D (3x3).
# 3. Create an identity matrix of size 4.
# 4. Perform element-wise addition of two arrays.
# 5. Find the mean and standard deviation of an array.
# 6. Replace all negative values in an array with 0.
# 7. Create an array of 10 random numbers between 0 and 1.
# 8. Extract elements greater than 5 from an array.
# 9. Stack two arrays vertically and horizontally.
# 10. Flatten a multi-dimensional array.
# Matplotlib (10 Questions)
# 1. Plot a line graph of x vs y.
# 2. Create a bar chart showing sales by product.
# 3. Plot a histogram of a given data array.
# 4. Display a scatter plot with labels and title.
# 5. Plot multiple lines on the same chart.
# 6. Show a pie chart with percentages and labels.
# 7. Create a boxplot for a dataset.
# 8. Customize plot with grid, labels, and legend.
# 9. Save a plot as an image file.
# 10. Plot a heatmap using imshow().
# Functions (5 Questions)
# 1. Write a function to return the square of a number.
# 2. Create a function that checks if a string is a palindrome.
# 3. Write a function to count vowels in a string.
# 4. Define a function that returns the factorial of a number.
# 5. Write a function that takes variable number of arguments and returns their sum.
# OOPs (5 Questions)
# 1. Create a class with a constructor and a method.
# 2. Demonstrate single inheritance using classes.
# 3. Create a class with a classmethod and staticmethod.
# 4. Overload the + operator in a class using __add__.
# 5. Use super() to call a method from the parent class.
# Python Basics (10 Questions)
# 1. Write a loop to print even numbers from 1 to 20.
# 2. Reverse a string without using slicing.
# 3. Count the frequency of each word in a string using a dictionary.
# 4. Add an element to a set and check membership.
# 5. Convert a list into a tuple and vice versa.
# 6. Create a dictionary with student names and their marks.
# 7. Find the max, min, and sum of a list of numbers.
# 8. Use a for loop to print each key-value pair from a dictionary.
# 9. Merge two lists into a dictionary using zip().
# 10. Demonstrate the use of break and continue in a loop.



#pandas practise questions :

import pandas as pd
import numpy as np

# Sample Data
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
    'age': [24, 30, np.nan, 45, 22, 28, 30],
    'salary': [50000, 60000, 55000, np.nan, 52000, 60000, 60000],
    'region': ['North', 'South', 'East', 'West', 'East', 'South', 'North']
}

# Create and save sample DataFrame
df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

# 1. Load a CSV file and display the first 5 rows
df = pd.read_csv('sample_data.csv')
print("1. First 5 rows:\n", df.head())

# 2. Check for null values in a DataFrame
print("\n2. Null values:\n", df.isnull())

# 3. Replace all null values with the mean of the column
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)
print("\n3. Nulls replaced with mean:\n", df)

# 4. Group data by a column and calculate the average
grouped = df.groupby('region').mean(numeric_only=True)
print("\n4. Average by region:\n", grouped)

# 5. Filter rows where column 'age' > 25
filtered = df[df['age'] > 25]
print("\n5. Age > 25:\n", filtered)

# 6. Sort the DataFrame based on column 'salary'
sorted_df = df.sort_values(by='salary', ascending=False)
print("\n6. Sorted by salary:\n", sorted_df)

# 7. Rename columns in a DataFrame
renamed_df = df.rename(columns={'name': 'employee_name', 'salary': 'monthly_salary'})
print("\n7. Renamed columns:\n", renamed_df.head())

# 8. Drop duplicate rows from a DataFrame
# Add a duplicate row to test
df = df.append(df.iloc[0], ignore_index=True)
df_no_duplicates = df.drop_duplicates()
print("\n8. Dropped duplicates:\n", df_no_duplicates)

# 9. Merge two DataFrames on a common column
df2 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Grace'],
    'department': ['HR', 'Engineering', 'Finance']
})
merged = pd.merge(df, df2, on='name', how='left')
print("\n9. Merged DataFrames:\n", merged)

# 10. Create a pivot table showing mean sales by region
sales_data = pd.DataFrame({
    'region': ['North', 'South', 'East', 'West', 'East', 'South', 'North'],
    'sales': [200, 300, 250, 400, 300, 350, 220]
})
pivot = pd.pivot_table(sales_data, values='sales', index='region', aggfunc='mean')
print("\n10. Pivot table (mean sales by region):\n", pivot)



#numpy practise questions:
import numpy as np

# 1. Create a NumPy array of numbers from 1 to 10
arr1 = np.arange(1, 11)
print("1. Array from 1 to 10:\n", arr1)

# 2. Reshape a 1D array into 2D (3x3)
arr2 = np.arange(1, 10).reshape(3, 3)
print("\n2. Reshaped 3x3 array:\n", arr2)

# 3. Create an identity matrix of size 4
identity = np.eye(4)
print("\n3. Identity matrix (4x4):\n", identity)

# 4. Perform element-wise addition of two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
add = a + b
print("\n4. Element-wise addition:\n", add)

# 5. Find the mean and standard deviation of an array
data = np.array([10, 20, 30, 40, 50])
print("\n5. Mean:", np.mean(data))
print("   Standard Deviation:", np.std(data))

# 6. Replace all negative values in an array with 0
arr3 = np.array([2, -4, 5, -1, 0])
arr3[arr3 < 0] = 0
print("\n6. Replace negatives with 0:\n", arr3)

# 7. Create an array of 10 random numbers between 0 and 1
rand_arr = np.random.rand(10)
print("\n7. Random array:\n", rand_arr)

# 8. Extract elements greater than 5 from an array
arr4 = np.array([1, 6, 3, 8, 4, 10])
filtered = arr4[arr4 > 5]
print("\n8. Elements > 5:\n", filtered)

# 9. Stack two arrays vertically and horizontally
arr5 = np.array([[1, 2], [3, 4]])
arr6 = np.array([[5, 6], [7, 8]])
vstack = np.vstack((arr5, arr6))
hstack = np.hstack((arr5, arr6))
print("\n9. Vertical stack:\n", vstack)
print("   Horizontal stack:\n", hstack)

# 10. Flatten a multi-dimensional array
multi_arr = np.array([[1, 2], [3, 4]])
flat = multi_arr.flatten()
print("\n10. Flattened array:\n", flat)





#metplotlib practise questions:
import matplotlib.pyplot as plt
import numpy as np

# 1. Plot a line graph of x vs y
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.figure()
plt.plot(x, y)
plt.title("1. Line Graph")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.show()

# 2. Create a bar chart showing sales by product
products = ['A', 'B', 'C', 'D']
sales = [100, 150, 80, 200]
plt.figure()
plt.bar(products, sales, color='skyblue')
plt.title("2. Sales by Product")
plt.xlabel("Product")
plt.ylabel("Sales")
plt.show()

# 3. Plot a histogram of a given data array
data = np.random.randn(1000)
plt.figure()
plt.hist(data, bins=20, color='green', edgecolor='black')
plt.title("3. Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 4. Display a scatter plot with labels and title
x = np.random.rand(50)
y = np.random.rand(50)
plt.figure()
plt.scatter(x, y, color='red')
plt.title("4. Scatter Plot")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.grid(True)
plt.show()

# 5. Plot multiple lines on the same chart
x = np.arange(0, 10, 0.1)
plt.figure()
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.title("5. Multiple Lines")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# 6. Show a pie chart with percentages and labels
labels = ['Apple', 'Banana', 'Cherry', 'Dates']
sizes = [30, 25, 25, 20]
plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("6. Pie Chart")
plt.axis('equal')  # Equal aspect ratio ensures pie is a circle.
plt.show()

# 7. Create a boxplot for a dataset
data = [np.random.normal(0, std, 100) for std in range(1, 4)]
plt.figure()
plt.boxplot(data)
plt.title("7. Boxplot")
plt.xlabel("Dataset")
plt.ylabel("Value")
plt.show()

# 8. Customize plot with grid, labels, and legend
x = np.linspace(0, 10, 100)
plt.figure()
plt.plot(x, np.exp(x), label='exp(x)', color='purple')
plt.title("8. Customized Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend()
plt.show()

# 9. Save a plot as an image file
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9], label="Line")
plt.title("9. Saved Plot")
plt.legend()
plt.savefig("saved_plot.png")
plt.close()
print("9. Plot saved as 'saved_plot.png'")

# 10. Plot a heatmap using imshow()
matrix = np.random.rand(5, 5)
plt.figure()
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.title("10. Heatmap")
plt.colorbar(label='Intensity')
plt.show()


#functions baesd questions:

# 1. Function to return the square of a number
def square(num):
    return num ** 2

print("1. Square of 5:", square(5))


# 2. Function to check if a string is a palindrome
def is_palindrome(s):
    s = s.lower().replace(" ", "")  # ignore case and spaces
    return s == s[::-1]

print("2. Is 'Radar' a palindrome?:", is_palindrome("Radar"))


# 3. Function to count vowels in a string
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

print("3. Number of vowels in 'Hello World':", count_vowels("Hello World"))


# 4. Function that returns the factorial of a number
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

print("4. Factorial of 5:", factorial(5))


# 5. Function that takes variable number of arguments and returns their sum
def sum_all(*args):
    return sum(args)

print("5. Sum of 1, 2, 3, 4:", sum_all(1, 2, 3, 4))




#oops practise questions:
# 1. Create a class with a constructor and a method
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, my name is {self.name}"

p1 = Person("Alice")
print("1.", p1.greet())


# 2. Demonstrate single inheritance using classes
class Animal:
    def speak(self):
        return "Animal speaks"

class Dog(Animal):  # Single inheritance
    def bark(self):
        return "Dog barks"

d = Dog()
print("2.", d.speak(), "and", d.bark())


# 3. Create a class with a classmethod and staticmethod
class Utility:
    count = 0

    @classmethod
    def increment(cls):
        cls.count += 1
        return cls.count

    @staticmethod
    def greet(name):
        return f"Hello, {name}!"

print("3. Classmethod result:", Utility.increment())
print("   Staticmethod result:", Utility.greet("Bob"))


# 4. Overload the + operator in a class using __add__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

p1 = Point(1, 2)
p2 = Point(3, 4)
print("4. Overloaded + result:", p1 + p2)


# 5. Use super() to call a method from the parent class
class Parent:
    def show(self):
        return "Parent method"

class Child(Parent):
    def show(self):
        return f"Child overrides — {super().show()}"

c = Child()
print("5.", c.show())


#loops practise questions:
# 1. Write a loop to print even numbers from 1 to 20
print("1. Even numbers from 1 to 20:")
for i in range(1, 21):
    if i % 2 == 0:
        print(i, end=' ')
print("\n")

# 2. Reverse a string without using slicing
s = "Python"
reversed_str = ""
for char in s:
    reversed_str = char + reversed_str
print("2. Reversed string of 'Python':", reversed_str)

# 3. Count the frequency of each word in a string using a dictionary
text = "this is a test this is only a test"
words = text.split()
freq = {}
for word in words:
    freq[word] = freq.get(word, 0) + 1
print("3. Word frequency:\n", freq)

# 4. Add an element to a set and check membership
my_set = {1, 2, 3}
my_set.add(4)
print("4. Set after adding 4:", my_set)
print("   Is 2 in set?", 2 in my_set)

# 5. Convert a list into a tuple and vice versa
my_list = [10, 20, 30]
my_tuple = tuple(my_list)
new_list = list(my_tuple)
print("5. Tuple:", my_tuple)
print("   Back to list:", new_list)

# 6. Create a dictionary with student names and their marks
students = {"Alice": 85, "Bob": 90, "Charlie": 78}
print("6. Student Marks Dictionary:\n", students)

# 7. Find the max, min, and sum of a list of numbers
nums = [4, 8, 15, 16, 23, 42]
print("7. Max:", max(nums), "Min:", min(nums), "Sum:", sum(nums))

# 8. Use a for loop to print each key-value pair from a dictionary
print("8. Key-Value pairs in dictionary:")
for name, mark in students.items():
    print(f"{name}: {mark}")

# 9. Merge two lists into a dictionary using zip()
keys = ['name', 'age', 'city']
values = ['Alice', 25, 'New York']
merged_dict = dict(zip(keys, values))
print("9. Merged dictionary:\n", merged_dict)

# 10. Demonstrate the use of break and continue in a loop
print("10. Loop with break and continue:")
for i in range(1, 10):
    if i == 5:
        print("  Skipping 5 (continue)")
        continue
    if i == 8:
        print("  Breaking at 8")
        break
    print("  Number:", i)

import csv
import pandas as pd
import matplotlib.pyplot as plt
#creating csv file with student data
data = [
    ['RollNo', 'Name', 'Gender', 'Attendance', 'InternalMarks', 'ExternalMarks'],
    [101, 'Ravi', 'M', 85, 32, 55],
    [102, 'Priya', 'F', 92, 38, 58],
    [103, 'Kiran', 'M', 78, 28, 45],
    [104, 'Anu', 'F', 95, 40, 60],
    [105, 'Sita', 'F', 88, 35, 54],
    [106, 'Raj', 'M', 70, 25, 40],
    [107, 'Divya', 'F', 90, 37, 57],
    [108, 'Arun', 'M', 82, 30, 48],
    [109, 'Keerthi', 'F', 96, 39, 59],
    [110, 'Naveen', 'M', 75, 27, 44]
]
with open('student_data.csv', 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(data)
#TASK-A
#importing the csv file into pandas dataframe
df=pd.read_csv('student_data.csv')
print(df.to_string())
print(df.head())
print(df.tail())
rows, cols = df.shape
print("Number of rows:", rows)
print("Number of columns:", cols)
print("Column names:", df.columns.tolist())
print("Data types of each column:", df.dtypes)
df.info()
#TASK-B
print("Mean of Attendance:", df['Attendance'].mean())
print("Mean of InternalMarks:", df['InternalMarks'].mean())
print("Mean of ExternalMarks:", df['ExternalMarks'].mean())
print("Maximum internal marks:", df['InternalMarks'].max())
print("Minimum internal marks:", df['InternalMarks'].min())
print("Maximum external marks:", df['ExternalMarks'].max())
print("Minimum external marks:", df['ExternalMarks'].min()) 
print("Standard deviation of Attendance:", df['Attendance'].std())
print(df.describe())
#TASK-C
df['TotalMarks'] = df['InternalMarks'] + df['ExternalMarks']
df['Percentage'] = (df['TotalMarks'] / 100) * 100
print(df.to_string())
#TASK-D
print("Students with attendance greater than 90%:",df[df['Attendance'] > 90])
print("Students with total marks greater than 90:",df[df['TotalMarks'] > 90 ])
print("Female students:",df[df['Gender']== 'F' ])
print("Male students:",df[df['Gender']== 'M' ])
print("Students with less than 50 External marks:",df[df['ExternalMarks'] < 50])
#TASK-E
print("Student with highest total marks:",df[df['TotalMarks'] == df['TotalMarks'].max()])
print("Student with lowest total marks:",df[df['TotalMarks'] == df['TotalMarks'].min()])
print("The top 3 students based on total marks:",df.nlargest(3, 'TotalMarks').to_string())
print("The bottom 3 students based on total marks:",df.nsmallest(3, 'TotalMarks').to_string())
num_of_students_greater_than_average = df[df['TotalMarks']> df['TotalMarks'].mean()].shape[0]
print("Number of students with total marks greater than average:", num_of_students_greater_than_average)
#TASK-F
print("Average marks of male students:", df[df['Gender'] == 'M']['TotalMarks'].mean())
print("Average marks of female students:", df[df['Gender'] == 'F']['TotalMarks'].mean())
lol =df[df['Gender']== 'M']['Attendance'].mean()
xoxo = df[df['Gender']== 'F']['Attendance'].mean()
if lol > xoxo:
    print("Male students have better attendance than female students.")
elif lol < xoxo:
    print("Female students have better attendance than male students.")
else:
    print(" Male and Female students have the same attendance.")
#TASK-G
df['TotalMarks'].plot(kind='bar', title='Total Marks of Students', xlabel='RollNo', ylabel='Total Marks', color='lavender')
plt.show()
df['Attendance'].plot(kind='hist', title='Attendance of Students', xlabel='Attendance', ylabel='Frequency', color='lightblue')
plt.show()
df['Gender'].value_counts().plot(kind='pie', title='Gender of students', colors=['lightcoral', 'lightskyblue'])
plt.show()
df.plot.scatter(x='Attendance', y='TotalMarks', title='Attendance vs Total Marks', color='green')
plt.show() 
#TASK-H
#Three Observations:
#Female students scored better overall than male students.
#Students with high attendance usually got higher total marks.
#Anu got the highest total marks, while Raj got the lowest.
#Attendance vs Academic Performance
#Yes, higher attendance seems to be related to better academic performance.
#Most students with attendance above 90% scored high marks, 
#while students with lower attendance generally scored fewer marks.
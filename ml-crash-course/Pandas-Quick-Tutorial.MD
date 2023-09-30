# Pandas UltraQuick Tutorial

#@title Copyright 2020 Google LLC. Double-click here for license information.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Colabs

Machine Learning Crash Course uses Colaboratories (Colabs) for all programming exercises. Colab is Google's implementation of [Jupyter Notebook](https://jupyter.org/). For more information about Colabs and how to use them, go to [Welcome to Colaboratory](https://research.google.com/colaboratory).

# Pandas DataFrame UltraQuick Tutorial

This Colab introduces [**DataFrames**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html), which are the central data structure in the pandas API. This Colab is not a comprehensive DataFrames tutorial.  Rather, this Colab provides a very quick introduction to the parts of DataFrames required to do the other Colab exercises in Machine Learning Crash Course.

A DataFrame is similar to an in-memory spreadsheet. Like a spreadsheet:

  * A DataFrame stores data in cells. 
  * A DataFrame has named columns (usually) and numbered rows.

## Import NumPy and pandas modules

Run the following code cell to import the NumPy and pandas modules. 

```py
import numpy as np
import pandas as pd
```

## Creating a DataFrame

The following code cell creates a simple DataFrame containing 10 cells organized as follows:

  * 5 rows
  * 2 columns, one named `temperature` and the other named `activity`

The following code cell instantiates a `pd.DataFrame` class to generate a DataFrame. The class takes two arguments:

  * The first argument provides the data to populate the 10 cells. The code cell calls `np.array` to generate the 5x2 NumPy array.
  * The second argument identifies the names of the two columns.

**Note**: Do not redefine variables in the following code cell. Subsequent code cells use these variables.

```py
# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(my_dataframe)

#    temperature  activity
# 0            0         3
# 1           10         7
# 2           20         9
# 3           30        14
# 4           40        15
```

## Adding a new column to a DataFrame

You may add a new column to an existing pandas DataFrame just by assigning values to a new column name. For example, the following code creates a third column named `adjusted` in `my_dataframe`: 

```py
# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire DataFrame
print(my_dataframe)

#    temperature  activity  adjusted
# 0            0         3         5
# 1           10         7         9
# 2           20         9        11
# 3           30        14        16
# 4           40        15        17
```

## Specifying a subset of a DataFrame

Pandas provide multiples ways to isolate specific rows, columns, slices or cells in a DataFrame. 

```py
print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n') #1 inclusive to 4 exclusive

print("Column 'temperature':")
print(my_dataframe['temperature'])

# Rows #0, #1, and #2:
#    temperature  activity  adjusted
# 0            0         3         5
# 1           10         7         9
# 2           20         9        11 
# 
# Row #2:
#    temperature  activity  adjusted
# 2           20         9        11 
# 
# Rows #1, #2, and #3:
#    temperature  activity  adjusted
# 1           10         7         9
# 2           20         9        11
# 3           30        14        16 
# 
# Column 'temperature':
# 0     0
# 1    10
# 2    20
# 3    30
# 4    40
# Name: temperature, dtype: int64∏
```

## Task 1: Create a DataFrame

Do the following:

  1. Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named `Eleanor`,  `Chidi`, `Tahani`, and `Jason`.  Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

  2. Output the following:

     * the entire DataFrame
     * the value in the cell of row #1 of the `Eleanor` column

  3. Create a fifth column named `Janet`, which is populated with the row-by-row sums of `Tahani` and `Jason`.

To complete this task, it helps to know the NumPy basics covered in the NumPy UltraQuick Tutorial. 


# Write your code here.

```py
column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

data = []
for i in range(3):
  data.append(np.random.randint(low=0, high=100, size=4))
dataframe = pd.DataFrame(data=data, columns=column_names)

print("Entire Dataframe: ")
print(dataframe, '\n')

print('Row #1 of the Eleanor column')
print(dataframe.iloc[1, 0], '\n') # note index is [row, column]

dataframe["Janet"] = dataframe['Tahani'] + dataframe['Jason']
print("Entire Dataframe with Janet: ")
print(dataframe, '\n')

# Entire Dataframe: 
#    Eleanor  Chidi  Tahani  Jason
# 0        8     67       9     20
# 1       18     15      48      5
# 2       20     22      31     89 
# 
# Row #1 of the Eleanor column
# 18 
# 
# Entire Dataframe with Janet: 
#    Eleanor  Chidi  Tahani  Jason  Janet
# 0        8     67       9     20     29
# 1       18     15      48      5     53
# 2       20     22      31     89    120 
```

Model answer:

```py
# Create a Python list that holds the names of the four columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create a 3x4 numpy array, each cell populated with a random integer.
my_data = np.random.randint(low=0, high=101, size=(3, 4))

# Create a DataFrame.
df = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire DataFrame
print(df)

# Print the value in row #1 of the Eleanor column.
print("\nSecond row of the Eleanor column: %d\n" % df['Eleanor'][1])

# Create a column named Janet whose contents are the sum
# of two other columns.
df['Janet'] = df['Tahani'] + df['Jason']

# Print the enhanced DataFrame
print(df)
```

## Copying a DataFrame (optional)

Pandas provides two different ways to duplicate a DataFrame:

* **Referencing.** If you assign a DataFrame to a new variable, any change to the DataFrame or to the new variable will be reflected in the other. 
* **Copying.** If you call the `pd.DataFrame.copy` method, you create a true independent copy.  Changes to the original DataFrame or to the copy will not be reflected in the other. 

The difference is subtle, but important.

```py
# Create a reference by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_dataframe = dataframe

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % dataframe['Jason'][1])
print("  Starting value of reference_to_dataframe: %d\n" % reference_to_dataframe['Jason'][1])

# Modify a cell in dataframe.
dataframe.at[1, 'Jason'] = dataframe['Jason'][1] + 5
print("  Updated dataframe: %d" % dataframe['Jason'][1])
print("  Updated reference_to_dataframe: %d\n\n" % reference_to_dataframe['Jason'][1])

# Create a true copy of my_dataframe
print("Experiment with a true copy:")
copy_of_my_dataframe = my_dataframe.copy()

# Print the starting value of a particular cell.
print("  Starting value of my_dataframe: %d" % my_dataframe['activity'][1])
print("  Starting value of copy_of_my_dataframe: %d\n" % copy_of_my_dataframe['activity'][1])

# Modify a cell in dataframe.
my_dataframe.at[1, 'activity'] = my_dataframe['activity'][1] + 3
print("  Updated my_dataframe: %d" % my_dataframe['activity'][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_my_dataframe['activity'][1])

# Experiment with a reference:
#   Starting value of df: 5
#   Starting value of reference_to_dataframe: 5
# 
#   Updated dataframe: 10
#   Updated reference_to_dataframe: 10
# 
# 
# Experiment with a true copy:
#   Starting value of my_dataframe: 7
#   Starting value of copy_of_my_dataframe: 7
# 
#   Updated my_dataframe: 10
#   copy_of_my_dataframe does not get updated: 7
```


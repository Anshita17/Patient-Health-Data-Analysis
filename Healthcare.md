# Import Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

```

# Load the dataset


```python
data=pd.read_csv('/Users/anshitapriyadarshini/Downloads/Health care/Healthcare data.csv')
```

# shape


```python
data.shape
```




    (51000, 15)



# Display


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Blood_Pressure_Systolic</th>
      <th>Blood_Pressure_Diastolic</th>
      <th>Heart_Rate</th>
      <th>Cholesterol_Level</th>
      <th>Medical_Conditions</th>
      <th>Medications</th>
      <th>Visit_Date</th>
      <th>Diagnosis</th>
      <th>Hospital_Visits_Past_Year</th>
      <th>BMI</th>
      <th>Smoker_Status</th>
      <th>Physical_Activity_Level</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5f437e43-a631-4077-98c0-a425bc40d229</td>
      <td>37</td>
      <td>Female</td>
      <td>158</td>
      <td>99</td>
      <td>110</td>
      <td>224</td>
      <td>Asthma</td>
      <td>Statins</td>
      <td>03/06/24</td>
      <td>Follow-up</td>
      <td>7</td>
      <td>25.5</td>
      <td>Yes</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dc84404a-4df8-447b-9b69-858b955295ff</td>
      <td>32</td>
      <td>Female</td>
      <td>156</td>
      <td>71</td>
      <td>72</td>
      <td>185</td>
      <td>Hypertension</td>
      <td>Statins</td>
      <td>25/04/24</td>
      <td>Follow-up</td>
      <td>7</td>
      <td>21.6</td>
      <td>Yes</td>
      <td>Sedentary</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a2186ae7-f70c-4403-9460-a4149509d84b</td>
      <td>57</td>
      <td>Female</td>
      <td>119</td>
      <td>69</td>
      <td>85</td>
      <td>198</td>
      <td>Asthma</td>
      <td>No Medication</td>
      <td>10/09/24</td>
      <td>Routine Check</td>
      <td>2</td>
      <td>30.2</td>
      <td>Yes</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d5ba3bf9-224a-49ff-9663-afa858fb3222</td>
      <td>38</td>
      <td>Other</td>
      <td>130</td>
      <td>118</td>
      <td>69</td>
      <td>195</td>
      <td>Asthma</td>
      <td>Metformin</td>
      <td>21/09/24</td>
      <td>Routine Check</td>
      <td>8</td>
      <td>30.8</td>
      <td>No</td>
      <td>Active</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3bcc4f12-5f2a-4090-a320-779580432058</td>
      <td>42</td>
      <td>Male</td>
      <td>152</td>
      <td>91</td>
      <td>94</td>
      <td>278</td>
      <td>Asthma</td>
      <td>Lisinopril</td>
      <td>10/08/24</td>
      <td>Infection</td>
      <td>5</td>
      <td>24.3</td>
      <td>No</td>
      <td>Sedentary</td>
    </tr>
  </tbody>
</table>
</div>



# Overview


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51000 entries, 0 to 50999
    Data columns (total 15 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   Patient_ID                 51000 non-null  object 
     1   Age                        51000 non-null  int64  
     2   Gender                     51000 non-null  object 
     3   Blood_Pressure_Systolic    51000 non-null  int64  
     4   Blood_Pressure_Diastolic   51000 non-null  int64  
     5   Heart_Rate                 51000 non-null  int64  
     6   Cholesterol_Level          51000 non-null  int64  
     7   Medical_Conditions         51000 non-null  object 
     8   Medications                51000 non-null  object 
     9   Visit_Date                 51000 non-null  object 
     10  Diagnosis                  51000 non-null  object 
     11  Hospital_Visits_Past_Year  51000 non-null  int64  
     12  BMI                        51000 non-null  float64
     13  Smoker_Status              51000 non-null  object 
     14  Physical_Activity_Level    51000 non-null  object 
    dtypes: float64(1), int64(6), object(8)
    memory usage: 5.8+ MB


# Missing values


```python
data.isnull().sum()
```




    Patient_ID                   0
    Age                          0
    Gender                       0
    Blood_Pressure_Systolic      0
    Blood_Pressure_Diastolic     0
    Heart_Rate                   0
    Cholesterol_Level            0
    Medical_Conditions           0
    Medications                  0
    Visit_Date                   0
    Diagnosis                    0
    Hospital_Visits_Past_Year    0
    BMI                          0
    Smoker_Status                0
    Physical_Activity_Level      0
    dtype: int64



# Summary


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Blood_Pressure_Systolic</th>
      <th>Blood_Pressure_Diastolic</th>
      <th>Heart_Rate</th>
      <th>Cholesterol_Level</th>
      <th>Hospital_Visits_Past_Year</th>
      <th>BMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>51000.000000</td>
      <td>51000.000000</td>
      <td>51000.000000</td>
      <td>51000.000000</td>
      <td>51000.000000</td>
      <td>51000.000000</td>
      <td>51000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>53.524686</td>
      <td>134.389176</td>
      <td>89.443196</td>
      <td>89.338510</td>
      <td>224.741765</td>
      <td>4.473314</td>
      <td>26.770871</td>
    </tr>
    <tr>
      <th>std</th>
      <td>20.830735</td>
      <td>25.948291</td>
      <td>17.260013</td>
      <td>17.272534</td>
      <td>43.381797</td>
      <td>2.871291</td>
      <td>4.777884</td>
    </tr>
    <tr>
      <th>min</th>
      <td>18.000000</td>
      <td>90.000000</td>
      <td>60.000000</td>
      <td>60.000000</td>
      <td>150.000000</td>
      <td>0.000000</td>
      <td>18.500000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35.000000</td>
      <td>112.000000</td>
      <td>75.000000</td>
      <td>74.000000</td>
      <td>187.000000</td>
      <td>2.000000</td>
      <td>22.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>54.000000</td>
      <td>134.000000</td>
      <td>89.000000</td>
      <td>89.000000</td>
      <td>225.000000</td>
      <td>4.000000</td>
      <td>26.800000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>72.000000</td>
      <td>157.000000</td>
      <td>104.000000</td>
      <td>104.000000</td>
      <td>262.000000</td>
      <td>7.000000</td>
      <td>30.900000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>89.000000</td>
      <td>179.000000</td>
      <td>119.000000</td>
      <td>119.000000</td>
      <td>299.000000</td>
      <td>9.000000</td>
      <td>35.000000</td>
    </tr>
  </tbody>
</table>
</div>



# Data Types


```python
data.dtypes
```




    Patient_ID                    object
    Age                            int64
    Gender                        object
    Blood_Pressure_Systolic        int64
    Blood_Pressure_Diastolic       int64
    Heart_Rate                     int64
    Cholesterol_Level              int64
    Medical_Conditions            object
    Medications                   object
    Visit_Date                    object
    Diagnosis                     object
    Hospital_Visits_Past_Year      int64
    BMI                          float64
    Smoker_Status                 object
    Physical_Activity_Level       object
    dtype: object



# Data Type Conversions


```python
Categorical = ['Patient_ID', 'Gender', 'Medical_Conditions', 'Medications', 'Diagnosis', 'Smoker_Status', 'Physical_Activity_Level']
data[Categorical] = data[Categorical].astype('category')
```


```python
data['Cholesterol_Level']=data['Cholesterol_Level'].astype('float')
```


```python
data['Visit_Date'] = pd.to_datetime(data['Visit_Date'])
```

    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/3656220163.py:1: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
      data['Visit_Date'] = pd.to_datetime(data['Visit_Date'])


# View Particular Column


```python
data['Medical_Conditions'].describe()
```




    count        51000
    unique           4
    top       Diabetes
    freq         12986
    Name: Medical_Conditions, dtype: object



# Columns


```python
data.columns
```




    Index(['Patient_ID', 'Age', 'Gender', 'Blood_Pressure_Systolic',
           'Blood_Pressure_Diastolic', 'Heart_Rate', 'Cholesterol_Level',
           'Medical_Conditions', 'Medications', 'Visit_Date', 'Diagnosis',
           'Hospital_Visits_Past_Year', 'BMI', 'Smoker_Status',
           'Physical_Activity_Level'],
          dtype='object')




```python
data['Gender'].describe()
```




    count     51000
    unique        3
    top        Male
    freq      17077
    Name: Gender, dtype: object



# EDA
## Gender


```python
ax = sns.countplot(x='Gender',data=data)

```


    
![png](output_26_0.png)
    



```python
ax = sns.countplot(x='Gender', data=data)

# Adding bar labels
for bars in ax.containers:
    ax.bar_label(bars)


```


    
![png](output_27_0.png)
    


# Medical Conditions


```python
ax = sns.countplot(x='Medications',data=data)

for bars in ax.containers:
    ax.bar_label(bars)


```


    
![png](output_29_0.png)
    



```python

# Exclude 'NO Medication' from the plot by specifying the order
ax = sns.countplot(
    x='Medications', 
    data=data, 
    order=[med for med in data['Medications'].unique() if med != 'No Medication']
)

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Show the plot
plt.show()


```


    
![png](output_30_0.png)
    


# Medical Conditions


```python
ax = sns.countplot(
    x='Medical_Conditions', 
    data=data)
# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Show the plot
plt.show()

```


    
![png](output_32_0.png)
    



```python
ax = sns.countplot(
    x='Medical_Conditions', 
    data=data,
order=[med for med in data['Medical_Conditions'].unique() if med  != 'No Medical Condition'])

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Show the plot
plt.show()

```


    
![png](output_33_0.png)
    


# Diagnosis


```python
ax = sns.countplot(x='Diagnosis',data=data)

for bars in ax.containers:
    ax.bar_label(bars)


```


    
![png](output_35_0.png)
    


# Smoker Status


```python
ax = sns.countplot(x='Smoker_Status',data=data)

for bars in ax.containers:
    ax.bar_label(bars)

```


    
![png](output_37_0.png)
    


# Physical Activity Level


```python
ax = sns.countplot(x='Physical_Activity_Level',data=data)

for bars in ax.containers:
    ax.bar_label(bars)

```


    
![png](output_39_0.png)
    


# Medical Conditions by Gender


```python
# Group the data by Gender and Medical_Conditions, then count the occurrences
Medbygender = data.groupby(['Gender', 'Medical_Conditions'],observed=False).size().reset_index(name='Count')


# Create a barplot
ax = sns.barplot(x='Medical_Conditions', y='Count', hue='Gender', data=Medbygender)

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Rotate the x-axis labels for better readability

plt.xticks(rotation=45, ha='right')

plt.legend(title='Gender', bbox_to_anchor=(1, 1), loc='upper left')


# Show the plot
plt.show()

```


    
![png](output_41_0.png)
    


# Age Bins


```python
# Define the age bins and corresponding labels
age_bins = [18, 29, 39, 49, 59, 69, 79, 89]  # You can adjust these bins as needed
age_labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

# Create a new column 'Age_Group' to categorize ages
data['Age_Group'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, right=True)

# Check the result
print(data[['Age', 'Age_Group']].head(10))

```

       Age Age_Group
    0   37     30-39
    1   32     30-39
    2   57     50-59
    3   38     30-39
    4   42     40-49
    5   64     60-69
    6   73     70-79
    7   22     18-29
    8   88     80-89
    9   51     50-59


#  Age Groups and Gender


```python


# Increase figure size
plt.figure(figsize=(14, 8))  # Larger width and height for better clarity

# Create a countplot for Age Group vs Gender
ax = sns.countplot(x='Age_Group', data=data)

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()  # Automatically adjusts layout to avoid clipping of labels/legend
plt.show()

```


    
![png](output_45_0.png)
    



```python


# Increase figure size
plt.figure(figsize=(14, 8))  # Larger width and height for better clarity

# Create a countplot for Age Group vs Gender
ax = sns.countplot(x='Age_Group',hue='Gender', data=data)

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')

# Adjust the legend to avoid overlap (move outside the plot)
plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()  # Automatically adjusts layout to avoid clipping of labels/legend
plt.show()

```


    
![png](output_46_0.png)
    


# Medical Conditions by Age group


```python
# Group the data by Gender and Medical_Conditions, then count the occurrences
Medbygender = data.groupby(['Age_Group', 'Medical_Conditions'],observed=False).size().reset_index(name='Count')


# Create a barplot
plt.figure(figsize=(16,8))
ax = sns.barplot(x='Medical_Conditions', y='Count',hue='Age_Group',  data=Medbygender)

# Add bar labels
for bars in ax.containers:
    ax.bar_label(bars)

# Rotate the x-axis labels for better readability

plt.xticks(rotation=45, ha='right')

plt.legend(title='Gender', bbox_to_anchor=(1, 1), loc='upper left')


# Show the plot
plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


# Blood Pressure Across Age groups


```python


# Increase the figure size for better readability
plt.figure(figsize=(14, 8))

# Create a boxplot for Blood Pressure (Systolic) vs Age Group and Gender
sns.boxplot(x='Age_Group', y='Blood_Pressure_Systolic', hue='Gender', data=data)

# Add title and labels
plt.title('Blood Pressure (Systolic) vs Age Group and Gender', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Systolic Blood Pressure (mmHg)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the legend to avoid overlap (move outside the plot)
plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_50_0.png)
    



```python
plt.figure(figsize=(14, 8))

# Create a boxplot for Blood Pressure (Diastolic) vs Age Group and Gender
sns.boxplot(x='Age_Group', y='Blood_Pressure_Diastolic', hue='Gender', data=data)

# Add title and labels
plt.title('Blood Pressure (Diastolic) vs Age Group and Gender', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Diastolic Blood Pressure (mmHg)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the legend to avoid overlap (move outside the plot)
plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_51_0.png)
    



```python
# Reshape the data to have 'Blood_Pressure_Type' as a column (Systolic or Diastolic)
data_melted = data.melt(id_vars=['Age_Group',], value_vars=['Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic'], 
                        var_name='Blood_Pressure_Type', value_name='Blood_Pressure_Value')

# Increase the figure size for better readability
plt.figure(figsize=(14, 8))

# Create the boxplot comparing Systolic vs Diastolic across Age Group and Gender
sns.boxplot(x='Age_Group', y='Blood_Pressure_Value', hue='Blood_Pressure_Type', data=data_melted)

# Add title and labels
plt.title('Comparison of Systolic and Diastolic Blood Pressure by Age Group', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the legend to avoid overlap (move outside the plot)
plt.legend(title='Blood Pressure Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_52_0.png)
    



```python
# Reshape the data to have 'Blood_Pressure_Type' as a column (Systolic or Diastolic)
data_melted = data.melt(id_vars=['Gender'], value_vars=['Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic'], 
                        var_name='Blood_Pressure_Type', value_name='Blood_Pressure_Value')

# Increase the figure size for better readability
plt.figure(figsize=(14, 8))

# Create the boxplot comparing Systolic vs Diastolic across Age Group and Gender
sns.boxplot(x='Gender', y='Blood_Pressure_Value', hue='Blood_Pressure_Type', data=data_melted)

# Add title and labels
plt.title('Comparison of Systolic and Diastolic Blood Pressure by Gender', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust the legend to avoid overlap (move outside the plot)
plt.legend(title='Blood Pressure Type', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_53_0.png)
    



```python
# Increase the figure size for better readability
plt.figure(figsize=(14, 8))

# Create a boxplot for Systolic Blood Pressure vs Medical Conditions
sns.boxplot(x='Medical_Conditions', y='Blood_Pressure_Systolic', data=data)
plt.title('Systolic Blood Pressure by Medical Conditions', fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Systolic Blood Pressure (mmHg)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Create a boxplot for Diastolic Blood Pressure vs Medical Conditions
plt.figure(figsize=(14, 8))
sns.boxplot(x='Medical_Conditions', y='Blood_Pressure_Diastolic', data=data)
plt.title('Diastolic Blood Pressure by Medical Conditions', fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Diastolic Blood Pressure (mmHg)', fontsize=12)

plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

```


    
![png](output_54_0.png)
    



    
![png](output_54_1.png)
    


# Heart rate


```python
# Increase the figure size for better readability
plt.figure(figsize=(4, 4))

# Create a histogram to visualize the distribution of Heart Rate
sns.histplot(data['Heart_Rate'],kde=True, bins=30, color='blue')

# Add title and labels
plt.title('Distribution of Heart Rate', fontsize=16)
plt.xlabel('Heart Rate (bpm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()
```


    
![png](output_56_0.png)
    



```python
plt.figure(figsize=(2, 4))
sns.boxplot(y=data['Heart_Rate'], color='green')

# Add title and labels
plt.title('Boxplot of Heart Rate', fontsize=16)
plt.xlabel('Heart Rate (bpm)', fontsize=12)

plt.tight_layout()
plt.show()

```


    
![png](output_57_0.png)
    



```python
 data['Heart_Rate'].describe()
```




    count    51000.000000
    mean        89.338510
    std         17.272534
    min         60.000000
    25%         74.000000
    50%         89.000000
    75%        104.000000
    max        119.000000
    Name: Heart_Rate, dtype: float64




```python
# Calculate average heart rate for each physical activity level
average_heart_rate_by_condition = data.groupby('Medical_Conditions',observed=False)['Heart_Rate'].mean().reset_index()

# Display the result
print(average_heart_rate_by_condition)



```

         Medical_Conditions  Heart_Rate
    0                Asthma   89.060642
    1              Diabetes   89.311412
    2          Hypertension   89.505614
    3  No Medical Condition   89.474831



```python
# Calculate the overall average heart rate
average_heart_rate = data['Heart_Rate'].mean()

print(f"Overall Average Heart Rate: {average_heart_rate} bpm")

# Flagging patients with heart rate above average
data['Heart_Rate_Above_Avg'] = data['Heart_Rate'] > average_heart_rate


# Filter data for patients with heart rate above average
above_avg_heart_rate_data = data[data['Heart_Rate_Above_Avg']]

# Group by medical conditions and count the occurrences
medical_conditions_count = above_avg_heart_rate_data['Medical_Conditions'].value_counts()

print(medical_conditions_count)

# Increase the figure size for better readability
plt.figure(figsize=(4, 4))

# Create a bar plot for medical conditions in patients with above average heart rate
sns.countplot(x='Medical_Conditions', data=above_avg_heart_rate_data, palette='viridis')

# Add title and labels
plt.title('Medical Conditions for Patients with Heart Rate Above Average', fontsize=16)
plt.xlabel('Medical Condition', fontsize=12)
plt.ylabel('Count of Patients', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()

```

    Overall Average Heart Rate: 89.33850980392157 bpm
    Medical_Conditions
    Hypertension            6437
    Diabetes                6383
    No Medical Condition    6252
    Asthma                  6166
    Name: count, dtype: int64


    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/854138610.py:22: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      sns.countplot(x='Medical_Conditions', data=above_avg_heart_rate_data, palette='viridis')



    
![png](output_60_2.png)
    


# Cholestrol Level


```python
plt.figure(figsize=(2, 4))
sns.boxplot(y=data['Cholesterol_Level'], color='green')
```




    <Axes: ylabel='Cholesterol_Level'>




    
![png](output_62_1.png)
    



```python

```


```python
# Calculate the overall average heart rate
average_Cholesterol_Level = data['Cholesterol_Level'].mean()

print(f"Overall Average Cholesterol Level: {average_Cholesterol_Level} bpm")

# Flagging patients with heart rate above average
data['Cholesterol_Level_Above_Avg'] = data['Cholesterol_Level'] > average_Cholesterol_Level


# Filter data for patients with heart rate above average
above_average_Cholesterol_Level_data = data[data['Cholesterol_Level_Above_Avg']]

# Group by medical conditions and count the occurrences
medical_conditions_count = above_average_Cholesterol_Level_data['Medical_Conditions'].value_counts()

print(medical_conditions_count)

# Increase the figure size for better readability
plt.figure(figsize=(6, 6))

# Create a bar plot for medical conditions in patients with above average heart rate
ax= sns.countplot(x='Medical_Conditions', data=above_average_Cholesterol_Level_data)
for bars in ax.containers:
    ax.bar_label(bars)

# Add title and labels
plt.title('Medical Conditions for Patients with Cholesterol Level above average', fontsize=16)
plt.xlabel('Medical Condition', fontsize=12)
plt.ylabel('Count of Patients', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show plot
plt.tight_layout()
plt.show()

```

    Overall Average Cholesterol Level: 224.74176470588236 bpm
    Medical_Conditions
    Hypertension            6535
    Diabetes                6398
    Asthma                  6343
    No Medical Condition    6303
    Name: count, dtype: int64



    
![png](output_64_1.png)
    


# BMI


```python
plt.figure(figsize=(2,4))
sns.boxplot(y=data['BMI'],color='pink')

```




    <Axes: ylabel='BMI'>




    
![png](output_66_1.png)
    



```python

# Group by Smoker_Status and calculate the proportion of patients with BMI above average
avg_bmi_by_smoker = data.groupby('Smoker_Status')['Medical_Conditions'].count().reset_index()

# Create a barplot to visualize the proportion of patients with above-average BMI by smoker status
plt.figure(figsize=(2, 4))  # Adjusting figure size for better readability
sns.barplot(x='Smoker_Status', y='Medical_Conditions', data=date, color='pink')

# Add title and labels
plt.title(' Patients with Above-Average BMI by Smoker Status', fontsize=16)
plt.xlabel('Smoker Status', fontsize=12)
plt.ylabel('Proportion with Above-Average BMI', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

```

    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/761456739.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      avg_bmi_by_smoker = data.groupby('Smoker_Status')['Medical_Conditions'].count().reset_index()



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[109], line 6
          4 # Create a barplot to visualize the proportion of patients with above-average BMI by smoker status
          5 plt.figure(figsize=(2, 4))  # Adjusting figure size for better readability
    ----> 6 sns.barplot(x='Smoker_Status', y='Medical_Conditions', data=date, color='pink')
          8 # Add title and labels
          9 plt.title(' Patients with Above-Average BMI by Smoker Status', fontsize=16)


    NameError: name 'date' is not defined



    <Figure size 200x400 with 0 Axes>



```python


# Group by Age Group and calculate the proportion of patients with BMI above average
avg_bmi_by_age_group = data.groupby('Age_Group',observed=False)['above_avg_bmi'].count().reset_index()

# Create a barplot to visualize the proportion of patients with above-average BMI by age group
plt.figure(figsize=(6, 6))  # Adjusting figure size for better readability
ax = sns.barplot(x='Age_Group', y='above_avg_bmi', data=avg_bmi_by_age_group, color='lightblue')
for bars in ax.containers:
    ax.bar_label(bars)

# Add title and labels
plt.title(' Patients with Above-Average BMI by Age Group' ,fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Proportion with Above-Average BMI', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

```

# Medical Conditions and Diagnosis


```python
# Group by Medical Conditions and Smoker Status, and count the number of patients in each group
condition_smoker_count = data.groupby(['Medical_Conditions', 'Smoker_Status',],observed=False).size().reset_index(name='Patient_Count')

# Create a barplot to visualize the count of patients by Medical Conditions and Smoker Status
plt.figure(figsize=(6, 6))  # Adjusting figure size for better readability
ax = sns.barplot(x='Medical_Conditions', y='Patient_Count', hue='Smoker_Status', data=condition_smoker_count, palette='coolwarm')
for bars in ax.containers:
    ax.bar_label(bars)
# Add title and labels
plt.title('Patients by Medical Conditions and Smoker Status', fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.legend(title='Smoker Status', bbox_to_anchor=(1.05, 1), loc='upper left')
# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_70_0.png)
    



```python
# Group by Medical Conditions and Diagnosis, and count the number of patients in each group

condition_diagnosis_count = data.groupby(['Medical_Conditions', 'Diagnosis'],observed=False).size().reset_index(name='Patient_Count')

# Create a barplot to visualize the count of patients by Medical Conditions and Diagnosis
plt.figure(figsize=(10, 6))  # Adjusting figure size for better readability
ax = sns.barplot(x='Medical_Conditions', y='Patient_Count', hue='Diagnosis', data=condition_diagnosis_count, palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
    
# Add title and labels
plt.title('Number of Patients by Medical Conditions and Diagnosis', fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_71_0.png)
    



```python
# Filter the data for only Emergency cases
emergency_data = data[data['Diagnosis'] == 'Emergency']

# Group by Medical Conditions and count the number of patients in each group
emergency_condition_count = emergency_data.groupby('Medical_Conditions',observed=False).size().reset_index(name='Patient_Count')

# Create a barplot to visualize the count of patients by Medical Conditions for Emergency cases
plt.figure(figsize=(6, 6))  # Adjusting figure size for better readability
ax= sns.barplot(x='Medical_Conditions', y='Patient_Count', data=emergency_condition_count, palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)
# Add title and labels
plt.title('Number of Emergency Patients by Medical Conditions',fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

```

    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/1847406534.py:9: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax= sns.barplot(x='Medical_Conditions', y='Patient_Count', data=emergency_condition_count, palette='Set2')



    
![png](output_72_1.png)
    


# Hospital Visits


```python
# Group by Diagnosis and sum the Hospital_Visits_Past_Year for each Diagnosis type
diagnosis_visit_count = data.groupby('Medical_Conditions',observed=False)['Hospital_Visits_Past_Year'].sum().reset_index()

# Sort the data by the total visits in descending order
diagnosis_visit_count = diagnosis_visit_count.sort_values(by='Hospital_Visits_Past_Year', ascending=False)

# Create a barplot to visualize the total number of visits for different Diagnosis types
plt.figure(figsize=(12, 6))  # Adjusting figure size for better readability
ax=sns.barplot(x='Medical_Conditions', y='Hospital_Visits_Past_Year', data=diagnosis_visit_count, palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)

# Add title and labels
plt.title('Total Number of Visits for Different Diagnosis Types',fontsize=16)
plt.xlabel('Medical Condition', fontsize=12)
plt.ylabel('Total Visits in Past Year', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

```

    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/192875344.py:9: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `x` variable to `hue` and set `legend=False` for the same effect.
    
      ax=sns.barplot(x='Medical_Conditions', y='Hospital_Visits_Past_Year', data=diagnosis_visit_count, palette='Set2')



    
![png](output_74_1.png)
    



```python

# Group by Age_Group and Gender, and sum Hospital_Visits_Past_Year
age_gender_visit_count = data.groupby(['Age_Group', 'Gender'],observed=False)['Hospital_Visits_Past_Year'].sum().reset_index()

# Create a barplot to visualize the number of visits for each combination of age group and gender
plt.figure(figsize=(13, 8))  # Adjusting figure size for better readability
ax=sns.barplot(x='Age_Group', y='Hospital_Visits_Past_Year', hue='Gender', data=age_gender_visit_count, palette='Set2')
for bars in ax.containers:
    ax.bar_label(bars)

# Add title and labels
plt.title('Hospital Visits Across Different Age Groups and Genders', fontsize=16)
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Total Hospital Visits (Past Year)', fontsize=12)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_75_0.png)
    


#  Medical Conditions and Medications


```python



# Group the data by Medications and Medical_Conditions, and count occurrences
med = data.groupby(['Medications', 'Medical_Conditions']).size().reset_index(name='Count')

# Set the figure size for better readability
plt.figure(figsize=(12, 6))

# Create the bar plot
ax = sns.barplot(x='Medical_Conditions', y='Count', hue='Medications', data=med)

for bars in ax.containers:
    ax.bar_label(bars)
# Add labels and title
plt.title('Count of Medical Conditions by Medications', fontsize=16)
plt.xlabel('Medical Conditions', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='best')
# Adjust the layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()


```

    /var/folders/j3/7w62rnrd73l6_l84llgzn7vc0000gn/T/ipykernel_1034/2206723407.py:2: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      med = data.groupby(['Medications', 'Medical_Conditions']).size().reset_index(name='Count')



    
![png](output_77_1.png)
    


# Percentage of Smokers


```python
# Calculate the total number of patients
total_patients = len(data)

# Count the number of smokers (assuming 'Smoker_Status' column has 'Smoker' and 'Non-Smoker')
smokers_count = data[data['Smoker_Status'] == 'Yes'].shape[0]

# Calculate the percentage of smokers
smokers_percentage = (smokers_count / total_patients) * 100

# Display the result
print(f"Percentage of smokers: {smokers_percentage:.2f}%")

```

    Percentage of smokers: 49.89%



```python


# Select only numerical columns
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()

# Set the figure size for better readability
plt.figure(figsize=(12, 8))

# Create a heatmap to visualize the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add title to the heatmap
plt.title('Correlation Matrix of Numerical Variables', fontsize=16)

# Show the plot
plt.tight_layout()
plt.show()

```


    
![png](output_80_0.png)
    



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

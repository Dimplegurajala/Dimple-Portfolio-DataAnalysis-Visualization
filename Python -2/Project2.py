#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
from bs4 import BeautifulSoup
import sqlite3

# URL of the website
url = "https://startuptalky.com/top-unicorn-startups-india/"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract data from the table
    table = soup.find('table')
    rows = table.find_all('tr')[1:]  # Skip the header row

    # Connect to SQLite database
    conn = sqlite3.connect("unicorn_data.db")
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS unicorn_data (
        startup_name TEXT,
        industry TEXT,
        founding_year INTEGER,
        unicorn_entry_year INTEGER,
        profit_loss_fy22 TEXT,
        current_valuation TEXT,
        acquisitions TEXT,
        status TEXT
    );
    """
    cursor.execute(create_table_query)

    # Insert data into the table
    for row in rows:
        cols = row.find_all(['td', 'th'])
        values = [col.get_text(strip=True) for col in cols]

        # Handle 'NA' in Profit/Loss FY22
        values[4] = values[4] if values[4] != 'NA' else None

        # Ensure the correct number of values (8)
        if len(values) == 8:
            cursor.execute("INSERT INTO unicorn_data VALUES (?, ?, ?, ?, ?, ?, ?, ?);", values)
        else:
            print(f"Skipping row: {values}")

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    print("Data has been successfully crawled and saved to SQLite database.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")


# In[4]:


import sqlite3

# Connect to SQLite database
conn = sqlite3.connect("unicorn_data.db")
cursor = conn.cursor()

# Retrieve data from the table
select_query = "SELECT * FROM unicorn_data;"
cursor.execute(select_query)
data = cursor.fetchall()

# Close the connection
conn.close()

# Display the data
for row in data:
    print(row)


# In[5]:


#saving the data to .csv file
import requests
from bs4 import BeautifulSoup
import csv
import sqlite3

# URL of the website
url = "https://startuptalky.com/top-unicorn-startups-india/"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract data from the table
    table = soup.find('table')
    rows = table.find_all('tr')[1:]  # Skip the header row

    # Connect to SQLite database
    conn = sqlite3.connect("unicorn_data.db")
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    create_table_query = """
    CREATE TABLE IF NOT EXISTS unicorn_data (
        startup_name TEXT,
        industry TEXT,
        founding_year INTEGER,
        unicorn_entry_year INTEGER,
        profit_loss_fy22 TEXT,
        current_valuation TEXT,
        acquisitions TEXT,
        status TEXT
    );
    """
    cursor.execute(create_table_query)

    # Insert data into the table
    for row in rows:
        cols = row.find_all(['td', 'th'])
        values = [col.get_text(strip=True) for col in cols]

        # Handle 'NA' in Profit/Loss FY22
        values[4] = values[4] if values[4] != 'NA' else None

        # Ensure the correct number of values (8)
        if len(values) == 8:
            cursor.execute("INSERT INTO unicorn_data VALUES (?, ?, ?, ?, ?, ?, ?, ?);", values)
        else:
            print(f"Skipping row: {values}")

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    # Fetch data from SQLite and write to CSV
    conn = sqlite3.connect("unicorn_data.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM unicorn_data")
    data = cursor.fetchall()

    with open("unicorn_data.csv", "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(["Startup Name", "Industry", "Founding Year", "Unicorn Entry Year", 
                            "Profit/Loss FY22", "Current Valuation", "Acquisitions", "Status"])
        # Write data
        csv_writer.writerows(data)

    print("Data has been successfully crawled and saved to SQLite database and CSV file.")
    
    # Close the connection
    conn.close()
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")


# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from SQLite database into a DataFrame
conn = sqlite3.connect("unicorn_data.db")
df = pd.read_sql_query("SELECT * FROM unicorn_data;", conn)
conn.close()

# Descriptive Analysis
overview_stats = df.describe()
print("Overview Statistics:\n", overview_stats)

# Distribution Visualization
plt.figure(figsize=(12, 6))
sns.histplot(df['founding_year'], bins=20, kde=True)
plt.title('Distribution of Founding Year')
plt.xlabel('Founding Year')
plt.ylabel('Frequency')
plt.show()




# In[7]:


# Categorical Data Analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='industry', data=df, hue='status')
plt.title('Distribution of Startups Across Industries')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# In-Depth Analysis
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Regression Analysis
plt.figure(figsize=(12, 6))
sns.regplot(x='founding_year', y='current_valuation', data=df)
plt.title('Regression Analysis: Founding Year vs Current Valuation')
plt.xlabel('Founding Year')
plt.ylabel('Current Valuation')
plt.show()


# In[4]:


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3

# Load data from SQLite database into a DataFrame
conn = sqlite3.connect("unicorn_data.db")
df = pd.read_sql_query("SELECT * FROM unicorn_data;", conn)
conn.close()

# Handle missing values (if any)
df.dropna(subset=['founding_year', 'unicorn_entry_year', 'current_valuation'], inplace=True)

# Convert 'current_valuation' to numeric format
df['current_valuation'] = df['current_valuation'].replace('[\$,]', '', regex=True).astype(float)

# Split the data into features (X) and target variable (y)
X = df[['founding_year', 'unicorn_entry_year']]
y = df['current_valuation']

# Add a constant term to the features (required for statsmodels)
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model using Ordinary Least Squares (OLS)
model = sm.OLS(y_train, X_train).fit()

# Print regression summary
print(model.summary())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the regression results
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X_test['founding_year'], y_test, color='blue', label='Actual')
ax.scatter(X_test['founding_year'], y_pred, color='red', label='Predicted')
ax.set_xlabel('Founding Year / Unicorn Entry Year')
ax.set_ylabel('Current Valuation')
ax.set_title('Regression Analysis: Founding Year and Unicorn Entry Year vs Current Valuation')
ax.legend()
plt.show()


# In[6]:


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3

# Load data from SQLite database into a DataFrame
conn = sqlite3.connect("unicorn_data.db")
df = pd.read_sql_query("SELECT * FROM unicorn_data;", conn)
conn.close()

# Handle missing values (if any)
df.dropna(subset=['founding_year', 'unicorn_entry_year', 'current_valuation'], inplace=True)

# Extract numeric part from 'current_valuation' and handle 'Billion'
df['current_valuation'] = df['current_valuation'].str.replace('[\$,]', '', regex=True)
df['current_valuation'] = pd.to_numeric(df['current_valuation'].str.replace('Billion', '').replace('-', '0')) * 1_000_000_000

# Ensure numeric data types
df['founding_year'] = pd.to_numeric(df['founding_year'])
df['unicorn_entry_year'] = pd.to_numeric(df['unicorn_entry_year'])

# Split the data into features (X) and target variable (y)
X = df[['founding_year', 'unicorn_entry_year']]
y = df['current_valuation']

# Add a constant term to the features (required for statsmodels)
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model using Ordinary Least Squares (OLS)
model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

# Print regression summary
print(model.summary())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the regression results
fig, ax = plt.subplots(figsize=(12, 6))

# Scatter plot of actual values
ax.scatter(X_test['founding_year'], y_test, color='blue', label='Actual')

# Scatter plot of predicted values
ax.scatter(X_test['founding_year'], y_pred, color='red', label='Predicted')

# Line graph for the regression line
ax.plot(X_test['founding_year'], model.predict(X_test), color='green', label='Regression Line')

ax.set_xlabel('Founding Year / Unicorn Entry Year')
ax.set_ylabel('Current Valuation')
ax.set_title('Regression Analysis: Founding Year and Unicorn Entry Year vs Current Valuation')
ax.legend()
plt.show()


# In[5]:


import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sqlite3
# Load data from SQLite database into a DataFrame
conn = sqlite3.connect("unicorn_data.db")
df = pd.read_sql_query("SELECT * FROM unicorn_data;", conn)
conn.close()

# Handle missing values (if any)
df.dropna(subset=['founding_year', 'unicorn_entry_year', 'current_valuation'], inplace=True)

# Extract numeric part from 'current_valuation' and handle 'Billion'
df['current_valuation'] = df['current_valuation'].str.replace('[\$,]', '', regex=True)
df['current_valuation'] = pd.to_numeric(df['current_valuation'].str.replace('Billion', '').replace('-', '0')) * 1_000_000_000

# Ensure numeric data types
df['founding_year'] = pd.to_numeric(df['founding_year'])
df['unicorn_entry_year'] = pd.to_numeric(df['unicorn_entry_year'])

# Split the data into features (X) and target variable (y)
X = df[['founding_year', 'unicorn_entry_year']]
y = df['current_valuation']

# Add a constant term to the features (required for statsmodels)
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model using Ordinary Least Squares (OLS)
model = sm.OLS(y_train.astype(float), X_train.astype(float)).fit()

# Print regression summary
print(model.summary())

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualize the regression results
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(X_test['founding_year'], y_test, color='blue', label='Actual')
ax.scatter(X_test['founding_year'], y_pred, color='red', label='Predicted')
ax.set_xlabel('Founding Year / Unicorn Entry Year')
ax.set_ylabel('Current Valuation')
ax.set_title('Regression Analysis: Founding Year and Unicorn Entry Year vs Current Valuation')
ax.legend()
plt.show()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Function to convert strings with 'Million' and 'Billion' to numeric values
def convert_to_numeric(value):
    if 'Billion' in value:
        return float(value.replace('[\$,Billion]', '').replace('Billion', '')) * 1_000_000_000
    elif 'Million' in value:
        return float(value.replace('[\$,Million]', '').replace('Million', '')) * 1_000_000
    else:
        return 0.0

# Load the dataset
df = pd.read_csv('C:\\Users\\reshm\\AppData\\Local\\Temp\\unicorn_data.csv')

# Display the first few rows of the DataFrame to inspect the data
print(df.head())

# Data Preprocessing
# Convert 'Profit/Loss FY22' and 'Current Valuation' to numeric values
df['Profit/Loss FY22'] = df['Profit/Loss FY22'].apply(convert_to_numeric)
df['Current Valuation'] = df['Current Valuation'].apply(convert_to_numeric)

# Replace missing values with 0
df['Profit/Loss FY22'].fillna(0, inplace=True)

# Encode the 'Status' column to numerical values
df['Status'] = df['Status'].map({'Private': 0, 'Acquired': 1, 'Listed': 2, 'IPO-Bound': 3})

# Feature Selection
# Selecting 'Profit/Loss FY22' and 'Current Valuation' as features
X = df[['Profit/Loss FY22', 'Current Valuation']]

# Target variable is 'Status'
y = df['Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg_model = LogisticRegression(random_state=42)

# Fit the model on the training data
log_reg_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)
print('\nConfusion Matrix:')
print(conf_matrix)

# Visualize the decision boundary
sns.scatterplot(x='Profit/Loss FY22', y='Current Valuation', hue='Status', data=df, palette='viridis')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Profit/Loss FY22')
plt.ylabel('Current Valuation')
plt.legend(loc='upper right')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Web Crawing code
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the website
url = "https://finance.yahoo.com/trending-tickers"

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the data
    table = soup.find('table')

    # Extract column names
    columns = [header.text.strip() for header in table.find_all('th')]

    # Initialize an empty list to store data
    data = []

    # Extract data rows
    for row in table.find_all('tr')[1:]:
        values = [cell.text.strip() for cell in row.find_all('td')]
        data.append(values)
        # Create a DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Display the DataFrame
    print(df)
# Save the DataFrame to a CSV file
    df.to_csv('trending_data.csv', index=False)

    print("Data has been successfully crawled and saved to CSV.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")





# In[2]:


# Save the DataFrame to a CSV file
df.to_csv('ticket_data.csv', index=False)

print("Data has been successfully crawled, displayed, saved to SQLite database, and saved to CSV file.")


# In[3]:


#Analysis part
import pandas as pd

# Replace 'C:\\Users\\reshm\\ticket_data.csv' with the actual path to your CSV file
file_path = 'C:\\Users\\reshm\\ticket_data.csv'

# Read the CSV file into a DataFrame
stocks_data = pd.read_csv(file_path)

# Function to convert Market Cap values to numeric
def convert_market_cap(value):
    if isinstance(value, (float, int)):
        return value  # Return as-is for numeric values
    elif 'B' in value:
        return float(value.replace('B', '')) * 1e9  # Convert to billion
    elif 'M' in value:
        return float(value.replace('M', '')) * 1e6  # Convert to million
    elif 'T' in value:
        return float(value.replace('T', '')) * 1e12  # Convert to trillion
    else:
        return pd.NA  # For N/A values

# Apply the conversion function to the 'Market Cap' column
stocks_data['Market Cap'] = stocks_data['Market Cap'].apply(convert_market_cap)

# Display the cleaned DataFrame
print(stocks_data)



#descriptive analysis - Scatter
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(stocks_data['Last Price'], stocks_data['Market Cap'])
plt.title('Scatter Plot of Last Price vs. Market Cap')
plt.xlabel('Last Price')
plt.ylabel('Market Cap')
plt.show()


#Descriptive- Histogram
plt.figure(figsize=(10, 6))
plt.hist(stocks_data['Last Price'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Last Price')
plt.xlabel('Last Price')
plt.ylabel('Frequency')
plt.show()

#Descriptive: Bar
plt.figure(figsize=(12, 6))
plt.bar(stocks_data['Symbol'], stocks_data['Change'])
plt.title('Bar Plot of Change in Stock Prices')
plt.xlabel('Symbol')
plt.ylabel('Change')
plt.xticks(rotation=45, ha='right')
plt.show()


#Descriptive :
plt.figure(figsize=(12, 6))
for symbol in stocks_data['Symbol'].unique():
    symbol_data = stocks_data[stocks_data['Symbol'] == symbol]
    plt.plot(symbol_data['Market Time'], symbol_data['Volume'], label=symbol)

plt.title('Volume Trend Over Time for Different Symbols')
plt.xlabel('Market Time')
plt.ylabel('Volume')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)
plt.show()

#Regression Analysis
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Your DataFrame with the provided data
data = {
    'Symbol': ['ORCL', 'AVGO', 'COST', 'AMD', 'BTC-USD', 'INTC', 'BYND', 'SOXL', '^DJI', 'ETH-USD', '^GSPC', '^IXIC', 'AMC', 'FTCH', 'JTAI', 'NVDA', 'SHAK', 'HAS', 'INAB', 'LCID', 'SNAP', 'AMAT', 'SE', 'INVZ', 'OXY', 'LYFT', 'QCOM', 'MU', 'MARA', 'SOXS'],
    'Last Price': [115.13, 1029.24, 623.86, 134.41, 41642.70, 44.54, 9.94, 26.11, 36404.93, 2233.87, 4622.44, 14432.49, 7.11, 0.7304, 3.02, 466.27, 66.83, 48.89, 1.28, 4.61, 15.75, 155.14, 37.87, 2.3, 57.06, 14.36, 136.1, 77.79, 14.65, 7.25],
    'Market Cap': [315.385, 480.818, 276.208, 217.139, 814.839, 187.781, 641.537, 'N/A', 'N/A', 268.546, 'N/A', 'N/A', 1.41, 288.848, 27.227, 1.152, 2.827, 6.784, 40.949, 10.554, 25.931, 129.78, 21.539, 379.247, 50.234, 5.643, 151.479, 85.859, 3.261, 'N/A']
}

df = pd.DataFrame(data)

# Clean 'Market Cap' column by replacing 'N/A' with NaN and converting to numeric
df['Market Cap'] = pd.to_numeric(df['Market Cap'].replace('N/A', pd.NA), errors='coerce')

# Drop rows with missing values
df.dropna(subset=['Last Price', 'Market Cap'], inplace=True)

# Regression analysis
X = sm.add_constant(df['Market Cap'])
y = df['Last Price']

model = sm.OLS(y, X)
results = model.fit()

# Plotting the regression line and data points
plt.scatter(df['Market Cap'], df['Last Price'], label='Actual Data')
plt.plot(df['Market Cap'], results.predict(), color='red', label='Regression Line')
plt.xlabel('Market Cap')
plt.ylabel('Last Price')
plt.title('Regression Analysis: Last Price vs Market Cap')
plt.legend()
plt.show()








# In[6]:


#Regression calculations
import pandas as pd
import statsmodels.api as sm

# Your DataFrame with the provided data
data = {
    'Symbol': ['ORCL', 'AVGO', 'COST', 'AMD', 'BTC-USD', 'INTC', 'BYND', 'SOXL', '^DJI', 'ETH-USD', '^GSPC', '^IXIC', 'AMC', 'FTCH', 'JTAI', 'NVDA', 'SHAK', 'HAS', 'INAB', 'LCID', 'SNAP', 'AMAT', 'SE', 'INVZ', 'OXY', 'LYFT', 'QCOM', 'MU', 'MARA', 'SOXS'],
    'Last Price': [115.13, 1029.24, 623.86, 134.41, 41642.70, 44.54, 9.94, 26.11, 36404.93, 2233.87, 4622.44, 14432.49, 7.11, 0.7304, 3.02, 466.27, 66.83, 48.89, 1.28, 4.61, 15.75, 155.14, 37.87, 2.3, 57.06, 14.36, 136.1, 77.79, 14.65, 7.25],
    'Market Cap': [315.385, 480.818, 276.208, 217.139, 814.839, 187.781, 641.537, 'N/A', 'N/A', 268.546, 'N/A', 'N/A', 1.41, 288.848, 27.227, 1.152, 2.827, 6.784, 40.949, 10.554, 25.931, 129.78, 21.539, 379.247, 50.234, 5.643, 151.479, 85.859, 3.261, 'N/A']
}

df = pd.DataFrame(data)

# Clean 'Market Cap' column by replacing 'N/A' with NaN and converting to numeric
df['Market Cap'] = pd.to_numeric(df['Market Cap'].replace('N/A', pd.NA), errors='coerce')

# Drop rows with missing values
df.dropna(subset=['Last Price', 'Market Cap'], inplace=True)

# Regression analysis
X = sm.add_constant(df['Market Cap'])
y = df['Last Price']

model = sm.OLS(y, X)
results = model.fit()

# Print regression results
print(results.summary())



# In[4]:


#Sentimental Analysis : 
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'stocks_data' is your DataFrame
# Ensure that the 'Market Time' column is in datetime format
stocks_data['Market Time'] = pd.to_datetime(stocks_data['Market Time'])

# Convert 'Volume' to numeric, handling commas
stocks_data['Volume'] = pd.to_numeric(stocks_data['Volume'].str.replace(',', ''), errors='coerce')

# Plotting % Change and Volume over time
plt.figure(figsize=(12, 6))
plt.plot(stocks_data['Market Time'], stocks_data['% Change'], label='% Change', marker='o')
plt.bar(stocks_data['Market Time'], stocks_data['Volume'] / 1e6, alpha=0.5, label='Volume (Millions)', color='orange')
plt.xlabel('Market Time')
plt.ylabel('% Change / Volume (Millions)')
plt.title('Relationship between % Change and Trading Volume Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[10]:


#classification Analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assume 'df' is your DataFrame with the stock data

# Convert 'Market Cap' to binary classes (e.g., 'Small' and 'Large') based on a threshold
threshold = 1e9  # 1 Billion
df['Market Cap Class'] = df['Market Cap'].apply(lambda x: 'Small' if x <= threshold else 'Large')

# Features (X) and target variable (y)
X = df[['Last Price', 'Change', '% Change', 'Volume']]  # Adjust features as needed
y = df['Market Cap Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)


# In[14]:


#Classification Analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assume 'df' is your DataFrame with the stock data

# Convert 'Market Cap' to numeric values, handling 'N/A'
def convert_market_cap(value):
    try:
        # Check if the value is already a float
        if isinstance(value, float):
            return value
        else:
            return float(value.replace('B', '').replace('M', '').replace('T', '').replace('N/A', '').replace(',', ''))
    except ValueError:
        return pd.NA

# Apply the conversion function to 'Market Cap' column
df['Market Cap'] = df['Market Cap'].apply(convert_market_cap)

# Drop rows with missing values in 'Market Cap'
df = df.dropna(subset=['Market Cap'])

# Convert 'Change' to numeric values
df['Change'] = df['Change'].astype(float)

# Convert 'Market Cap' to binary classes (e.g., 'Small' and 'Large') based on a threshold
threshold = 1e9  # 1 Billion
df['Market Cap Class'] = df['Market Cap'].apply(lambda x: 'Small' if x <= threshold else 'Large')

# Features (X) and target variable (y)
X = df[['Last Price', 'Change', 'Volume']]  # Adjust features as needed
y = df['Market Cap Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nClassification Report:')
print(classification_rep)


# In[21]:


#Sentimental analysis
import pandas as pd

# Assuming 'df' is your actual DataFrame

# Convert 'Volume' to numeric values, handling 'M' and 'K'
def convert_volume(value):
    try:
        # Check if the value is already a float
        if isinstance(value, float):
            return value
        elif pd.isna(value):
            return pd.NA
        else:
            # Extract numeric value and multiplier (M, K)
            numeric_value, multiplier = float(value[:-1]), value[-1]

            # Apply multiplier to numeric value
            if multiplier == 'M':
                return numeric_value * 1e6
            elif multiplier == 'K':
                return numeric_value * 1e3
            else:
                return pd.NA
    except ValueError:
        return pd.NA

# Apply the conversion function to 'Volume' column
df['Volume'] = df['Volume'].apply(convert_volume)

# Sentiment analysis based on threshold
threshold = 1e7  # 10 Million
df['Sentiment'] = df['Volume'].apply(lambda x: 'Negative' if pd.isna(x) or x <= threshold else 'Positive')

# Display the result
print("DataFrame with Sentiment Analysis:")
print(df[['Symbol', 'Volume', 'Sentiment']])


# In[20]:


import pandas as pd

# Assuming 'df' is your actual DataFrame

# Convert 'Market Cap' to numeric values, handling 'N/A'
def convert_market_cap(value):
    try:
        # Check if the value is already a float
        if isinstance(value, float):
            return value
        elif pd.isna(value):
            return pd.NA
        else:
            # Extract numeric value and multiplier (B, M, T)
            numeric_value, multiplier = float(value[:-1]), value[-1]

            # Apply multiplier to numeric value
            if multiplier == 'B':
                return numeric_value * 1e9
            elif multiplier == 'M':
                return numeric_value * 1e6
            elif multiplier == 'T':
                return numeric_value * 1e12
            else:
                return pd.NA
    except ValueError:
        return pd.NA

# Apply the conversion function to 'Market Cap' column
df['Market Cap'] = df['Market Cap'].apply(convert_market_cap)

# Sentiment analysis based on threshold
threshold = 100  # 1 Billion
df['Sentiment'] = df['Market Cap'].apply(lambda x: 'Negative' if pd.isna(x) or x <= threshold else 'Positive')

# Display the result
print("DataFrame with Sentiment Analysis:")
print(df[['Symbol', 'Market Cap', 'Sentiment']])


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].str.replace('M', ''), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Visualize 'Volume' vs 'Last Price' using a scatter plot
plt.scatter(df['Volume'], df['Last Price'], color='blue')
plt.xlabel('Volume')
plt.ylabel('Last Price')
plt.title('Volume vs Last Price Scatter Plot')
plt.show()


# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].str.replace('M', ''), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[['Volume']], df['Last Price'], test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)

# Visualize actual Last Price values
plt.scatter(X_test, y_test, color='blue', label='Actual Last Price')

plt.xlabel('Volume')
plt.ylabel('Last Price')
plt.title('Actual Last Price vs. Volume')
plt.legend()
plt.show()


# In[27]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Create a pipeline with Polynomial Regression
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Make predictions on the test set
y_poly_pred = poly_model.predict(X_test)

# Visualize actual Last Price values and Polynomial Regression predictions
plt.scatter(X_test, y_test, color='blue', label='Actual Last Price')
plt.plot(X_test, y_poly_pred, color='red', label='Polynomial Regression')
plt.xlabel('Volume')
plt.ylabel('Last Price')
plt.title('Actual Last Price vs. Volume with Polynomial Regression')
plt.legend()
plt.show()


# In[43]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].str.replace('M', ''), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[['Volume']], df['Last Price'], test_size=0.2, random_state=42)

# Polynomial regression
poly_degree = 2  # Set the degree of the polynomial
poly_features = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred_poly = poly_model.predict(X_test_poly)

# Sort the values of X_test before plotting
sort_axis = np.argsort(X_test.values.flatten())
sorted_X_test = X_test.values.flatten()[sort_axis]

# Plot the original data
plt.scatter(df['Volume'], df['Last Price'], color='blue', label='Actual Last Price')

# Plot the polynomial regression line
plt.plot(sorted_X_test, y_pred_poly[sort_axis], color='red', linewidth=2, label='Polynomial Regression')

# NumPy example
x = np.linspace(0, 250, 10)
y = np.poly1d(np.polyfit([1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22], [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100], 3))(x)
plt.plot(x, y, label='NumPy Polynomial Regression', linestyle='dashed', color='green')

plt.xlabel('Volume')
plt.ylabel('Last Price')
plt.title('Volume vs Last Price with Polynomial Regression')
plt.legend()
plt.show()


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
    '% Change': [-7.44, 11.82, 0.40, 54.26, -21.19, 31.88, -2.44, 99.53, 1.64, 5.61, -7.71, 98.85, -3.78, -13.94, -12.45, -21.51, 2.08, -31.90, 32.98, 3.94, -0.97, -0.57, -1.27, 53.40, 9.25, 1.22, -0.58, 6.70, 11.48, 17.76],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].replace({'M': ''}, regex=True), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Choose independent and dependent variables
X = df[['% Change', 'Volume']]
y = df['Last Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)

# Visualize predictions vs. actual values
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Last Price')
plt.ylabel('Predicted Last Price')
plt.title('Actual vs. Predicted Last Price for Linear Regression')
plt.show()


# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Change': [-2.12, 42.3, 0.95, 0.1362, -0.1452, 6.75, -2.09, 2.1, 3.18, 2.04, -0.53, 0.0343, -1.14, -3.2, -9.39, -0.2, 0.03, -1.1899, 3.72, 18.25, -0.69, -0.09, -8.07, 0.55, 4.52, 0.5, -0.42, 8.78, 1.71, 7.88],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].str.replace('M', ''), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[['Change', 'Volume']], df['Last Price'], test_size=0.2, random_state=42)

# Initialize and fit the Multiple Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg_model.predict(X_test)

# Visualize predictions vs. actual values
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Last Price')
plt.ylabel('Predicted Last Price')
plt.title('Actual vs. Predicted Last Price for Multiple Linear Regression')
plt.show()


# In[40]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Sample dataset
data = {
    'Symbol': ['PFE', 'VRTX', 'TSLA', 'RSLS', 'CGC', 'ACAD', 'ETSY', 'STTK', 'AAPL', 'X', 'UEC', 'AVTX', 'LUV', 'RILY', 'MED', 'WEED.TO', 'FSR', 'GTHX', 'CHSN', 'NFLX', 'ZM', 'IEP', 'ADBE', 'ADCT', 'CALM', 'OILK', 'BABA', 'ELF', 'MARA', 'ABM'],
    'Last Price': [26.45, 400.03, 237.96, 0.3872, 0.5399, 27.92, 83.73, 4.21, 197.9, 38.43, 6.28, 0.069, 29.17, 19.78, 66.02, 0.73, 1.47, 2.5401, 15, 481.25, 70.59, 15.73, 625.59, 1.58, 53.41, 41.49, 70.97, 139.85, 16.6, 52.24],
    'Volume': ['124.124M', '4.744M', '113.366M', '195.46M', '80.503M', '12.393M', '10.635M', '53.636M', '41.875M', '23.186M', '21.505M', '244.384M', '12.708M', '3.784M', '887,241', '16.914M', '17.676M', '6.559M', '2.655M', '4.078M', '2.851M', '1.42M', '3.123M', '6.905M', '1.343M', '143,959', '16.149M', '1.418M', '45.24M', '1.837M'],
}

df = pd.DataFrame(data)

# Convert 'Volume' to numeric
df['Volume'] = pd.to_numeric(df['Volume'].str.replace('M', ''), errors='coerce')

# Impute missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df['Volume'] = imputer.fit_transform(df[['Volume']])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[['Volume']], df['Last Price'], test_size=0.2, random_state=42)

# Polynomial regression
poly_degree = 2  # Set the degree of the polynomial
poly_features = PolynomialFeatures(degree=poly_degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions on the test set
y_pred_poly = poly_model.predict(X_test_poly)

# Sort the values of X_test before plotting
sort_axis = np.argsort(X_test.values.flatten())
sorted_X_test = X_test.values.flatten()[sort_axis]

# Plot the original data
plt.scatter(df['Volume'], df['Last Price'], color='blue', label='Actual Last Price')

# Plot the polynomial regression line
plt.plot(sorted_X_test, y_pred_poly[sort_axis], color='red', linewidth=2, label='Polynomial Regression')

# NumPy example
x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
mymodel = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(1, 22, 100)
plt.plot(myline, mymodel(myline), label='NumPy Polynomial Regression', linestyle='dashed', color='green')

plt.xlabel('Volume')
plt.ylabel('Last Price')
plt.title('Volume vs Last Price with Polynomial Regression')
plt.legend()
plt.show()


# In[ ]:





import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import numpy as np
from sklearn.decomposition import TruncatedSVD
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.model_selection import train_test_split
from pmdarima.arima.utils import ndiffs, nsdiffs
import pmdarima as pm
from datetime import timedelta

def calculate_average_weekly_sales():
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    # Extract week number from the 'Date' column
    merged_data['Week_Number'] = merged_data['Date'].dt.isocalendar().week
    
    # Compute the total weekly sales for each week
    weekly_sales_total = merged_data.groupby('Week_Number')['Weekly_Sales'].sum()
    
    # Compute the total number of departments for each week
    department_count = merged_data.groupby('Week_Number')['Dept'].nunique()
    
    # Calculate the average weekly sales per department
    average_weekly_sales = weekly_sales_total / department_count
    
    # Plotting
    plt.figure(figsize=(12, 6))
    average_weekly_sales.plot(kind='bar', color='skyblue')
    plt.title('Average Weekly Sales by Week')
    plt.xlabel('Week Number')
    plt.ylabel('Average Weekly Sales')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def show_fourier_transform(selected_store):
    store_data = merged_data[merged_data['Store'] == selected_store].copy()
    
    # Perform Fourier transformation on weekly sales
    sales_values = store_data['Weekly_Sales'].values
    fourier_transform = np.fft.fft(sales_values)
    freq = np.fft.fftfreq(len(sales_values))
    
    peak_index = np.argmax(np.abs(fourier_transform))

    dominant_frequency = freq[peak_index]
    dominant_amplitude = np.abs(fourier_transform[peak_index])
    
    # Plot the Fourier transformation
    plt.figure(figsize=(10, 6))
    plt.plot(freq, np.abs(fourier_transform))
    plt.title(f'Fourier Transformation of Weekly Sales for Store {selected_store}')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f'fourier_transform_{store_number}.png')
        
        
def show_weekly_sales_by_year(selected_store):
    selected_store_data = merged_data[merged_data['Store'] == selected_store]
    
    # Group by date and sum the weekly sales across all departments
    weekly_sales_by_date = selected_store_data.groupby('Date')['Weekly_Sales'].sum()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_sales_by_date.index, weekly_sales_by_date.values, linestyle='-')
    plt.title(f'Weekly Sales for Store {selected_store}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def svd_decomposition():

    sales_matrix = pd.pivot_table(selected_store_data, values='Weekly_Sales', index='Date', columns='Dept', aggfunc='sum', fill_value=0)


    svd = TruncatedSVD(n_components=min(sales_matrix.shape)-1)
    svd.fit(sales_matrix)

    singular_values = svd.singular_values_
    components = svd.components_


    dominant_dept_index = np.argmax(singular_values)
    dominant_dept = sales_matrix.columns[dominant_dept_index]

    print(f"The most dominant department contributing to weekly sales is Department {dominant_dept}.")
    print("Explained Variance Ratio:", svd.explained_variance_ratio_)

def total_sale():
    total_sales_by_dept = merged_data.groupby('Dept')['Weekly_Sales'].sum()

    sorted_departments = total_sales_by_dept.sort_values(ascending=False)
    
    print("Total Sales by Department (Descending Order):")
    for dept, total_sales in sorted_departments.items():
        print(f"Department {dept}: {total_sales:,.2f}")

def growth_rate():
    sales_matrix = pd.pivot_table(selected_store_data, values='Weekly_Sales', index='Date', columns='Dept', aggfunc='sum', fill_value=0)


    growth_rate = sales_matrix.pct_change().mean()


    most_growth_dept = growth_rate.idxmax()

    print(f"The department with the highest average growth rate is Department {most_growth_dept}.")
    
def correlation_between_stores():

    store_sales_data = merged_data[['Store', 'Weekly_Sales', 'Date']].copy()
    
    pivot_data = store_sales_data.pivot_table(values='Weekly_Sales', index='Date', columns='Store')
    
    non_constant_columns = pivot_data.columns[pivot_data.nunique() > 1]
    pivot_data = pivot_data[non_constant_columns]
    
    # Check for NaN values and drop them
    pivot_data = pivot_data.dropna()
    
    # Calculate the correlation matrix between stores
    correlation_matrix = pivot_data.corr()
    
    # Plot the correlation matrix using Seaborn heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Heatmap between Stores based on Weekly Sales')
    plt.savefig(f'correlation_heatmap.png')   
    plt.show()
    
    
def correlation_matrix():
    summed_weekly_sales = merged_data.groupby('Store')['Weekly_Sales'].sum()
    merged_store_data = merged_data.groupby('Store').first()
    merged_store_data['Weekly_Sales'] = summed_weekly_sales
    

    correlation_matrix = merged_store_data[['Weekly_Sales', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()
    

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()
    


    store_markdowns = merged_data.groupby('Store')[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].sum()
    store_data = pd.concat([summed_weekly_sales, store_markdowns], axis=1)
    

    correlation_matrix = store_data.corr()
        
    correlation_values = correlation_matrix.loc['Weekly_Sales', 'MarkDown1':'MarkDown5'].to_frame().T
    
    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_values, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Correlation between Weekly Sales and Markdowns')
    plt.xlabel('Markdowns')
    plt.ylabel('Weekly Sales')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()

        
def most_profitable_quarter():

    # Group the data by quarter and sum the weekly sales
    merged_data['Quarter'] = merged_data['Date'].dt.to_period('Q')
    quarterly_sales = merged_data.groupby('Quarter')['Weekly_Sales'].sum()
    
    # Find the quarter with the highest total sales
    most_profitable_quarter = quarterly_sales.idxmax()
    total_sales_most_profitable_quarter = quarterly_sales.max()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    quarterly_sales.plot(kind='bar', color='skyblue')
    plt.title('Quarterly Sales')
    plt.xlabel('Quarter')
    plt.ylabel('Total Sales')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def most_profitable_weeks():
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    # Extract week number from the 'Date' column using isocalendar method
    merged_data['Week_Number'] = merged_data['Date'].dt.isocalendar().week
    
    # Group data by week number and calculate total profit (sum of weekly sales) for each week
    weekly_profit = merged_data.groupby('Week_Number')['Weekly_Sales'].sum().reset_index()
    
    # Sort weeks by total profit in descending order to find the weeks with highest profit
    most_profitable_weeks = weekly_profit.sort_values(by='Weekly_Sales', ascending=False)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(weekly_profit['Week_Number'], weekly_profit['Weekly_Sales'], color='blue', label='Total Weekly Sales')
    plt.bar(most_profitable_weeks['Week_Number'][:5], most_profitable_weeks['Weekly_Sales'][:5], color='red', label='Most Profitable Weeks')
    plt.xlabel('Week Number')
    plt.ylabel('Total Weekly Sales')
    plt.title('Total Weekly Sales')
    plt.legend()
    plt.grid(axis='y')
    plt.xticks(range(1, 53))
    plt.show()
    
def sales_forecast(selected_store):
# Select a store for SARIMA modeling
  
    store_data = merged_data[merged_data['Store'] == selected_store].copy()
    


    # Prepare data for SARIMA model
    date_sales = store_data[['Date', 'Weekly_Sales']].set_index('Date')
    date_sales.index = pd.to_datetime(date_sales.index)
    date_sales = date_sales.resample('W').sum()  
    
    # Train-test split
    train_size = int(len(date_sales) * 0.8)
    train, test = date_sales.iloc[:train_size], date_sales.iloc[train_size:]
    
    # Determine differencing orders
    d = ndiffs(train['Weekly_Sales'], test='adf')
    D = nsdiffs(train['Weekly_Sales'], m=12, test='ocsb')
    
    # Perform grid search for optimal parameters
    model = pm.auto_arima(train['Weekly_Sales'],
                          seasonal=True, m=12,
                          d=d, D=D,
                          suppress_warnings=True,
                          stepwise=True,
                          error_action='ignore',
                          trace=True)
    
    # Summary of the best model
    print(model.summary())
    
    # Make predictions on the test set
    forecast, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    forecast_index = pd.date_range(start=train.index[-1] + timedelta(days=1), periods=len(test), freq='W')
    
    # Plot the actual vs. predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train['Weekly_Sales'], label='Train')
    plt.plot(test.index, test['Weekly_Sales'], label='Test')
    plt.plot(forecast_index, forecast, label='SARIMA Forecast', color='red')
    plt.fill_between(forecast_index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.title(f'SARIMA Model Forecast for Store {selected_store}')
    plt.xlabel('Date')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.show()
    plt.savefig(f'sarima_model_{store_number}.png')
    

def seasonal_decomposition(store_number):
    selected_store_data = merged_data[merged_data['Store'] == store_number]
    
    # Aggregate weekly sales across departments for the selected store
    weekly_sales = selected_store_data.groupby('Date')['Weekly_Sales'].sum()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(weekly_sales, model='multiplicative', period=52)  # Assuming weekly data
    
    # Plot the decomposition
    plt.figure(figsize=(12, 8))
    
    plt.subplot(411)
    plt.plot(weekly_sales, label='Original')
    plt.legend()
    
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend()
    
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend()
    
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend()
    
    
    plt.tight_layout()
    plt.show()

def store_sales_graphs():
    max_sales_store = merged_data.groupby('Store')['Weekly_Sales'].sum().idxmax()
    
    # Find the store with minimum sales
    min_sales_store = merged_data.groupby('Store')['Weekly_Sales'].sum().idxmin()
    
    # Find the store with median sales
    median_sales_store = merged_data.groupby('Store')['Weekly_Sales'].sum().median()
    
    # Find the store closest to the median sales
    median_sales_store = merged_data.groupby('Store')['Weekly_Sales'].sum().sub(median_sales_store).abs().idxmin()
    
    
        
    # Filter data for the selected stores
    selected_stores = merged_data[merged_data['Store'].isin([max_sales_store, min_sales_store, median_sales_store])]
    
    total_sales_by_store = selected_stores.groupby(['Store', 'Date'])['Weekly_Sales'].sum().reset_index()

    # Plot the total weekly sales for each selected store
    plt.figure(figsize=(12, 8))
    for store, data in total_sales_by_store.groupby('Store'):
        plt.plot(data['Date'], data['Weekly_Sales'], label=f'Store {store}')
    
    plt.title('Total Weekly Sales for stores with max, min and median sales')
    plt.xlabel('Date')
    plt.ylabel('Total Weekly Sales')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    store_sales = merged_data.groupby('Store')['Weekly_Sales'].sum()
    store_sizes = merged_data.groupby('Store')['Size'].first()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(store_sizes, store_sales, alpha=0.5)
    plt.title('Effect of Store Size on Total Weekly Sales')
    plt.xlabel('Store Size')
    plt.ylabel('Total Weekly Sales')
    plt.grid(True)
    plt.show()
      
def impactful_departments():
    
    store_dept_sales = merged_data.groupby(['Store', 'Dept'])['Weekly_Sales'].sum().reset_index()
    
    # Finding the department with the highest sales for each store
    most_impactful_dept = store_dept_sales.loc[store_dept_sales.groupby('Store')['Weekly_Sales'].idxmax()]
    
    # Plotting the most impactful departments for each store
    plt.figure(figsize=(10, 6))
    plt.bar(most_impactful_dept['Store'], most_impactful_dept['Weekly_Sales'])
    plt.title('Most Impactful Departments for Each Store')
    plt.xlabel('Store')
    plt.ylabel('Total Weekly Sales')
    plt.xticks(most_impactful_dept['Store'])
    for i, dept in enumerate(most_impactful_dept['Dept']):
        plt.text(i + 1, most_impactful_dept['Weekly_Sales'].iloc[i], str(dept), ha='center', va='bottom')
    plt.grid(axis='y')
    plt.show()


plt.tight_layout()
plt.show()
sales_data = pd.read_csv('sales_data_set.csv')
stores_data = pd.read_csv('stores_data_set.csv')
features_data = pd.read_csv('features_data_set.csv')

# Merge the DataFrames based on common columns ('Store' and 'Date')
merged_data = pd.merge(sales_data, stores_data, on='Store', how='left')
merged_data = pd.merge(merged_data, features_data, on=['Store', 'Date'], how='left')


merged_data['Date'] = pd.to_datetime(merged_data['Date'], dayfirst=True)



merged_data['Unemployment'].interpolate(method='linear', inplace=True)
merged_data['CPI'].interpolate(method='linear', inplace=True)
merged_data = merged_data.drop(columns='IsHoliday_x', errors='ignore')

store_number = 42  # Change this to the desired store number
max_store_number = 44  # Maximum allowed store number

# Check if the store number is within the allowed range
if 1 <= store_number <= max_store_number:

    selected_store_data = merged_data[merged_data['Store'] == store_number].copy()

  
    selected_store_data['Date'] = pd.to_datetime(selected_store_data['Date'])

   
    selected_store_data['Year'] = selected_store_data['Date'].dt.year

    #show_weekly_sales_by_year(store_number)
    #calculate_average_weekly_sales()
    #show_fourier_transform(store_number)
    #svd_decomposition()
    #growth_rate()
    #correlation_matrix()
    #total_sale()
    #seasonal_decomposition(store_number)
    #sales_forecast(store_number)
    #correlation_between_stores()
    #most_profitable_quarter()
    #most_profitable_weeks()
    #impactful_departments()
    #store_sales_graphs()

    
    
    
else:
    print(f"Invalid store number. Please select a store number between 1 and {max_store_number}.")
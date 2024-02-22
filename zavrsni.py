import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import numpy as np
from sklearn.decomposition import TruncatedSVD
from statsmodels.tsa.seasonal import seasonal_decompose

def calculate_average_weekly_sales():
    selected_store_data['Week_of_Year'] = selected_store_data['Date'].dt.isocalendar().week
    average_weekly_sales = selected_store_data.groupby('Week_of_Year')['Weekly_Sales'].mean()
    print(f'Average sales by weeks for store {store_number} {average_weekly_sales} ')
    

def show_fourier_transform():
    # Filter data for the specific store
    store_data = selected_store_data.copy()

    # Assuming 'Date' is the datetime column
    store_data['Date'] = pd.to_datetime(store_data['Date'])
    
    # Assuming 'Weekly_Sales' is the column you want to analyze
    weekly_sales = store_data.groupby(store_data['Date'].dt.to_period("W"))["Weekly_Sales"].sum()

    N = len(weekly_sales)
    T = 1.0 / 52  
    yf = fft(weekly_sales.values)
    xf = np.fft.fftfreq(N, T)[:N]

    # Find the index of the peak frequency with amplitude above a threshold
    threshold=0.1
    peak_indices = np.where(np.abs(yf) > threshold)[0]
    if len(peak_indices) == 0:
        print(f"No dominant frequency found for Store {store_number}.")
    else:
        dominant_frequency = 1.0 / xf[peak_indices[np.argmax(np.abs(yf[peak_indices]))]]

        # Plot the Fourier Transform
        plt.figure(figsize=(12, 6))
        plt.plot(1.0 / xf, 2.0 / N * np.abs(yf), marker='o')
        plt.title(f'Fourier Transform for Weekly Sales (Store {store_number})')
        plt.xlabel('Period (Weeks)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        # Highlight the dominant frequency
        plt.axvline(x=dominant_frequency, color='r', linestyle='--', label=f'Dominant Frequency: {dominant_frequency:.2f} Weeks')
        plt.legend()

        plt.show()

        # Print the dominant frequency
        print(f'Dominant Frequency for Store {store_number}: {dominant_frequency:.2f} Weeks')
        
        
        
def show_weekly_sales_by_year():
    weekly_sales_by_year = selected_store_data.groupby(['Year', selected_store_data['Date'].dt.to_period("W")])['Weekly_Sales'].sum().reset_index()


    plt.figure(figsize=(14, 6))
    sns.set_palette("viridis", len(weekly_sales_by_year['Year'].unique())) 
    
    for year, data in weekly_sales_by_year.groupby('Year'):
        sns.lineplot(x=data['Date'].dt.weekofyear, y=data['Weekly_Sales'], label=f'Year {year}', marker='o')  # Use week of the year as x-axis

    plt.title(f'Weekly Sales for Store {store_number} by Year')
    plt.xlabel('Week')
    plt.ylabel('Weekly Sales')
    plt.xticks(range(1, 53))  
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left') 

    plt.grid(True)
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

def correlation_matrix():
    selected_columns = ['Weekly_Sales', 'Fuel_Price', 'CPI', 'IsHoliday_y', 'Unemployment', 'Temperature']

    selected_data = selected_store_data[selected_columns]

    correlation_matrix = selected_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()
    print("Conclusions:")
    print("-" * 40)
    correlation_sales_holiday = correlation_matrix.loc['Weekly_Sales', 'IsHoliday_y']
    correlation_sales_temperature = correlation_matrix.loc['Weekly_Sales', 'Temperature']

    for column in correlation_matrix.columns:
        for index in correlation_matrix.index:
            if column != index:
                correlation_value = correlation_matrix.loc[index, column]
                print(f"Correlation between {index} and {column}: {correlation_value:.2f}")
        
                if correlation_value > 0:
                    print("Conclusion: There is a positive correlation.")
                elif correlation_value < 0:
                    print("Conclusion: There is a negative correlation.")
                else:
                    print("Conclusion: There is no significant linear relationship.")
                
                print("-" * 50)


    total_sales_by_store = merged_data.groupby('Store')['Weekly_Sales'].sum()
    store_sizes = merged_data.groupby('Store')['Size'].max()

    store_sales_size_df = pd.DataFrame({'Total_Sales': total_sales_by_store, 'Store_Size': store_sizes})
    correlation_sales_size = store_sales_size_df['Total_Sales'].corr(store_sales_size_df['Store_Size'])

    print(f"Correlation between Total Sales and Store Size: {correlation_sales_size:.2f}")
    print("-" * 50)
    
    markdown_columns = ["Weekly_Sales","MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    selected_data_markdown = selected_store_data[markdown_columns].copy()

    selected_data_markdown.replace('NA', pd.NA, inplace=True)
  
    selected_data_markdown.fillna(0, inplace=True)
    

    correlation_sales_markdown1 = selected_data_markdown['Weekly_Sales'].corr(selected_data_markdown['MarkDown1'])
    correlation_sales_markdown2 = selected_data_markdown['Weekly_Sales'].corr(selected_data_markdown['MarkDown2'])
    correlation_sales_markdown3 = selected_data_markdown['Weekly_Sales'].corr(selected_data_markdown['MarkDown3'])
    correlation_sales_markdown4 = selected_data_markdown['Weekly_Sales'].corr(selected_data_markdown['MarkDown4'])
    correlation_sales_markdown5 = selected_data_markdown['Weekly_Sales'].corr(selected_data_markdown['MarkDown5'])
    

    print(f"Correlation with MarkDown1: {correlation_sales_markdown1:.2f}")
    print(f"Correlation with MarkDown2: {correlation_sales_markdown2:.2f}")
    print(f"Correlation with MarkDown3: {correlation_sales_markdown3:.2f}")
    print(f"Correlation with MarkDown4: {correlation_sales_markdown4:.2f}")
    print(f"Correlation with MarkDown5: {correlation_sales_markdown5:.2f}")
    print("-" * 50)

def seasonal_decomposition():
    selected_store_data_copy = selected_store_data.copy()
    
    time_series = selected_store_data_copy['Weekly_Sales']

    # Try different period values and observe the results
    periods_to_try = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 100]
    
    best_period = None
    best_quality_measure = float('inf')  # Initialize with a high value
    
    for period in periods_to_try:
        # Perform seasonal decomposition
        result = seasonal_decompose(time_series, model='additive', period=period)
        
        # Calculate a quality measure (e.g., standard deviation of the seasonal component)
        quality_measure = np.std(result.seasonal)
        
        # Update the best period if the current one is better
        if quality_measure < best_quality_measure:
            best_quality_measure = quality_measure
            best_period = period
    
    # Print the best period and quality measure
    print(f'Best Period: {best_period}')
    print(f'Corresponding Quality Measure (e.g., std of seasonal component): {best_quality_measure}')
    
    # Plot the components for the best period
    result = seasonal_decompose(time_series, model='additive', period=best_period)
    plt.figure(figsize=(12, 8))
    
    # Original time series
    plt.subplot(4, 1, 1)
    plt.plot(time_series)
    plt.title('Original Time Series')
    
    # Trend component
    plt.subplot(4, 1, 2)
    plt.plot(result.trend)
    plt.title('Trend Component')
    
    # Seasonal component
    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal)
    plt.title('Seasonal Component')
    
    # Residual component
    plt.subplot(4, 1, 4)
    plt.plot(result.resid)
    plt.title('Residual Component')
    
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

store_number = 10  # Change this to the desired store number
max_store_number = 44  # Maximum allowed store number

# Check if the store number is within the allowed range
if 1 <= store_number <= max_store_number:

    selected_store_data = merged_data[merged_data['Store'] == store_number].copy()

  
    selected_store_data['Date'] = pd.to_datetime(selected_store_data['Date'])

   
    selected_store_data['Year'] = selected_store_data['Date'].dt.year

    show_weekly_sales_by_year()
    calculate_average_weekly_sales()
    show_fourier_transform()
    svd_decomposition()
    growth_rate()
    correlation_matrix()
    total_sale()
    seasonal_decomposition()
    
    
else:
    print(f"Invalid store number. Please select a store number between 1 and {max_store_number}.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rawdata = pd.read_csv('/content/global_house_purchase_dataset.csv')
rawdata.sample(n=5)

rawdata.info()

rawdata.describe(include='all')

# Create a pivot table to see the count of property types based on furnishing status
pivot_table = pd.pivot_table(rawdata,
                              values='property_id', # You can use any non-null column for counting
                              index=['property_type'],
                              columns=['furnishing_status'],
                              aggfunc='count',
                              fill_value=0) # Fill missing values with 0

print("Pivot Table: Count of Property Types by Furnishing Status")
display(pivot_table)

# convert year data to
rawdata['constructed_year'] = pd.to_datetime(rawdata['constructed_year'], format='%Y', errors='coerce').dt.year

rawdata1 = rawdata.copy()
rawdata1.sample(n=5)

# convert year data to
rawdata['constructed_year'] = pd.to_datetime(rawdata['constructed_year'], format='%Y', errors='coerce').dt.year

rawdata1 = rawdata.copy()
rawdata1.sample(n=5)

rawdata1['property_type'].unique().tolist()

# Select only the specified numerical columns for correlation analysis
columns_for_correlation = [
    'property_size_sqft', 'price', 'rooms', 'bathrooms', 'garage', 'garden', 'crime_cases_reported',
    'legal_cases_on_property', 'customer_salary', 'loan_amount', 'loan_tenure_years', 'monthly_expenses', 'down_payment',
    'emi_to_income_ratio', 'satisfaction_score', 'neighbourhood_rating', 'connectivity_score'
    ]
numerical_cols_subset = rawdata1[columns_for_correlation]

# Calculate Pearson correlation matrix
pearson_corr = numerical_cols_subset.corr(method='pearson')

# Calculate Spearman correlation matrix
spearman_corr = numerical_cols_subset.corr(method='spearman')

# Create heatmaps
plt.figure(figsize=(12, 5))

# Pearson Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap (Selected Columns)')

# Spearman Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Heatmap (Selected Columns)')

plt.tight_layout()
plt.show()

# Select only the specified numerical columns for correlation analysis
columns_for_correlation = [
    'property_size_sqft', 'price', 'customer_salary', 'loan_amount', 'monthly_expenses', 'down_payment',
    'emi_to_income_ratio']
numerical_cols_subset = rawdata1[columns_for_correlation]

# Calculate Pearson correlation matrix
pearson_corr = numerical_cols_subset.corr(method='pearson')

# Calculate Spearman correlation matrix
spearman_corr = numerical_cols_subset.corr(method='spearman')

# Create heatmaps
plt.figure(figsize=(12, 5))

# Pearson Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Heatmap (Selected Columns)')

# Spearman Heatmap
plt.subplot(1, 2, 2)
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Spearman Correlation Heatmap (Selected Columns)')

plt.tight_layout()
plt.show()

# Create scatter plots for each pair of selected variables
columns_to_plot = [
    'property_size_sqft', 'price', 'customer_salary', 'loan_amount', 'monthly_expenses', 'down_payment',
    'emi_to_income_ratio']

plt.figure(figsize=(20, 15))

# Iterate through all pairs of columns and create scatter plots
num_cols = len(columns_to_plot)
for i in range(num_cols):
    for j in range(i + 1, num_cols):
        plt.subplot(num_cols, num_cols, i * num_cols + j + 1)
        sns.scatterplot(data=rawdata1, x=columns_to_plot[j], y=columns_to_plot[i])
        plt.title(f'{columns_to_plot[i]} vs {columns_to_plot[j]}')

plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency

data = pd.crosstab(rawdata1['property_type'], rawdata1['furnishing_status'])

stat, p, dof, expected = chi2_contingency(data)

alpha = 0.05

print("Contingency Table:")
display(data)

print("\np value is " + str(p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (H0 holds true)')
    end

farmhouse_data = rawdata1[rawdata1['property_type'] == 'Farmhouse']
Apartment_data = rawdata1[rawdata1['property_type'] == 'Apartment']
Townhouse_data = rawdata1[rawdata1['property_type'] == 'Townhouse']
Villa_data = rawdata1[rawdata1['property_type'] == 'Villa']
Studio_data = rawdata1[rawdata1['property_type'] == 'Studio']
Independent_data = rawdata1[rawdata1['property_type'] == 'Independent House']

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create a copy to avoid SettingWithCopyWarning
farmhouse_data_copy = farmhouse_data.copy()

data = farmhouse_data_copy[['price', 'loan_amount']]

x = farmhouse_data_copy['price'].values.reshape(-1, 1)
y = farmhouse_data_copy['loan_amount']

model = LinearRegression().fit(x, y)

# Print the coefficient and intercept
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Calculate and print R-squared
r_squared = model.score(x, y)
print(f"R-squared: {r_squared}")

farmhouse_data_copy['predict'] = model.predict(x)

plt.scatter(farmhouse_data_copy['price'], farmhouse_data_copy['loan_amount'])
plt.plot(farmhouse_data_copy['price'], farmhouse_data_copy['predict'], color='red')
plt.show()

# Linear Regression for Apartment
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

Apartment_data_copy = Apartment_data.copy()

x_apartment = Apartment_data_copy['price'].values.reshape(-1, 1)
y_apartment = Apartment_data_copy['loan_amount']

model_apartment = LinearRegression().fit(x_apartment, y_apartment)

print("Apartment Property Type:")
print(f"Coefficient: {model_apartment.coef_[0]}")
print(f"Intercept: {model_apartment.intercept_}")

r_squared_apartment = model_apartment.score(x_apartment, y_apartment)
print(f"R-squared: {r_squared_apartment}")

Apartment_data_copy['predict'] = model_apartment.predict(x_apartment)

plt.scatter(Apartment_data_copy['price'], Apartment_data_copy['loan_amount'])
plt.plot(Apartment_data_copy['price'], Apartment_data_copy['predict'], color='red')
plt.title('Apartment: Loan Amount vs Price')
plt.xlabel('Price')
plt.ylabel('Loan Amount')
plt.show()

# Linear Regression for Townhouse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

Townhouse_data_copy = Townhouse_data.copy()

x_townhouse = Townhouse_data_copy['price'].values.reshape(-1, 1)
y_townhouse = Townhouse_data_copy['loan_amount']

model_townhouse = LinearRegression().fit(x_townhouse, y_townhouse)

print("Townhouse Property Type:")
print(f"Coefficient: {model_townhouse.coef_[0]}")
print(f"Intercept: {model_townhouse.intercept_}")

r_squared_townhouse = model_townhouse.score(x_townhouse, y_townhouse)
print(f"R-squared: {r_squared_townhouse}")

Townhouse_data_copy['predict'] = model_townhouse.predict(x_townhouse)

plt.scatter(Townhouse_data_copy['price'], Townhouse_data_copy['loan_amount'])
plt.plot(Townhouse_data_copy['price'], Townhouse_data_copy['predict'], color='red')
plt.title('Townhouse: Loan Amount vs Price')
plt.xlabel('Price')
plt.ylabel('Loan Amount')
plt.show()

# Linear Regression for Villa
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

Villa_data_copy = Villa_data.copy()

x_villa = Villa_data_copy['price'].values.reshape(-1, 1)
y_villa = Villa_data_copy['loan_amount']

model_villa = LinearRegression().fit(x_villa, y_villa)

print("Villa Property Type:")
print(f"Coefficient: {model_villa.coef_[0]}")
print(f"Intercept: {model_villa.intercept_}")

r_squared_villa = model_villa.score(x_villa, y_villa)
print(f"R-squared: {r_squared_villa}")

Villa_data_copy['predict'] = model_villa.predict(x_villa)

plt.scatter(Villa_data_copy['price'], Villa_data_copy['loan_amount'])
plt.plot(Villa_data_copy['price'], Villa_data_copy['predict'], color='red')
plt.title('Villa: Loan Amount vs Price')
plt.xlabel('Price')
plt.ylabel('Loan Amount')
plt.show()

# Linear Regression for Studio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

Studio_data_copy = Studio_data.copy()

x_studio = Studio_data_copy['price'].values.reshape(-1, 1)
y_studio = Studio_data_copy['loan_amount']

model_studio = LinearRegression().fit(x_studio, y_studio)

print("Studio Property Type:")
print(f"Coefficient: {model_studio.coef_[0]}")
print(f"Intercept: {model_studio.intercept_}")

r_squared_studio = model_studio.score(x_studio, y_studio)
print(f"R-squared: {r_squared_studio}")

Studio_data_copy['predict'] = model_studio.predict(x_studio)

plt.scatter(Studio_data_copy['price'], Studio_data_copy['loan_amount'])
plt.plot(Studio_data_copy['price'], Studio_data_copy['predict'], color='red')
plt.title('Studio: Loan Amount vs Price')
plt.xlabel('Price')
plt.ylabel('Loan Amount')
plt.show()

# Linear Regression for Independent House
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

Independent_data_copy = Independent_data.copy()

x_independent = Independent_data_copy['price'].values.reshape(-1, 1)
y_independent = Independent_data_copy['loan_amount']

model_independent = LinearRegression().fit(x_independent, y_independent)

print("Independent House Property Type:")
print(f"Coefficient: {model_independent.coef_[0]}")
print(f"Intercept: {model_independent.intercept_}")

r_squared_independent = model_independent.score(x_independent, y_independent)
print(f"R-squared: {r_squared_independent}")

Independent_data_copy['predict'] = model_independent.predict(x_independent)

plt.scatter(Independent_data_copy['price'], Independent_data_copy['loan_amount'])
plt.plot(Independent_data_copy['price'], Independent_data_copy['predict'], color='red')
plt.title('Independent House: Loan Amount vs Price')
plt.xlabel('Price')
plt.ylabel('Loan Amount')
plt.show()

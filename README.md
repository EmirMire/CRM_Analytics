# CRM_Analytics
Data Analysis and Data Visualization of Sales and Customers Data.

* Some outputs not fully displaying, for full display please see the attached ".ipynb" file.

# IMPORTING LIBRARIES

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error

# READING THE CSV FILE

    df = pd.read_csv('CRM_Data_UTF-8.csv')


# DATA CLEANING AND PREPARING

    df.head()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/b1cc425c-14ca-4160-8fa0-0f32bacbb6b7)


    df.shape

(541909, 8)


    df.info()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/744072d8-b55a-4b0f-a98c-63bc9e88d3c6)


    df.isnull().sum()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/aba202e6-051f-4d99-afaa-394a0a3822d6)


# Dropping NaN values

    df.dropna(inplace=True)


    df.isnull().sum()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/d8af8f58-5477-44b8-ab78-513a490af210)


    df.shape

(403182, 8)


# Breakdown of orders by country in %

    df['Country'].value_counts(normalize=True)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/7150459a-3199-41d2-8e67-7a4fe2d8680c)

# 89% of sales are from the UK


# DESCRIPTIVE STATISTICS

    df.describe().round(2)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/caf6bb65-4cff-42fe-85e4-7810ff3f93c0)


# Finding the MIN and MAX values 

* QUANTITY

      indeks_min = df["Quantity"].idxmin()
      indeks_max = df["Quantity"].idxmax()

      df.loc[indeks_min]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/22f41655-109f-4052-b3dc-36c29893eac5)

      df.loc[indeks_max]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/13d2f7fc-05fe-4498-ab31-5615858cb071)


* UNIT PRICE

      indeks_min_Price = df["UnitPrice"].idxmin()
      indeks_max_Price = df["UnitPrice"].idxmax()

      df.loc[indeks_min_Price]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/74c2aa84-ac55-4c1e-9582-f0d0da6502f0)

      df.loc[indeks_max_Price]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/13b869f2-8d87-498a-a8ff-e4ae603f0668)


# Creating a new column "Total Price"

    df["Total Price"] = df["Quantity"]*df["UnitPrice"]


# Excluding the invoices with "C" in "InvoiceNo", which are the invoices with negative values

    df = df[~df["InvoiceNo"].str.contains("C")]


# Checking for zero or negative values in the columns "Quantity" and "UnitPrice"

    df[df["Quantity"] <=  0]

    df[df["UnitPrice"] <=  0]


# Deleting orders where "UnitPrice" is 0

    df = df[df["UnitPrice"] >  0]


# Converting the column "InvoiceDate" from STRING to DATETIME

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Converting the column "CustomerID" from FLOAT to INT

    df['CustomerID'] = df['CustomerID'].astype(int)


    df.info()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/46e1a494-bfe1-4bad-ad26-6682c4e4c010)


    df.describe().round(2)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/11f51ab7-61a4-40b4-9b9c-fac0fa772827)


    df.duplicated().sum()

5147


# Showing duplicated rows

    duplicated_rows = df[df.duplicated(keep=False)]
    duplicated_rows.sort_values(by=["InvoiceNo","Description","Quantity"])

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/c06c4f86-a957-45e1-a4e5-30f764aaa0b9)


# Dropping duplicates

    df.drop_duplicates(inplace = True)


    df.duplicated().sum()

0


# Describing the numerical values by Country

    columns_to_describe = ["Quantity", "UnitPrice", "Total Price"]

* Group by "Country" and then describe the specified columns

      description = df.groupby("Country")[columns_to_describe].describe().round(2)
      description

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/0f6622da-ca15-4c53-9ee0-5dd051dc6cab)


# TOTAL NUMBER OF ORDERS PER COUNTRY

    countries = df["Country"].value_counts()

    plt.figure(figsize=(15, 12))
    plt.title("Number of Orders per Country", fontsize=16)
    barplot = sns.barplot(x=countries.values, y=countries.index, palette='viridis')

* Add annotations to show the numbers on each bar
  
      for index, value in enumerate(countries.values):
          barplot.text(value, index, str(value), ha='left', va='center', fontsize=11)

      plt.xlabel('Number of Orders', fontsize=14)
      plt.ylabel('Country', fontsize=14)
      plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/7e47a49b-a3b0-4f67-86aa-a93c1819d92f)


# List of products with number of orders (DESC)

    df['Description'].value_counts()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/431c5843-3fc2-4df5-964b-c68fee7ebc67)

* Most selling (TOP) product - "WHITE HANGING HEART T-LIGHT HOLDER"


# SALES QUANTITY OF TOP PRODUCT BY COUNTRIES

    TOP_PRODUCT = df[df["Description"] == "WHITE HANGING HEART T-LIGHT HOLDER"]

    plt.figure(figsize=(15,15))
    plt.title("Sales quantity of top product by countries", fontsize= 16)
    sns.barplot(x="Quantity",y='Country', data = TOP_PRODUCT, palette= 'flare')

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/fee02629-22ff-4d57-a057-aa214a5cb976)


# TOP 10 MOST EXPENSIVE ITEMS IN THE DATA

    unique_products_prices = df[['Description', 'UnitPrice']].drop_duplicates()

    unique_products_prices.sort_values(by="UnitPrice", ascending = False).head(10)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/849dcfdc-0745-4f08-be29-6faacb9b2d68)


# TOP 10 COUNTRIES BY TOTAL SALES

    Sales = df.groupby("Country", as_index = False)["Total Price"].sum()

    Sales_top10 = Sales.sort_values(by="Total Price", ascending=False).head(10)

    Sales_top10
    
![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/cdca04c2-9283-4d9e-b304-b0205e8cc237)


# THE DISTRIBUTION OF SALES IN TOP 10 COUNTRIES

    plt.figure(figsize=(18, 6))
    sns.barplot(data= Sales_top10, x= 'Country', y= 'Total Price',palette= 'muted')
    plt.title('The distribution of sales in TOP 10 countries', fontsize= 16)
    plt.xlabel('Country', fontsize= 14)
    plt.ylabel('Total Sales', fontsize = 14)
    plt.xticks(fontsize= 12)
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/bdf0b68a-1dac-42d1-9bde-578091afc1d1)


# THE DISTRIBUTION OF SALES IN THE TOP COUNTRY (UK) PER MONTH AND YEAR.

    df['Month'] = df['InvoiceDate'].dt.month

    df['Year'] = df['InvoiceDate'].dt.year

    UK = df[df["Country"] == "United Kingdom"]

    UK_Sales = UK.groupby(["Year","Month"], as_index = False)["Total Price"].sum()

    plt.figure(figsize=(18, 6))
    sns.barplot(data= UK_Sales, x= 'Month', y= 'Total Price', hue='Year',palette= 'bright')
    plt.title('The distribution of sales in the UK per month and year', fontsize= 16)
    plt.xlabel('Invoice Month', fontsize= 14)
    plt.ylabel('Total Sales', fontsize = 14)
    plt.xticks(fontsize= 12)
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/14df1886-dd3c-447f-b76e-7d861af7c415)


# THE DISTRIBUTION OF SALES IN OTHER TOP 10 COUNTRIES PER YEAR/MONTH.

    TOP10_Other = Sales_top10[~Sales_top10["Country"].str.contains("United Kingdom")]

    Sales_top10_other = df[df["Country"].isin(TOP10_Other["Country"])].groupby(["Country", "Year", "Month"], as_index = False)["Total Price"].sum()

    Sales_top10_other

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/3fd7831d-5091-4750-903f-152a8598c3ba)


# THE DISTRIBUTION OF SALES IN OTHER TOP 10 COUNTRIES PER YEAR.

    plt.figure(figsize=(18, 6))
    sns.barplot(data= Sales_top10_other, x= 'Country', y= 'Total Price',hue ="Year",palette= 'Paired')
    plt.title('The distribution of sales in other TOP 10 countries per year', fontsize= 16)
    plt.xlabel('Country', fontsize= 18)
    plt.ylabel('Total Sales', fontsize = 18)
    plt.xticks(fontsize= 14)
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/65902181-cd6e-4150-a2bd-fd2ff82310f3)


# THE DISTRIBUTION OF SALES IN OTHER TOP 10 COUNTRIES PER MONTH.

    plt.figure(figsize=(18, 6))
    sns.barplot(data= Sales_top10_other, x= 'Country', y= 'Total Price',hue ="Month",palette= 'viridis')
    plt.title('The distribution of sales in other TOP 10 countries per year', fontsize= 16)
    plt.xlabel('Country', fontsize= 18)
    plt.ylabel('Total Sales', fontsize = 18)
    plt.xticks(fontsize= 14)
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/c6b41bed-7765-4b0a-a9fe-0593298f961c)


# TOTAL MONTHLY SALES TRENDLINE.

    Sales_per_month = df.groupby("Month", as_index=False)["Total Price"].sum()

    plt.figure(figsize=(12, 6))

    sns.lineplot(data=Sales_per_month, x="Month", y="Total Price", marker='o', color='b', linewidth=2)

    plt.title('Total Sales Monthly', fontsize=18, weight='bold')
    plt.xlabel('Month', fontsize=14, weight='bold')
    plt.ylabel('Total Sales', fontsize=14, weight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for x, y in zip(Sales_per_month["Month"], Sales_per_month["Total Price"]):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/bb96c6a4-dd43-4547-85c8-aecebc94e797)


# DISTRIBUTION OF TOTAL ORDER PRICES

    sns.histplot(data=df, x='Total Price', bins=np.arange(0,500,10))

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/3c19121d-3f59-45f7-bddd-32260f5f056d)


# OUTLIERS DETECTION

    fig = plt.figure(figsize= (16, 8))
    plt.title('The distribution of the values:', fontsize= 16)

    axs = fig.subplots(nrows=2, ncols= 1)

    sns.boxplot(data = df, x= 'Quantity', ax = axs[0])
    sns.boxplot(data = df, x= 'UnitPrice',ax = axs[1])

    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/5a5ba56a-251f-4da9-a4ad-a8d111493de2)


    df['Quantity'].sort_values()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/13934d4c-553c-4a6a-8bd0-c5998206a121)


    df.loc[61619]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/cfb69d72-06e7-4efe-aa2b-c8ecd683e740)


    df['UnitPrice'].sort_values()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/14a3fad2-f82d-408a-8576-45fc017eb941)


    df.loc[173382]

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/33d8d121-0791-4f49-aea5-41625dcb3ea9)


# Delete the extreme outliers by their index

    df = df.drop(index = [61619, 173382])


# TOP 10 CUSTOMER BY NUMBER OF ORDERS

    TOP10_CUSTOMERS = df['CustomerID'].value_counts().head(10)

    TOP10_CUSTOMERS

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/75ced686-0e19-40bc-a381-83143083a98f)


# Creating a DataFrame with sorted values

    TOP10_CUSTOMERS_df = pd.DataFrame({'CustomerID': TOP10_CUSTOMERS.index, 'Count': TOP10_CUSTOMERS.values})
    TOP10_CUSTOMERS_df = TOP10_CUSTOMERS_df.sort_values(by='Count', ascending=False)

# TOP 10 CUSTOMERS BY NUMBER OF ORDERS

    plt.figure(figsize=(12, 6))
    sns.barplot(x='CustomerID', y='Count', data=TOP10_CUSTOMERS_df, palette='viridis',
                order=TOP10_CUSTOMERS_df['CustomerID']) 

    plt.title('Top 10 Customers by Number of Orders', fontsize=18, weight='bold')
    plt.xlabel('Customer ID', fontsize=14, weight='bold')
    plt.ylabel('Number of Orders', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

* Adding data labels

        for x, y in zip(range(len(TOP10_CUSTOMERS_df)), TOP10_CUSTOMERS_df['Count']):
            plt.text(x, y, f'{y}', ha='center', va='bottom', fontsize=10, color='black')

        plt.show()

  ![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/c34718ab-7c5f-4ce5-8e38-60b455f48bb8)


# TOP 10 CUSTOMER BY TOTAL SPENDINGS

    TOP10_SPENDERS = df.groupby("CustomerID", as_index = False)["Total Price"].sum()

    TOP10_SPENDERS = TOP10_SPENDERS.sort_values(by = "Total Price", ascending = False).head(10)

    TOP10_SPENDERS 

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/970870a9-e28f-4a6e-b465-66467e19353b)


    plt.figure(figsize=(12, 8))
    sns.barplot(x="CustomerID", y="Total Price", data=TOP10_SPENDERS, palette='Blues', 
                order=TOP10_SPENDERS['CustomerID'])

    plt.title('Top 10 Spenders', fontsize=18, weight='bold')
    plt.xlabel('Customer ID', fontsize=14, weight='bold')
    plt.ylabel('Total Spending', fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

* Adding data labels

        for x, y in zip(range(len(TOP10_SPENDERS)), TOP10_SPENDERS['Total Price']):
            plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/530479c5-7a9e-4341-bcad-8fe57454adc1)


# DISTRIBUTION OF TOTAL SPENDINGS

    customer_spending = df.groupby("CustomerID", as_index=False)["Total Price"].sum()

    plt.figure(figsize=(12, 8))
    sns.histplot(data=customer_spending, x='Total Price', bins=np.arange(0,10000,100))

    plt.title('DISTRIBUTION OF TOTAL SPENDINGS', fontsize=14, weight='bold')
    plt.xlabel('Total Spending', fontsize=11, weight='bold')

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/c77dc8b9-084f-42be-bbb1-7f1a749fee84)


# CLASSIFYING CUSTOMERS BASED ON THEIR TOTAL SPENDING

* Defining spending ranges and labels

        spending_bins = [0, 100, 500, 2000, 10000, float('inf')]
        spending_labels = ['Very Low (0-100)', 'Low (100-500)', 'Medium (500-2,000)', 'High (2,000-10,000)', 'Very High (> 10,000)']

* Creating a new column 'TotalSpendingCategory' based on the total spending

        customer_spending['TotalSpendingCategory'] = pd.cut(customer_spending['Total Price'], bins=spending_bins, labels=spending_labels, right=False)

        customer_spending["TotalSpendingCategory"].value_counts(normalize=True)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/a5fa88a4-b240-4cbc-b064-d8b9a253a3ab)


# CUSTOMER SPENDING DISTRIBUTION

    spending_distribution = customer_spending["TotalSpendingCategory"].value_counts(normalize=True)

    plt.figure(figsize=(8, 8))
    sns.set_palette("Blues")

    plt.pie(spending_distribution, labels=spending_distribution.index, autopct='%1.1f%%', startangle=180)
    plt.title('Customer Spending Distribution', fontsize=16, weight='bold')
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/49e8ce9a-68bc-4e79-9104-0f56aa82bfb8)


# DISTRIBUTION OF TOTAL ORDERS

    customer_orders = df.groupby("CustomerID", as_index=False)["InvoiceNo"].count()

    plt.figure(figsize=(12, 8))
    sns.histplot(data=customer_orders, x='InvoiceNo', bins=np.arange(0,1000,20))

    plt.title('DISTRIBUTION OF TOTAL ORDERS', fontsize=14, weight='bold')
    plt.xlabel('Total Orders', fontsize=11, weight='bold')

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/e64d50c9-8385-4309-bb35-25637bf6a4c3)

    
# CLASSIFYING CUSTOMERS BASED ON THEIR TOTAL ORDERS

* Defining spending ranges and labels

        spending_bins = [0, 5, 20, 50, 200, float('inf')]
        spending_labels = ['Very Low (0-5)', 'Low (5-20)', 'Medium (20-50)', 'High (50-200)', 'Very High (> 200)']

* Creating a new column 'TotalOrderCategory' based on the total spending
  
        customer_orders['TotalOrdersCategory'] = pd.cut(customer_orders['InvoiceNo'], bins=spending_bins, labels=spending_labels, right=False)

        customer_orders["TotalOrdersCategory"].value_counts(normalize=True)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/53261639-18a3-4638-bbad-4d3a6aa0b9a6)


# CUSTOMER ORDERS DISTRIBUTION

    total_orders_distribution = customer_orders["TotalOrdersCategory"].value_counts(normalize=True)

    plt.figure(figsize=(8, 8))
    plt.pie(total_orders_distribution, labels=total_orders_distribution.index, autopct='%1.1f%%', startangle=90)
    plt.title('Customer Orders Distribution', fontsize=16, weight='bold')
   
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/e0ef8acc-9e09-42c0-bfe3-9285ca03c65a)



    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error

    
# RELATIONSHIP BETWEEN TWO VARIABLES: QUANTITY AND UNIT PRICE

sns.scatterplot(data=df, x='UnitPrice', y='Quantity') 


# REGRESSION

* FEATURES - INDEPENDANT VARIABLE "UnitPrice"

      X = df['UnitPrice'].values.reshape(-1,1)

* TARGET - DEPENDANT VARIABLE "Quantity"

        y = df['Quantity'].values

* SPLITTING THE DATA INTO TRAINING SAMPLE (80%) AND TESTING SAMPLE(20%)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# REGRESSION PLOT 

    plt.figure(figsize=(12, 10))
    sns.regplot(x=X_train, y=y_train, line_kws={'color': 'blue'}, scatter_kws={'color': 'orange'})

    plt.xlabel('UnitPrice')
    plt.ylabel('Quantity')
    plt.show()

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/51a68eca-6887-4b5d-aa48-1152cc7157e3)


* HIGHER UNIT PRICES NEGATIVELY AFFECTS (DECREASES) QUANTITY PURCHASED


# LINEAR REGRESSION

        lr = LinearRegression()
        lr.fit(X_train, y_train)
  
        print ('coefficients : ',lr.coef_) 
        print ('Intercept : ',lr.intercept_)

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/78c0503c-bf8d-429a-b61b-e82910402659)


* PREDICTING THE QUANTITY IF THE UNIT PRICE IS 10.

        lr.predict(np.array([10]).reshape(1,-1))

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/7ab62dc9-1490-4483-83b4-9ea2484ae686)


  # MAE, MSE, R2
  
    y_train_pred = lr.predict(X_train)

    y_test_pred = lr.predict(X_test)    * USING THE TEST PREDICTION FOR CALCULATIONS

    print('Mean absolute error: %.2f ' % mean_absolute_error(y_test, y_test_pred))
    print('Mean sum of squares (MSE): %.2f ' % mean_squared_error(y_test, y_test_pred))
    print('R2-score: %.2f' % r2_score(y_test, y_test_pred) )

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/c6815580-ebc0-4260-934c-71c43f42329a)



# POWER BI


* SALES MAP WITH TRENDLINES OVER TIME

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/137aaa83-fa4a-4a64-b1e9-03c522f070b5)


* TOTAL SALES DISTRIBUTION MAP

![image](https://github.com/EmirMire/CRM_Analytics/assets/121452974/3eca9431-6f3b-429e-9482-55c14510a7cb)










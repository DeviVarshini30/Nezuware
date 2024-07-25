import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'C:/Users/deviv/Downloads/Cities_in_India_with_pincodes.csv/Cities_in_India_with_pincodes.csv'
df = pd.read_csv(file_path)
print(df)

print(df.describe())

# Count the number of unique cities
unique_cities = df['Location'].nunique()
print("Number of unique cities:", unique_cities)

postcodes_per_city = df.groupby('Location')['Pincode'].count()
print("Post codes per city:\n", postcodes_per_city)


city_with_most_postcodes = postcodes_per_city.idxmax()
max_postcodes = postcodes_per_city.max()
print("City with the most post codes:", city_with_most_postcodes)
print("Number of post codes:", max_postcodes)


selected_city = 'Mumbai'
filtered_df = df[df['Location'] == selected_city]
print("Post codes for", selected_city, ":\n", filtered_df)


pincode_filter = 786160
filtered_locations = df[df['Pincode'] == pincode_filter]
print(f"\nLocations with Pincode {pincode_filter}:")
print(filtered_locations)



# Filter the locations with a specific pincode
pincode_filter = 86160
filtered_locations = df[df['Pincode'] == pincode_filter]
print(f"\nLocations with Pincode {pincode_filter}:")
print(filtered_locations)


data = {
    'State': ['Assam'] * 15,
    'District': ['Tinsukia'] * 15,
    'Location': ['Adarshgaon', 'Agbandha Bengali gaon', 'Alubari', 'Amarpur', 'Amguri',
                 'Aroimuria', 'Arunodaya', 'Ashok Nagar', 'Baghbari Gaon', 'Balijan',
                 'Barchapri', 'Barhapjan', 'Barhullung', 'Barmajan', 'Baruahola'],
    'Pincode': [786174, 786187, 786181, 786157, 786160, 786154, 786160, 786170,
                786189, 786171, 786183, 786150, 786160, 786174, 786183]
}
df1 = pd.DataFrame(data)

pincode_counts = df1['Pincode'].value_counts().reset_index()
pincode_counts.columns = ['Pincode', 'Count']

plt.figure(figsize=(10, 6))
sns.barplot(x='Pincode', y='Count', data=pincode_counts)
plt.xlabel('Pincode')
plt.ylabel('Count')
plt.title('Number of Locations in Each Pincode')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8, 8))
plt.pie(pincode_counts['Count'], labels=pincode_counts['Pincode'], autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Percentage of Locations in Each Pincode')
plt.show()


plt.figure(figsize=(12, 6))
sns.countplot(x='Pincode', data=df1)
plt.xlabel('Pincode')
plt.ylabel('Location')
plt.title('Number of Locations by Pincode')
plt.xticks(rotation=45)
plt.show()



print(df.head(200))



pincode_counts = df['District'].value_counts(50)

# Create a bar plot
plt.figure(figsize=(12, 6))
pincode_counts.plot(kind='bar')
plt.title('Number of Pin Codes per District')
plt.xlabel('District')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=90)
plt.show()



pincode_counts = df['District'].value_counts()

# Select the top 50 districts
top_50_districts = pincode_counts.head(50)

# Create a bar plot
plt.figure(figsize=(12, 6))
top_50_districts.plot(kind='bar')
plt.title('Number of Pin Codes per District (Top 50)')
plt.xlabel('District')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=90)
plt.show()



# Select the top 20 districts
top_20_districts = pincode_counts.head(20)

# Create a bar plot
plt.figure(figsize=(12, 6))
top_20_districts.plot(kind='bar')
plt.title('Number of Pin Codes per District (Top 20)')
plt.xlabel('District')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=45)
plt.show()



# Count the number of pin codes per district
pincode_counts = df['District'].value_counts()


district_with_most_pin_codes = pincode_counts.idxmax()
max_pin_codes = pincode_counts.max()

print("District with the most number of pin codes:", district_with_most_pin_codes)
print("Number of pin codes:", max_pin_codes)




pincode_counts_by_province = df['State'].value_counts()

# Create a bar plot
plt.figure(figsize=(12, 6))
pincode_counts_by_province.plot(kind='bar')
plt.title('Number of Pin Codes by Province')
plt.xlabel('Province')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=80)
plt.show()



# Count the number of pin codes by state
pincode_counts_by_state = df['State'].value_counts()


filtered_states = pincode_counts_by_state[pincode_counts_by_state < 500]

# Create a bar plot
plt.figure(figsize=(12, 6))
filtered_states.plot(kind='bar')
plt.title('Number of Pin Codes by State (States with < 500 Pin Codes)')
plt.xlabel('State')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=45)
plt.show()



# Count the number of pin codes by state
pincode_counts_by_state = df['State'].value_counts()


filtered_states = pincode_counts_by_state[pincode_counts_by_state < 100]

# Create a bar plot
plt.figure(figsize=(12, 6))
filtered_states.plot(kind='bar')
plt.title('Number of Pin Codes by State (States with < 100 Pin Codes)')
plt.xlabel('State')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=45)
plt.show()



# Count the number of pin codes by state
pincode_counts_by_state = df['State'].value_counts()

# Determine the states with the highest, lowest, and middle pin code counts
highest_pincode_state = pincode_counts_by_state.idxmax()
lowest_pincode_state = pincode_counts_by_state.idxmin()
middle_pincode_state = pincode_counts_by_state.idxmin()  # Assuming there is an odd number of states


selected_states = [highest_pincode_state, lowest_pincode_state, middle_pincode_state]
selected_counts = pincode_counts_by_state[selected_states]

# Create a bar plot
plt.figure(figsize=(12, 6))
selected_counts.plot(kind='bar')
plt.title('Number of Pin Codes by State')
plt.xlabel('State')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=45)
plt.show()



# Count the number of pin codes by state
pincode_counts_by_state = df['State'].value_counts()

# Sort the states based on pin code counts in descending order
sorted_states = pincode_counts_by_state.sort_values(ascending=False)

# Determine the states with the highest and second lowest pin code counts
highest_pincode_state = sorted_states.index[0]
second_lowest_pincode_state = sorted_states.index[-2]

# Get the pin code counts for the selected states
selected_states = [highest_pincode_state, second_lowest_pincode_state]
selected_counts = pincode_counts_by_state[selected_states]

# Create a bar plot
plt.figure(figsize=(12, 6))
selected_counts.plot(kind='bar')
plt.title('Number of Pin Codes by State')
plt.xlabel('State')
plt.ylabel('Number of Pin Codes')
plt.xticks(rotation=90)
plt.show()




# Filter the dataset for Punjab state
punjab_df = df[df['State'] == 'Punjab']

# Count the number of pin codes per district in Punjab
pincode_counts_by_district = punjab_df['District'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(pincode_counts_by_district, labels=pincode_counts_by_district.index, autopct='%1.1f%%', startangle=90)
plt.title('Pin Codes in Punjab by District')
plt.axis('equal')
plt.show()

















































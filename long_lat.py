import pandas as pd

data_country = pd.read_csv("https://developers.google.com/public-data/docs/canonical/countries_csv")

oldCountry = data["country"]
result_long = []
result_lat = []
result_found = []

for i in range(len(country_name)):
	result_found.append(false)
	for j in range(len(data_country["name"])):
		if country_name[i] == data_country["name"][j]:
			result_long.append(data_country["longitude"][j])
			result_lat.append(data_country["latitude"][j])
			result_found[i] = true

print(str(result_lat))
print(str(result_long))

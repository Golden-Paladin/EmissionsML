import csv

#
MAX_LINES = 15611
TEST_LINES_START = 15612

#
state = []
year = []
facility_name = []
industrySecter = []
totaEmissions = []

setUnique = set()
tupleToEmissions = {}
emmisionsSum2021 = 0
emmisionsSum2022 = 0
emmisionsSum2023 = 0

minNy = float('inf')
maxNy = float('-inf')
sum2023 = 0
num2023 = 0

with open('../epa_ghgrp_2021_2023_aggregate.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for i, row in enumerate(reader, start=1):
        state.append(row['state'])
        year.append(row['reporting_year'])
        facility_name.append(row['facility_name'])
        industrySecter.append(row['industry_sector'])
        totaEmissions.append(row['total_ghg_emissions_tonnes'])
        key = (row['industry_sector'], row['state'])
        setUnique.add(key)


        if row['industry_sector'] == 'Power Plants' and row['state'] == 'NY' and row['reporting_year'] == '2023':
            sum2023 += float(row['total_ghg_emissions_tonnes'])
            num2023 += 1


        if row['industry_sector'] == 'Power Plants' and row['state'] == 'NY':
            print(float(row['total_ghg_emissions_tonnes']))
            if float(row['total_ghg_emissions_tonnes']) >= maxNy:
                maxNy = float(row['total_ghg_emissions_tonnes'])
            if float(row['total_ghg_emissions_tonnes']) <= minNy:
                minNy = float(row['total_ghg_emissions_tonnes'])

        # Calculate sums in the same loop
        if row['reporting_year'] == '2021':
            emmisionsSum2021 += float(row['total_ghg_emissions_tonnes'])
        if row['reporting_year'] == '2022':
            emmisionsSum2022 += float(row['total_ghg_emissions_tonnes'])
        if row['reporting_year'] == '2023':
            emmisionsSum2023 += float(row['total_ghg_emissions_tonnes'])

        # Update the tupleToEmissions dictionary
        if key in setUnique:
            if key not in tupleToEmissions:
                tupleToEmissions[key] = 0
            tupleToEmissions[key] += float(row['total_ghg_emissions_tonnes'])

        if i >= MAX_LINES:
            break

print()
print(num2023)
print(sum2023)
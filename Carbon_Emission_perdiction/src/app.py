import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('/Users/abhishekkanade/Documents/notebook/My_Project/Carbon_Emission_perdiction/src/model/forecasting_co2_emmision.pkl')

# Load the cleaned data
data = pd.read_csv('/Users/abhishekkanade/Documents/notebook/My_Project/Carbon_Emission_perdiction/src/data/data_cleaned.csv')

countries=['Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Bangladesh', 'Bulgaria', 'Belarus', 'Bolivia', 'Brazil', 'Canada', 'Switzerland', 'Chile', 'China', "Cote d'Ivoire", 'Cameroon', 'Congo, Rep.', 'Colombia', 'Costa Rica', 'Czech Republic', 'Germany', 'Denmark', 'Dominican Republic', 'Algeria', 'Europe & Central Asia', 'Ecuador', 'Egypt, Arab Rep.', 'Euro area', 'Spain', 'Ethiopia', 'Finland', 'France', 'United Kingdom', 'Ghana', 'Greece', 'Guatemala', 'Honduras', 'Hungary', 'Indonesia', 'India', 'Ireland', 'Iran, Islamic Rep.', 'Israel', 'Italy', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Korea, Rep.', 'Latin America & Caribbean', 'Low income', 'Lower middle income', 'Low & middle income', 'Morocco', 'Mexico', 'Middle income', 'Middle East & North Africa', 'Mozambique', 'Malaysia', 'Nigeria', 'Netherlands', 'New Zealand', 'Pakistan', 'Panama', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Paraguay', 'Romania', 'Russian Federation', 'South Asia', 'Saudi Arabia', 'Sudan', 'Senegal', 'El Salvador', 'Sub-Saharan Africa', 'Sweden', 'Syrian Arab Republic', 'Togo', 'Thailand', 'Turkey', 'Tanzania', 'Ukraine', 'Upper middle income', 'Uruguay', 'United States', 'Uzbekistan', 'Venezuela, RB', 'Vietnam', 'World', 'Yemen, Rep.', 'South Africa', 'Congo, Dem. Rep.', 'Zambia']
# Define the list of selected countries for forecasting
country_code = ['AGO', 'ARG', 'ARM', 'AUS', 'AUT', 'BGD', 'BGR', 'BLR', 'BOL', 'BRA', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COG', 'COL', 'CRI', 'CZE', 'DEU', 'DNK', 'DOM', 'DZA', 'ECA', 'ECU', 'EGY', 'EMU', 'ESP', 'ETH', 'FIN', 'FRA', 'GBR', 'GHA', 'GRC', 'GTM', 'HND', 'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'ISR', 'ITA', 'JOR', 'JPN', 'KAZ', 'KEN', 'KOR', 'LAC', 'LIC', 'LMC', 'LMY', 'MAR', 'MEX', 'MIC', 'MNA', 'MOZ', 'MYS', 'NGA', 'NLD', 'NZL', 'PAK', 'PAN', 'PER', 'PHL', 'POL', 'PRT', 'PRY', 'ROM', 'RUS', 'SAS', 'SAU', 'SDN', 'SEN', 'SLV', 'SSA', 'SWE', 'SYR', 'TGO', 'THA', 'TUR', 'TZA', 'UKR', 'UMC', 'URY', 'USA', 'UZB', 'VEN', 'VNM', 'WLD', 'YEM', 'ZAF', 'ZAR', 'ZMB']

selected_features = ['cereal_yield', 'gni_per_cap', 'en_per_cap',
                     'pop_urb_aggl_perc', 'prot_area_perc',
                     'pop_growth_perc', 'urb_pop_growth_perc']

# Function to calculate CAGR
def calculate_cagr(start_value, end_value, years):
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    return (end_value / start_value) ** (1 / years) - 1

# Function to forecast CO₂ emissions
def forecast_co2_emissions(country, future_years, growth_rates,selected_features ):
    country_data = data[data['country'] == country].sort_values('year')
    latest_row = country_data.iloc[-1][selected_features].copy()
    forecast_results = []
    
    for year in future_years:
        for feature in selected_features:
            growth_rate = growth_rates.get(country, {}).get(feature, 0.0)
            latest_row[feature] *= (1 + growth_rate)

        input_features = latest_row.values.reshape(1, -1)
        predicted_co2 = model.predict(input_features)[0]
        forecast_results.append({'year': year, 'co2_percap': predicted_co2})

    return pd.DataFrame(forecast_results)

# Function to calculate growth rates for selected features
# This function calculates the Compound Annual Growth Rate (CAGR) for each feature over the available
def growth_rates_calculation(country, selected_features):
    
    df_filtered = data[data['country'] == country]

    growth_rates = {}


    country_data = df_filtered[(data['country'] == country)].sort_values('year')

    start_year = country_data['year'].min()
    end_year = country_data['year'].max()
    years = end_year - start_year

    country_growth = {}

    if years  <= 0:
        print(f"Skipping {country} due to insufficient year range.")
        exit()

    for feature in selected_features:
        start_value = country_data[country_data['year'] == start_year][feature].values

        end_value = country_data[country_data['year'] == end_year][feature].values

        if len(start_value) == 0 or len(end_value) == 0:
            continue
        start_value = start_value[0]
        end_value = end_value[0]    

        if start_value <= 0 or end_value <= 0 or not np.isfinite(start_value) or not np.isfinite(end_value):
            continue

        cagr = (end_value / start_value) ** (1/years) - 1
        country_growth[feature] = cagr  # Convert to percentage

    growth_rates[country] = country_growth
    return growth_rates
               

# Streamlit app layout
st.title("CO₂ Emissions Forecasting")
st.write("Forecast CO₂ emissions per capita for selected countries over the next 20 years.")

# User input for country selection
country = st.selectbox("Select a country:", countries)
country = country_code[countries.index(country)]  # Convert country name to code
# Define the range of years to forecast
last_year = data['year'].max()
future_years = list(range(last_year + 1, last_year + 21))

# Placeholder for growth rates (to be calculated or loaded)
growth_rates = growth_rates_calculation(country, selected_features)

# Forecast CO₂ emissions when the button is clicked
if st.button("Forecast CO₂ Emissions"):
    df_forecast = forecast_co2_emissions(country, future_years, growth_rates,selected_features )
    st.write(f"Forecasted CO₂ Emissions per Capita for {countries[country_code.index(country)]} over the next 20 years:")
    st.line_chart(df_forecast.set_index('year')['co2_percap'], use_container_width=True,x_label="Year", y_label="CO₂ Emissions per Capita (metric tons)")
    st.write("Yearly Forecast Data:")
    st.dataframe(df_forecast)

    
    
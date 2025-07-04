# CO₂ Emission Prediction Streamlit App

This project is a Streamlit web application designed to forecast CO₂ emissions per capita for selected countries over the next 20 years. It utilizes a trained Random Forest model to make predictions based on user inputs and visualizes the results in an interactive interface.

## Project Structure

```
co2-emission-streamlit-app
├── src
│   ├── app.py                  # Main entry point for the Streamlit application
│   ├── model
│   │   └── forecasting_co2_emmision.pkl  # Trained Random Forest model
│   ├── utils
│   │   └── forecast.py         # Utility functions for data processing and predictions
│   └── data
│       └── data_cleaned.csv    # Cleaned dataset used for training and predictions
├── requirements.txt            # List of dependencies for the application
└── README.md                   # Documentation for the project
```

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd co2-emission-streamlit-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To start the Streamlit application, run the following command in your terminal:

```
streamlit run src/app.py
```

This will launch the application in your default web browser.

## Functionality

- Users can select countries to visualize forecasted CO₂ emissions per capita for the next 20 years.
- The application displays a line plot of the forecasted emissions, allowing users to compare trends across different countries.
- The last 5 years of forecasted data for India are displayed in a tabular format.

## Dependencies

The application requires the following Python packages:

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Make sure all dependencies are installed as specified in the `requirements.txt` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
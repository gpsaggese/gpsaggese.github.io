from data_processing.load_data import DataLoader

gdrive_urls = {
    "mobility": "https://drive.google.com/open?id=1TMqG8Z8vbxmQAv1rNKczYYPCzwT4ZS_q",
    "cases": "https://drive.google.com/open?id=1ZfZtoV3PpZblZYES0A5LHCwp54cR8RJL",
    "deaths": "https://drive.google.com/open?id=1kYC9nrCnKbNpnoZKz8o6TDMM371gyxbl",
    "vaccine": "https://drive.google.com/open?id=1ulTFLBbZxz636_PFqQvixLpqQV9s-P_v",
}

loader = DataLoader(gdrive_urls=gdrive_urls, data_dir="data")
# Download files (will skip if file already exists)
loader.download_all()
# Load CSVs into dataframes
loader.load_all()

# Access a dataframe:
df_cases = loader.get_dataframe('cases')
print(df_cases.shape)
print("DataLoader setup complete. DataFrames are loaded and accessible.")


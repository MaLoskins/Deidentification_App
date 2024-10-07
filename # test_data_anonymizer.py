# test_data_anonymizer.py

import pandas as pd
from src.data_anonymizer import DataAnonymizer
import datetime

def debug_print(message):
    """Simple debug callback that prints messages to the console."""
    print(f"DEBUG: {message}")

def create_sample_dataframe():
    """Creates a sample DataFrame with various dtypes."""
    data = {
        'Age': [25, 34, 45, 23, 35, 40, 29, 31, 38, 27],
        'Salary': [50000.0, 60000.5, 75000.0, 48000.0, 52000.0, 61000.0, 58000.0, 62000.0, 73000.0, 49000.0],
        'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
        'Department': ['Sales', 'Engineering', 'HR', 'Sales', 'Engineering', 'HR', 'Sales', 'Engineering', 'HR', 'Sales'],
        'Joining_Date': [
            '2015-06-01', '2016-07-15', '2017-08-20', '2018-09-25', '2019-10-30',
            '2020-11-05', '2021-12-10', '2022-01-15', '2023-02-20', '2024-03-25'
        ],
        'Sensitive_Info': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C']
    }

    df = pd.DataFrame(data)
    
    # Convert data types
    df['Age'] = df['Age'].astype(int)
    df['Salary'] = df['Salary'].astype(float)
    df['Gender'] = df['Gender'].astype('category')
    df['Department'] = df['Department'].astype('category')
    df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])
    df['Sensitive_Info'] = df['Sensitive_Info'].astype('category')
    
    return df

def main():
    # Step 1: Create the sample DataFrame
    df = create_sample_dataframe()
    print("Original DataFrame:")
    print(df)
    print("\nDataFrame dtypes:")
    print(df.dtypes)
    print("\n" + "="*50 + "\n")
    
    # Step 2: Instantiate DataAnonymizer with debug callback
    k_value = 2  # Example k value
    anonymizer = DataAnonymizer(original_data=df, k=k_value, debug_callback=debug_print)
    
    # Define quasi-identifiers and sensitive attribute
    # Exclude 'Joining_Date' to prevent uniqueness
    quasi_identifiers = ['Age', 'Gender', 'Department']
    sensitive_attribute = 'Sensitive_Info'
    
    # Step 3: Apply k-anonymity
    print("\nApplying k-anonymity...")
    anonymizer.anonymize(
        method='k-anonymity',
        quasi_identifiers=quasi_identifiers,
        generalize_bin_size=5,  # Example bin size
        cat_threshold=0.2,      # Example categorical threshold
        datetime_freq='Y'       # Yearly generalization (not used since 'Joining_Date' is excluded)
    )
    
    # Retrieve and print anonymized data and report
    anonymized_df_k = anonymizer.get_anonymized_data()
    report_k = anonymizer.get_report()
    
    print("\nAnonymized DataFrame after k-anonymity:")
    print(anonymized_df_k)
    print("\nAnonymization Report:")
    print(report_k)
    print("\n" + "="*50 + "\n")
    
    # Step 4: Apply l-diversity
    print("\nApplying l-diversity...")
    anonymizer.anonymize(
        method='l-diversity',
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        generalize_bin_size=5,
        cat_threshold=0.2,
        datetime_freq='Y'
    )
    
    # Retrieve and print anonymized data and report
    anonymized_df_l = anonymizer.get_anonymized_data()
    report_l = anonymizer.get_report()
    
    print("\nAnonymized DataFrame after l-diversity:")
    print(anonymized_df_l)
    print("\nAnonymization Report:")
    print(report_l)
    print("\n" + "="*50 + "\n")
    
    # Step 5: Apply t-closeness
    print("\nApplying t-closeness...")
    anonymizer.anonymize(
        method='t-closeness',
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        generalize_bin_size=5,
        cat_threshold=0.2,
        datetime_freq='Y'
    )
    
    # Retrieve and print anonymized data and report
    anonymized_df_t = anonymizer.get_anonymized_data()
    report_t = anonymizer.get_report()
    
    print("\nAnonymized DataFrame after t-closeness:")
    print(anonymized_df_t)
    print("\nAnonymization Report:")
    print(report_t)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()

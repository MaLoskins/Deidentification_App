# another_script.py

from Process_Data import DataProcessor

def run_processing():
    processor = DataProcessor(
        input_filepath='Data.csv',
        output_filepath='Processed_Data.csv',
        report_path='Type_Conversion_Report.csv',
        return_category_mappings=True,
        mapping_directory='Category_Mappings',
        parallel_processing=False,
        date_threshold=0.6,
        numeric_threshold=0.9,
        factor_threshold_ratio=0.2,
        factor_threshold_unique=500,
        dayfirst=True,
        log_level='INFO',
        log_file=None,
        convert_factors_to_int=True,
        save_type = 'pickle'
    )
    processor.process()

run_processing()

Data = pd.read_csv('Processed_Data.csv') 
# src/info.py

import streamlit as st

# Define a dictionary to hold tooltip texts
TOOLTIP_TEXTS = {
    "upload_section": "Upload your dataset in CSV or Pickle format to begin processing.",
    "output_file_type": "Choose the desired format for the output file. Note that CSV may lose some metadata.",
    "binning_method": "Select the method for binning your data. Quantile ensures equal distribution, while Equal Width divides data into bins of equal size.",
    "binning_columns": "Select the columns you wish to bin. Binning helps in data generalization and privacy preservation.",
    "geocoding": "Geocoding converts location data into geographic coordinates, enabling map visualizations and spatial analysis.",
    "granularity": "Choose the level of location granularity for your data, ranging from address-level to continent-level.",
    "unique_id_columns": "Select columns that will be analyzed for unique identification to assess data privacy risks.",
    "k_anonymity_columns": "Choose columns for which you want to enforce k-anonymity to protect individual data points.",
    "k_value": "Set the value of k for k-anonymity. A higher k increases privacy but may reduce data utility.",
    # Add more tooltip keys and texts as needed
}

# CSS for tooltip styling
TOOLTIP_CSS = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
    color: blue;
    border-bottom: 1px dotted black;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    padding: 5px 10px;
    border-radius: 6px;

    /* Position the tooltip text */
    position: absolute;
    z-index: 1;
    bottom: 125%; /* Adjust to position the tooltip above the element */
    left: 50%;
    margin-left: -100px;

    /* Fade in effect */
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

def load_tooltip_css():
    """Injects the tooltip CSS into the Streamlit app."""
    st.markdown(TOOLTIP_CSS, unsafe_allow_html=True)

def get_tooltip_html(key: str) -> str:
    """
    Returns the HTML string for a tooltip based on the provided key.

    Args:
        key (str): The key identifying the tooltip text.

    Returns:
        str: HTML string for the tooltip.
    """
    tooltip_text = TOOLTIP_TEXTS.get(key, "No tooltip available.")
    tooltip_html = f"""
    <span class='tooltip' title=''>
        &#9432; <!-- Unicode for info symbol -->
        <span class='tooltiptext'>{tooltip_text}</span>
    </span>
    """
    return tooltip_html

def add_tooltip(label: str, tooltip_key: str) -> str:
    """
    Combines a label with its corresponding tooltip.

    Args:
        label (str): The main text label.
        tooltip_key (str): The key to fetch tooltip text.

    Returns:
        str: Combined HTML string with label and tooltip.
    """
    tooltip_html = get_tooltip_html(tooltip_key)
    combined_html = f"{label} {tooltip_html}"
    return combined_html

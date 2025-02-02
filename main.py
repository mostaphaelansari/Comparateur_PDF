# Import necessary libraries
import streamlit as st  # For building the web interface
import os  # For file operations
import re  # For regular expressions
import json  # For JSON handling
import requests  # For making HTTP requests
import numpy as np  # For numerical operations
import zipfile  # For working with ZIP files
from datetime import datetime  # For date and time handling
from PIL import Image, ImageEnhance, ImageFilter, ExifTags  # For image processing
import pdfplumber # For PDF text extraction
from inference_sdk import InferenceHTTPClient  # For model inference
import easyocr  # For OCR (Optical Character Recognition)
from pyzbar.pyzbar import decode  # For barcode decoding
import io  # For input/output operations
import asyncio
import concurrent.futures
import tempfile  # For temporary file handling

# Configuration variables
API_URL = "https://detect.roboflow.com"  # URL for the inference API
MODEL_ID = "medical-object-classifier/3"  # Model ID for classification
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}  # Allowed file types

# Initialize clients
client = InferenceHTTPClient(api_url=API_URL, api_key=st.secrets["API_KEY"])  # Initialize inference client
reader = easyocr.Reader(['en'], gpu=True)  # Initialize OCR reader

# Session state initialization
if 'processed_data' not in st.session_state:  # Check if session state exists
    st.session_state.processed_data = {  # Initialize session state for processed data
        'RVD': {},  # Data from RVD reports
        'AEDG5': {},  # Data from AED G5 reports
        'AEDG3': {},  # Data from AED G3 reports
        'images': [],  # Processed images
        'files': [],  # Uploaded files
        'comparisons': {  # Comparison results
            'rvd_vs_aed': {},  # RVD vs AED comparison
            'rvd_vs_images': {}  # RVD vs images comparison
        }
    }

if 'dae_type' not in st.session_state:  # Check if AED type is set
    st.session_state.dae_type = 'G5'  # Default AED type to G5

# Function to fix image orientation based on EXIF data
def fix_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():  # Loop through EXIF tags
            if ExifTags.TAGS[orientation] == 'Orientation':  # Find the Orientation tag
                break
        exif = dict(img._getexif().items())  # Get EXIF data as a dictionary
        if exif.get(orientation) == 3:  # Rotate 180 degrees
            img = img.rotate(180, expand=True)
        elif exif.get(orientation) == 6:  # Rotate 270 degrees
            img = img.rotate(270, expand=True)
        elif exif.get(orientation) == 8:  # Rotate 90 degrees
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):  # Handle exceptions
        pass
    return img  # Return the corrected image

# Async function for OCR processing
@st.cache_data  # Cache the results for efficiency
async def process_ocr(image):
    return reader.readtext(np.array(image))  # Perform OCR on the image

# Async function for classifying images
async def classify_image(image_path):
    return client.infer(image_path, model_id=MODEL_ID)  # Use the inference client to classify the image

# Async function for extracting text from PDFs
async def extract_text_from_pdf(uploaded_file):
    text = ""  # Initialize an empty string for extracted text
    try:
        with pdfplumber.open(uploaded_file) as pdf:  # Open the PDF file
            for page in pdf.pages:  # Loop through each page
                text += page.extract_text()  # Extract text from the page
    except Exception as e:  # Handle exceptions
        st.error(f"Error extracting text from PDF: {str(e)}")  # Display error message
    return text  # Return the extracted text


def extract_rvd_data(text):
    keywords = [
        "Commentaire fin d'intervention et recommandations",
        "Numéro de série DEFIBRILLATEUR",
        "Date-Heure rapport vérification défibrillateur",
        "Changement batterie",
        "Changement électrodes adultes",
        "Code site",
        "Numéro de série Batterie",
        "Date mise en service BATTERIE",
        "Niveau de charge de la batterie en %",
        "N° série nouvelle batterie",
        "Date mise en service",
        "Niveau de charge nouvelle batterie",
        "Numéro de série ELECTRODES ADULTES",
        "Numéro de série ELECTRODES ADULTES relevé",
        "Numéro de série relevé 2",
        "Date fabrication DEFIBRILLATEUR",
        "Date fabrication BATTERIE",
        "Date fabrication relevée",
        "Date fabrication nouvelle batterie",
        "Date de péremption ELECTRODES ADULTES",
        "Date de péremption ELECTRODES ADULTES relevée",
        "N° série nouvelles électrodes",
        "Date péremption des nouvelles éléctrodes",
    ]
    results = {}  # Initialize results dictionary
    for keyword in keywords:  # Loop through each keyword
        pattern = re.compile(re.escape(keyword) + r"[\s:]*([^\n]*)")  # Create regex pattern
        match = pattern.search(text)  # Search for the keyword in the text
        if match:  # If a match is found
            value = match.group(1).strip()  # Extract the value
            results[keyword] = value  # Add the value to the results dictionary
    return results  # Return the extracted data

# Function to extract AED G5 data from text
def extract_aed_g5_data(text):
    if not isinstance(text, str):  # Validate input
        st.error("Invalid input: Expected a string for text extraction.")
        return {}
    
    keywords = [
        "N° série DAE",
        "Capacité restante de la batterie",
        "Date d'installation :",
        "Rapport DAE - Erreurs en cours",
        "Date / Heure:",
    ]
    results = {}
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword) + r"[\s:]*([^\n]*)")
        match = pattern.search(text)
        if match:
            results[keyword] = match.group(1).strip()
    return results

# Function to extract AED G3 data from text
def extract_aed_g3_data(text):
    keywords = [
        "Série DSA",
        "Dernier échec de DSA",
        "Numéro de lot",
        "Date de mise en service",
        "Capacité initiale de la batterie 12V",
        "Capacité restante de la batterie 12V",
        "Autotest",
    ]
    results = {}  # Initialize results dictionary
    lines = text.split('\n')  # Split the text into lines
    for i, line in enumerate(lines):  # Loop through each line
        for keyword in keywords:  # Loop through each keyword
            if keyword in line:  # If the keyword is found in the line
                value = lines[i+1].strip() if i+1 < len(lines) else ""  # Extract the value
                results[keyword] = value  # Add the value to the results dictionary
    return results  # Return the extracted data

def extract_important_info_g3(results):
    serial_number = None
    date_of_fabrication = None
    serial_pattern = r'\b(\d{5,10})\b'
    date_pattern = r'\b(\d{4}-\d{2}-\d{2}|\d{6})\b'

    for _, text, _ in results:
        if not serial_number:
            serial_search = re.search(serial_pattern, text)
            if serial_search:
                serial_number = serial_search.group(1)
        if not date_of_fabrication:
            date_search = re.search(date_pattern, text)
            if date_search:
                date_of_fabrication = date_search.group(1)

    return serial_number, date_of_fabrication

async def extract_important_info_g5(results):
    serial_number = None
    date_of_fabrication = None
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    serial_pattern = r"([A-Za-z]*\s*[\dOo]+)"
    sn_found = False
    for _, text, _ in results:
        if "SN" in text or "Serial Number" in text:
            sn_found = True
            continue
        if sn_found:
            serial_match = re.search(serial_pattern, text)
            if serial_match:
                processed_serial = serial_match.group().replace('O', '0').replace('o', '0')
                serial_number = processed_serial
                sn_found = False
        date_search = re.search(date_pattern, text)
        if date_search:
            date_of_fabrication = date_search.group(1)
    return serial_number, date_of_fabrication
    
def extract_important_info_batterie(results):
    serial_number = None
    date_of_fabrication = None
    date_pattern = r"(\d{4}-\d{2}-\d{2})"
    sn_pattern = r"\b(?:SN|LOT|Lon|Loz|Lo|LO|Lot|Lool|LOTI|Lotl|LOI|Lod)\b"
    serial_number_pattern = r"\b(?:SN|LOT|Lon|Loz|Lot|Lotl|LoI|Lool|Lo|Lod|LO|LOTI|LOI)?\s*([0-9A-Za-z\-]{5,})\b"
    sn_found = False

    for _, text, _ in results:
        if re.search(sn_pattern, text, re.IGNORECASE):
            sn_found = True
            continue
        if sn_found:
            serial_match = re.search(serial_number_pattern, text)
            if serial_match:
                serial_number = serial_match.group(1)
                sn_found = False
        if re.search(date_pattern, text):
            date_of_fabrication = re.search(date_pattern, text).group(0)

    return serial_number, date_of_fabrication

def extract_important_info_electrodes(image):
    try:
        width, height = image.size
        crop_box = (width * 0.2, height * 0.10, width * 1, height * 1)
        cropped_image = image.crop(crop_box)
        
        enhancer = ImageEnhance.Contrast(cropped_image)
        enhanced_image = enhancer.enhance(2.5)
        enhanced_image = enhanced_image.filter(ImageFilter.SHARPEN)
        
        barcodes = decode(enhanced_image)
        serial_number, expiration_date = None, None
        
        if barcodes:
            if len(barcodes) >= 2:
                serial_number = barcodes[0].data.decode('utf-8')
                expiration_date = barcodes[1].data.decode('utf-8')
        
        return serial_number, expiration_date
    except Exception as e:
        st.error(f"Erreur lors du traitement de l'image : {e}")
        return None, None

# Function to parse dates
def parse_date(date_str):
    formats = [  # List of supported date formats
        '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y',
        '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d',
        '%d %b %Y', '%d %B %Y'
    ]
    if not date_str or str(date_str).lower() == 'nan':  # Check for empty or invalid dates
        return None, "No date provided"
    clean_date = str(date_str).split()[0].strip()  # Clean the date string
    for fmt in formats:  # Loop through each format
        try:
            return datetime.strptime(clean_date, fmt).date(), None  # Try parsing the date
        except ValueError:  # Handle parsing errors
            continue
    return None, f"Unrecognized format: {clean_date}"  # Return error if no format matches

# Async function to normalize serial numbers
async def normalize_serial(serial):
    return re.sub(r'[^A-Z0-9]', '', str(serial).upper())  # Remove non-alphanumeric characters and convert to uppercase

# Async function to compare RVD and AED data
async def compare_rvd_aed():
    try:
        results = {}
        aed_type = f'AEDG{st.session_state.dae_type[-1]}'
        
        if not st.session_state.processed_data.get('RVD'):
            st.error("Données RVD manquantes pour la comparaison")
            return {}
        if not st.session_state.processed_data.get(aed_type):
            st.error(f"Données {aed_type} manquantes pour la comparaison")
            return {}

        rvd = st.session_state.processed_data['RVD']
        aed = st.session_state.processed_data[aed_type]
        
        # Comparaison du numéro de série
        aed_key = 'N° série DAE' if st.session_state.dae_type == 'G5' else 'Série DSA'
        results['serial'] = {
            'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
            'aed': aed.get(aed_key, 'N/A'),
            'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) == 
                    normalize_serial(aed.get(aed_key, ''))
        }
        
        # Comparaison des dates
        rvd_date, rvd_err = parse_date(rvd.get('Date-Heure rapport vérification défibrillateur', ''))
        aed_date_key = 'Date / Heure:' if st.session_state.dae_type == 'G5' else 'Date de mise en service'
        aed_date, aed_err = parse_date(aed.get(aed_date_key, ''))
        results['date'] = {
            'rvd': rvd.get('Date-Heure rapport vérification défibrillateur', 'N/A'),
            'aed': aed.get(aed_date_key, 'N/A'),
            'match': rvd_date == aed_date if not (rvd_err or aed_err) else False,
            'errors': [e for e in [rvd_err, aed_err] if e]
        }
        
        # Comparaison de la batterie
        try:
            rvd_batt = float(rvd.get('Niveau de charge de la batterie en %', 0))
            aed_batt_text = aed.get('Capacité restante de la batterie', '0') if st.session_state.dae_type == 'G5' \
                        else aed.get('Capacité restante de la batterie 12V', '0')
            aed_batt = float(re.search(r'\d+', aed_batt_text).group())
            results['battery'] = {
                'rvd': f"{rvd_batt}%",
                'aed': f"{aed_batt}%",
                'match': abs(rvd_batt - aed_batt) <= 2
            }
        except Exception as e:
            results['battery'] = {
                'error': f"Données de batterie invalides : {str(e)}",
                'match': False
            }
        
        st.session_state.processed_data['comparisons']['rvd_vs_aed'] = results
        return results
    
    except KeyError as e:
        st.error(f"Clé de données manquante : {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Erreur de comparaison : {str(e)}")
        return {}

# Async function to compare RVD and image data
async def compare_rvd_images():
    try:
        results = {}
        if not st.session_state.processed_data.get('RVD'):
            st.error("Données RVD manquantes pour la comparaison")
            return {}
            
        rvd = st.session_state.processed_data['RVD']
        images = st.session_state.processed_data['images']
        
        # Comparaison de la batterie
        battery_data = next((i for i in images if i['type'] == 'Batterie'), None)
        if battery_data:
            results['battery_serial'] = {
                'rvd': rvd.get('N° série nouvelle batterie', 'N/A'),
                'image': battery_data.get('serial', 'N/A'),
                'match': normalize_serial(rvd.get('N° série nouvelle batterie', '')) == 
                        normalize_serial(battery_data.get('serial', ''))
            }
            
            rvd_date, rvd_err = parse_date(rvd.get('Date fabrication nouvelle batterie', ''))
            img_date, img_err = parse_date(battery_data.get('date', ''))
            results['battery_date'] = {
                'rvd': rvd.get('Date fabrication nouvelle batterie', 'N/A'),
                'image': battery_data.get('date', 'N/A'),
                'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                'errors': [e for e in [rvd_err, img_err] if e]
            }
        
        # Comparaison des électrodes
        electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
        if electrode_data:
            results['electrode_serial'] = {
                'rvd': rvd.get('N° série nouvelles électrodes', 'N/A'),
                'image': electrode_data.get('serial', 'N/A'),
                'match': normalize_serial(rvd.get('N° série nouvelles électrodes', '')) == 
                        normalize_serial(electrode_data.get('serial', ''))
            }
            
            rvd_date, rvd_err = parse_date(rvd.get('Date péremption des nouvelles éléctrodes', ''))
            img_date, img_err = parse_date(electrode_data.get('date', ''))
            results['electrode_date'] = {
                'rvd': rvd.get('Date péremption des nouvelles éléctrodes', 'N/A'),
                'image': electrode_data.get('date', 'N/A'),
                'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                'errors': [e for e in [rvd_err, img_err] if e]
            }
        
        st.session_state.processed_data['comparisons']['rvd_vs_images'] = results
        return results
    
    except KeyError as e:
        st.error(f"Clé de données manquante : {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Erreur de comparaison : {str(e)}")
        return {}

def display_comparison(title, comparison):
    if not comparison:
        st.warning("Aucune donnée de comparaison disponible")
        return
    
    st.subheader(title)
    
    for field, data in comparison.items():
        with st.container():
            cols = st.columns([3, 2, 2, 1])
            
            cols[0].markdown(f"**{field.replace('_', ' ').title()}**")
            cols[1].markdown(f"*RVD:*  \n`{data.get('rvd', 'N/A')}`")
            
            compare_type = 'AED' if 'aed' in data else 'Image'
            compare_value = data.get(compare_type.lower(), 'N/A')
            cols[2].markdown(f"*{compare_type}:*  \n`{compare_value}`")
            
            if data.get('match', False):
                cols[3].success("✅")
            else:
                cols[3].error("❌")
                
            if 'errors' in data:
                for err in data['errors']:
                    st.error(err)
            if 'error' in data:
                st.error(data['error'])
                
        st.markdown("---")

def main():
    st.set_page_config(page_title="Medical Device Inspector", layout="wide")
    # Header Section
    st.title("Medical Device Inspection System")
    st.markdown("---")

    # Configuration Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.session_state.dae_type = st.radio("AED Type", ("G5", "G3"), index=0)
        st.session_state.enable_ocr = st.checkbox("Enable OCR Processing", True)
        st.markdown("---")
        st.write("Developed by [Your Company]")

    # File Upload Section
    with st.expander("Upload Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Drag and drop files here",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help="Upload PDF reports and device images"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = asyncio.run(extract_text_from_pdf(uploaded_file))
                    if 'rapport de vérification' in uploaded_file.name.lower():
                        st.session_state.processed_data['RVD'] = extract_rvd_data(text)
                        st.success(f"Processed RVD: {uploaded_file.name}")
                    elif 'aed' in uploaded_file.name.lower():
                        if st.session_state.dae_type == "G5":
                            st.session_state.processed_data['AEDG5'] = extract_aed_g5_data(text)
                        else:
                            st.session_state.processed_data['AEDG3'] = extract_aed_g3_data(text)
                        st.success(f"Processed AED {st.session_state.dae_type} Report: {uploaded_file.name}")
                else:
                    try:
                        image = Image.open(uploaded_file)
                        image = fix_orientation(image)
                        image = image.convert('RGB')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                            image.save(temp_file, format='JPEG')
                            temp_file_path = temp_file.name
                        result = asyncio.run(classify_image(temp_file_path))  # Await the async function
                        detected_classes = [
                            pred['class'] for pred in result.get('predictions', [])
                            if pred['confidence'] > 0.5
                        ]
                        if detected_classes:
                            img_data = {
                                'type': detected_classes[0],
                                'serial': None,
                                'date': None,
                                'image': image
                            }
                            if "Defibrillateur" in detected_classes[0]:
                                results = asyncio.run(process_ocr(image))  # Await the async function
                                if "G3" in detected_classes[0]:
                                    img_data['serial'], img_data['date'] = extract_important_info_g3(results)
                                else:
                                    img_data['serial'], img_data['date'] = asyncio.run(
                                        extract_important_info_g5(results)
                                    )
                            elif "Batterie" in detected_classes[0]:
                                results = asyncio.run(process_ocr(image))  # Await the async function
                                img_data['serial'], img_data['date'] = asyncio.run(
                                    extract_important_info_batterie(results)
                                )
                            elif "Electrodes" in detected_classes[0]:
                                img_data['serial'], img_data['date'] = asyncio.run(
                                    extract_important_info_electrodes(image)
                                )
                            st.session_state.processed_data['images'].append(img_data)
                            st.success(f"Processed {detected_classes[0]} image: {uploaded_file.name}")
                        else:
                            st.warning(f"No classifications found for: {uploaded_file.name}")
                        os.unlink(temp_file_path)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")

if __name__ == "__main__":
    main()
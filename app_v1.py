import streamlit as st
import os
import re
import json
import requests
import numpy as np
import zipfile
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
from inference_sdk import InferenceHTTPClient
import easyocr
from pyzbar.pyzbar import decode
import io
import pdfplumber
from typing import Dict
import tempfile
import base64
import binascii

st.set_page_config(
    page_title="Inspecteur de dispositifs médicaux",
    layout="wide",
    page_icon="🏥"
)
# Set page config at the very beginning of the script
# Configuration
API_URL = "https://detect.roboflow.com"
MODEL_ID = "medical-object-classifier/3"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
 
# Initialisation des clients
client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=st.secrets["API_KEY"]
)
reader = easyocr.Reader(['en'], gpu=True)

# Initialisation de l'état de session
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {
        'RVD': {},
        'AEDG5': {},
        'AEDG3': {},
        'images': [],
        'files': [],
        'comparisons': {
            'rvd_vs_aed': {},
            'rvd_vs_images': {}
        }
    }

if 'dae_type' not in st.session_state:
    st.session_state.dae_type = 'G5'

def fix_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img._getexif().items())
        if exif.get(orientation) == 3:
            img = img.rotate(180, expand=True)
        elif exif.get(orientation) == 6:
            img = img.rotate(270, expand=True)
        elif exif.get(orientation) == 8:
            img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return img

@st.cache_data
def process_ocr(image):
    return reader.readtext(np.array(image))

def classify_image(image):
    """
    Placeholder for image classification logic.
    Ensure the image is properly processed before passing it to the model.
    """
    try:
        # If the image is a file path, read it
        if isinstance(image, str):
            with open(image, "rb") as image_file:
                image_data = image_file.read()
        # If the image is a file-like object, read it
        elif hasattr(image, "read"):
            image_data = image.read()
        # If the image is already bytes, use it directly
        elif isinstance(image, bytes):
            image_data = image
        else:
            raise ValueError("Unsupported image format")
        
        # Convert image data to Base64 (if needed)
        image_base64 = base64.b64encode(image_data).decode("utf-8")
        
        # Ensure proper padding
        padding = len(image_base64) % 4
        if padding:
            image_base64 += "=" * (4 - padding)
        
        # Simulate inference (replace with actual model call)
        result = {"predictions": [{"class": "Defibrillateur G5", "confidence": 0.9}]}
        return result
    
    except binascii.Error as e:
        raise ValueError(f"Base64 decoding error: {e}")
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Use `or ""` to handle pages without text
    return text


def extract_rvd_data(text: str) -> Dict[str, str]:
    """
    Extract data from text using specific keywords and pattern matching.
    
    Args:
        text (str): Input text to process
        
    Returns:
        Dict[str, str]: Dictionary of extracted key-value pairs
    """
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
    
    results = {}
    lines = text.splitlines()

    for keyword in keywords:
        value = "Non trouvé"
        
        # Special pattern for serial numbers
        if any(x in keyword.lower() for x in ["n° série", "numéro de série"]):
            pattern = re.compile(
                re.escape(keyword) + r"[\s:]*([A-Za-z0-9\-]+)(?=\s|$)", 
                re.IGNORECASE
            )
        # Special pattern for Code site
        elif keyword == "Code site":
            pattern = re.compile(
                r"Code site\s+([A-Z0-9]+)", 
                re.IGNORECASE
            )
        # Default pattern for other fields
        else:
            pattern = re.compile(
                re.escape(keyword) + r"[\s:]*([^\n]*)"
            )
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # Check if line starts with keyword (case-insensitive)
            if stripped_line.lower().startswith(keyword.lower()):
                match = pattern.search(stripped_line)
                if match:
                    value = match.group(1).strip()
                    # Special handling for serial numbers
                    if any(x in keyword.lower() for x in ["n° série", "numéro de série"]):
                        value = value.split()[0]  # Take first part if value is split
                else:
                    # If not found on same line, check next lines
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        # Skip verification/validation lines
                        if next_line and not any(x in next_line for x in ["Vérification", "Validation"]):
                            # Additional validation for specific fields
                            if ("date" in keyword.lower() and 
                                not re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', next_line)):
                                j += 1
                                continue
                            value = next_line
                            break
                        j += 1
                break
            
            # Special handling for Code site if not found at start of line
            elif keyword == "Code site":
                match = pattern.search(stripped_line)
                if match:
                    value = match.group(1)
                    break
        
        # Clean up extracted values
        if value != "Non trouvé":
            # Remove any trailing verification/validation text
            value = re.sub(r'\s*(?:Vérification|Validation).*$', '', value)
            # Clean up dates
            if "date" in keyword.lower() and re.search(r'\d{2}[/-]\d{2}[/-]\d{4}', value):
                value = re.search(r'\d{2}[/-]\d{2}[/-]\d{4}(?:\s+\d{2}:\d{2})?', value).group(0)
            # Clean up percentages
            elif "%" in keyword:
                value = re.sub(r'[^\d.]', '', value)
        
        results[keyword] = value

    return results

def extract_aed_g5_data(text):
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
    
    results = {}
    lines = text.split('\n')
    for i, line in enumerate(lines):
        for keyword in keywords:
            if keyword in line:
                value = lines[i+1].strip() if i+1 < len(lines) else ""
                results[keyword] = value
    return results

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

def extract_important_info_g5(results):
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

def parse_date(date_str):
    """Enhanced date parsing with better format handling"""
    formats = [
        '%d/%m/%Y %H:%M',       # Handles date with time (e.g., "08/01/2025 15:11")
        '%d/%m/%Y %H:%M:%S',    # Handles date with time including seconds (e.g., "08/01/2025 15:08:44")
        '%Y-%m-%d %H:%M',       # Alternative date format with time
        '%Y-%m-%d %H:%M:%S',    # Alternative date format with time including seconds
        '%d/%m/%Y',             # Handles date without time
        '%Y-%m-%d',             # Handles date without time (ISO format)
        '%m/%d/%Y',             # Handles date in US format
        '%d-%m-%Y',             # Handles date with hyphens
        '%Y/%m/%d',             # Handles date with slashes
        '%Y%m%d',               # Handles compact date format
        '%d %b %Y',             # Handles date with abbreviated month name (e.g., "08 Jan 2025")
        '%d %B %Y',             # Handles date with full month name (e.g., "08 January 2025")
    ]

    # Clean the date string: Remove unwanted characters and standardize separators
    clean_date = re.sub(r'[^\d:/ -]', '', str(date_str)).strip()  # Retain digits, :, /, -, and spaces
    clean_date = re.sub(r'/', '-', clean_date)  # Standardize to hyphens

    # Extract the date part by splitting at the first occurrence of time or extra text
    if ' ' in clean_date:
        date_part = clean_date.split(' ')[0]  # Take only the date part
    else:
        date_part = clean_date

    # Try parsing with cleaned date
    for fmt in formats:
        try:
            parsed_date = datetime.strptime(date_part, fmt).date()
            return parsed_date, None
        except ValueError:
            continue

    # If no format matches, return an error message
    return None, f"Unrecognized format: {clean_date}"

def normalize_serial(serial):
    return re.sub(r'[^A-Z0-9]', '', str(serial).upper())

def compare_rvd_aed():
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

def compare_rvd_images():
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
# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        .card {
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            background: white;
        }
        .header {
            background-color: #006699;
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
        .success-box {
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-color: #ffeeba;
            color: #856404;
            padding: 1rem;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)
def main():
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {'RVD': {}, 'images': []}
    
    # Header Section
    with st.container():
        col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://www.locacoeur.com/wp-content/uploads/2020/04/Locacoeur_Logo.png", width=100)
    with col2:
        st.title("Système d'inspection des dispositifs médicaux")
        st.markdown("*Solution intégrée pour la gestion et l'inspection des DAE*")
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres de configuration")
        st.markdown("---")
        
        # Device Configuration
        st.subheader("📱 Configuration du dispositif")
        st.session_state.dae_type = st.radio(
            "Type d'AED",
            ("G5", "G3"),
            index=0,
            help="Sélectionnez le type de dispositif à inspecter"
        )
        
        # Processing Options
        st.subheader("🔧 Options de traitement")
        st.session_state.enable_ocr = st.checkbox(
            "Activer l'OCR",
            True,
            help="Active la reconnaissance de texte sur les images"
        )
        st.session_state.auto_classify = st.checkbox(
            "Classification automatique",
            True,
            help="Active la classification automatique des documents"
        )
        
        # Help Section
        st.markdown("---")
        st.markdown("#### 🔍 Guide d'utilisation")
        with st.expander("Comment utiliser l'application ?", expanded=False):
            st.markdown("""
                1. **Préparation** 📋
                   - Vérifiez que vos documents sont au format requis
                   - Assurez-vous que les images sont nettes
                
                2. **Téléversement** 📤
                   - Glissez-déposez vos fichiers
                   - Attendez le traitement complet
                
                3. **Vérification** ✅
                   - Examinez les données extraites
                   - Validez les résultats
                
                4. **Export** 📥
                   - Choisissez le format d'export
                   - Téléchargez vos résultats
            """)
        
        st.markdown("---")
        st.caption("Développé par Locacoeur • v2.1.0")
    
    
    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["📂 Téléversement", "📊 Résultats", "📤 Export"])
    
    with tab1:
        # File Upload Section
        st.markdown("### 📂 Téléversement des documents")
        with st.expander("Zone de dépôt des fichiers", expanded=True):
            uploaded_files = st.file_uploader(
                "Glissez et déposez vos fichiers ici",
                type=ALLOWED_EXTENSIONS,
                accept_multiple_files=True,
                help="Formats acceptés : PDF, JPG, PNG",
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                progress_bar = st.progress(0)
                processing_status = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # File processing logic...
                    if uploaded_file.type == "application/pdf":
                        text = extract_text_from_pdf(uploaded_file)
                        if 'rapport de vérification' in uploaded_file.name.lower():
                            st.session_state.processed_data['RVD'] = extract_rvd_data(text)
                        elif 'aed' in uploaded_file.name.lower():
                            if st.session_state.dae_type == "G5":
                                st.session_state.processed_data['AEDG5'] = extract_aed_g5_data(text)
                            else:
                                st.session_state.processed_data['AEDG3'] = extract_aed_g3_data(text)
                    else:
                        image = Image.open(uploaded_file)
                        try:
                            # Pass the file directly to classify_image
                            result = classify_image(uploaded_file)
                            detected_classes = [pred['class'] for pred in result.get('predictions', []) if pred['confidence'] > 0.5]
                            if detected_classes:
                                img_data = {
                                    'type': detected_classes[0],
                                    'serial': None,
                                    'date': None,
                                    'image': image
                                }
                                st.session_state.processed_data['images'].append(img_data)
                        except Exception as e:
                            st.error(f"Erreur lors du traitement de {uploaded_file.name} : {str(e)}")
                
                processing_status.success(f"Traitement terminé - {len(uploaded_files)} fichiers analysés")

                progress_bar.empty()
                
    with tab2:
        # Results Display
        st.markdown("### 📋 Synthèse des résultats")
        
        # Data Summary Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            with st.container():
                st.markdown("<div class='card'>"
                           "<h4>📄 Documents traités</h4>"
                           f"<h2>{len(uploaded_files)}</h2>"
                           "</div>", unsafe_allow_html=True)
        
        # Processed Data Display
        st.markdown("#### 🔍 Données extraites")
        with st.expander("Détails des rapports", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Rapport de vérification (RVD)")
                if st.session_state.processed_data['RVD']:
                    rvd_data = st.session_state.processed_data['RVD']
                    st.markdown(f"""
                    - **Code site:** `{rvd_data.get('Code site', 'N/A')}`
                    - **Date inspection:** `{rvd_data.get('Date', 'N/A')}`
                    - **Statut:** `{rvd_data.get('Statut', 'N/A')}`
                    """)
                else:
                    st.markdown("<div class='warning-box'>Aucune donnée RVD disponible</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"##### Rapport AED {st.session_state.dae_type}")
                aed_type = f'AEDG{st.session_state.dae_type[-1]}'
                aed_data = st.session_state.processed_data.get(aed_type, {})
                if aed_data:
                    st.markdown(f"""
                    - **Modèle:** `{aed_data.get('Modèle', 'N/A')}`
                    - **Numéro série:** `{aed_data.get('Série', 'N/A')}`
                    - **Dernière maintenance:** `{aed_data.get('Maintenance', 'N/A')}`
                    """)
                else:
                    st.markdown("<div class='warning-box'>Aucune donnée AED disponible</div>", unsafe_allow_html=True)
        
        # Image Analysis Results
        if st.session_state.processed_data['images']:
            st.markdown("#### 📸 Analyse des images")
            cols = st.columns(3)
            for idx, img_data in enumerate(st.session_state.processed_data['images']):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.image(img_data['image'], use_container_width=True)
                        st.markdown(f"""
                        **Type:** `{img_data['type']}`  
                        **Série:** `{img_data.get('serial', 'N/A')}`  
                        **Date:** `{img_data.get('date', 'N/A')}`
                        """)
                        st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        # Export Section
        st.markdown("### 📤 Export des résultats")
        
        # Export Configuration
        with st.form("export_config"):
            col1, col2 = st.columns(2)
            with col1:
                export_format = st.selectbox("Format d'export", ["ZIP", "PDF", "CSV"])
            with col2:
                include_images = st.checkbox("Inclure les images", True)
            
            if st.form_submit_button("Générer l'export"):
                # Export processing logic...
                st.success("Package d'export généré avec succès")
        
        # Download Section
        if os.path.exists('export.zip'):
            with open("export.zip", "rb") as f:
                st.download_button(
                    label="📥 Télécharger l'export",
                    data=f,
                    file_name=f"Inspection_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    help="Cliquez pour télécharger le package complet"
                )

if __name__ == "__main__":
    main()
import streamlit as st
import os
import re
import json
import requests
import numpy as np
import zipfile
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
from PyPDF2 import PdfReader
from inference_sdk import InferenceHTTPClient
import easyocr
from pyzbar.pyzbar import decode
import io
import tempfile

# Configuration
API_URL = "https://detect.roboflow.com"
MODEL_ID = "medical-object-classifier/3"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# Initialize clients
client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=st.secrets["API_KEY"]
)
reader = easyocr.Reader(['en'], gpu=False)

# Session state initialization
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

def classify_image(image_path):
    return client.infer(image_path, model_id=MODEL_ID)

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

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
    
    results = {}
    for keyword in keywords:
        # Special handling for battery serial number
        if keyword == "N° série nouvelle batterie":
            # Use more precise pattern to capture serial number
            pattern = re.compile(
                re.escape(keyword) + 
                r"[\s:]*([A-Za-z0-9\-]+)(?=\s|$)",
                re.IGNORECASE
            )
        else:
            pattern = re.compile(re.escape(keyword) + r"[\s:]*([^\n]*)")
        
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            # Additional cleanup for battery serial
            if keyword == "N° série nouvelle batterie":
                value = value.split()[0]  # Take first part if space separated
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
    formats = [
        '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', 
        '%d-%m-%Y', '%Y/%m/%d', '%Y%m%d',
        '%d %b %Y', '%d %B %Y'
    ]
    
    if not date_str or str(date_str).lower() == 'nan':
        return None, "No date provided"
    
    clean_date = str(date_str).split()[0].strip()
    
    for fmt in formats:
        try:
            return datetime.strptime(clean_date, fmt).date(), None
        except ValueError:
            continue
    return None, f"Unrecognized format: {clean_date}"

def normalize_serial(serial):
    return re.sub(r'[^A-Z0-9]', '', str(serial).upper())

def compare_rvd_aed():
    try:
        results = {}
        aed_type = f'AEDG{st.session_state.dae_type[-1]}'
        
        if not st.session_state.processed_data.get('RVD'):
            st.error("Missing RVD data for comparison")
            return {}
        if not st.session_state.processed_data.get(aed_type):
            st.error(f"Missing {aed_type} data for comparison")
            return {}

        rvd = st.session_state.processed_data['RVD']
        aed = st.session_state.processed_data[aed_type]
        
        # Serial Number Comparison
        aed_key = 'N° série DAE' if st.session_state.dae_type == 'G5' else 'Série DSA'
        results['serial'] = {
            'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
            'aed': aed.get(aed_key, 'N/A'),
            'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) == 
                    normalize_serial(aed.get(aed_key, ''))
        }
        
        # Date Comparison
        rvd_date, rvd_err = parse_date(rvd.get('Date-Heure rapport vérification défibrillateur', ''))
        aed_date_key = 'Date / Heure:' if st.session_state.dae_type == 'G5' else 'Date de mise en service'
        aed_date, aed_err = parse_date(aed.get(aed_date_key, ''))
        results['date'] = {
            'rvd': rvd.get('Date-Heure rapport vérification défibrillateur', 'N/A'),
            'aed': aed.get(aed_date_key, 'N/A'),
            'match': rvd_date == aed_date if not (rvd_err or aed_err) else False,
            'errors': [e for e in [rvd_err, aed_err] if e]
        }
        
        # Battery Comparison
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
                'error': f"Invalid battery data: {str(e)}",
                'match': False
            }
        
        st.session_state.processed_data['comparisons']['rvd_vs_aed'] = results
        return results
    
    except KeyError as e:
        st.error(f"Missing data key: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        return {}

def compare_rvd_images():
    try:
        results = {}
        if not st.session_state.processed_data.get('RVD'):
            st.error("Missing RVD data for comparison")
            return {}
            
        rvd = st.session_state.processed_data['RVD']
        images = st.session_state.processed_data['images']
        
        # Battery Comparison
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
        
        # Electrode Comparison
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
        st.error(f"Missing data key: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Comparison error: {str(e)}")
        return {}

def display_comparison(title, comparison):
    if not comparison:
        st.warning("No comparison data available")
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
                    text = extract_text_from_pdf(uploaded_file)
                    
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
                        
                        result = classify_image(temp_file_path)
                        detected_classes = [pred['class'] for pred in result.get('predictions', []) if pred['confidence'] > 0.5]
                        
                        if detected_classes:
                            img_data = {
                                'type': detected_classes[0],
                                'serial': None,
                                'date': None,
                                'image': image
                            }
                            
                            if "Defibrillateur" in detected_classes[0]:
                                results = process_ocr(image)
                                if "G3" in detected_classes[0]:
                                    img_data['serial'], img_data['date'] = extract_important_info_g3(results)
                                else:
                                    img_data['serial'], img_data['date'] = extract_important_info_g5(results)
                            
                            elif "Batterie" in detected_classes[0]:
                                results = process_ocr(image)
                                img_data['serial'], img_data['date'] = extract_important_info_batterie(results)
                            
                            elif "Electrodes" in detected_classes[0]:
                                img_data['serial'], img_data['date'] = extract_important_info_electrodes(image)
                            
                            st.session_state.processed_data['images'].append(img_data)
                            st.success(f"Processed {detected_classes[0]} image: {uploaded_file.name}")
                        
                        else:
                            st.warning(f"No classifications found for: {uploaded_file.name}")
                        os.unlink(temp_file_path)
                    
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Processed Data Display
    with st.expander("Processed Data", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RVD Data")
            st.json(st.session_state.processed_data['RVD'], expanded=False)
        
        with col2:
            st.subheader(f"AED {st.session_state.dae_type} Data")
            aed_type = f'AEDG{st.session_state.dae_type[-1]}'
            aed_data = st.session_state.processed_data.get(aed_type, {})
            st.json(aed_data if aed_data else {"status": "No AED data found"}, expanded=False)
    
    # Image Results Display
    if st.session_state.processed_data['images']:
        with st.expander("Image Analysis Results", expanded=True):
            cols = st.columns(3)
            for idx, img_data in enumerate(st.session_state.processed_data['images']):
                with cols[idx % 3]:
                    st.image(img_data['image'], use_column_width=True)
                    st.markdown(f"""
                    **Type:** {img_data['type']}  
                    **Serial:** {img_data.get('serial', 'N/A')}  
                    **Date:** {img_data.get('date', 'N/A')}
                    """)
    
    # Enhanced Comparison Section
    with st.expander("Document Comparison", expanded=True):
        if st.button("Run Full Analysis"):
            try:
                aed_results = compare_rvd_aed()
                image_results = compare_rvd_images()
                
                display_comparison("RVD vs AED Report Comparison", aed_results)
                display_comparison("RVD vs Image Data Comparison", image_results)
                
                all_matches = all(
                    item.get('match', False)
                    for comp in [aed_results, image_results] 
                    for item in comp.values() 
                )
                
                if all_matches:
                    st.success("All checks passed! Device is compliant.")
                else:
                    failed = [
                        k for comp in [aed_results, image_results] 
                        for k, v in comp.items() 
                        if not v.get('match', True)
                    ]
                    st.error(f"Validation failed for: {', '.join(failed)}")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    # File Management Section
    with st.expander("File Export", expanded=True):
        if st.button("Generate Export Package"):
            if not st.session_state.processed_data.get('RVD'):
                st.warning("No RVD data available for naming")
                return
            
            code_site = st.session_state.processed_data['RVD'].get('Code site', 'UNKNOWN')
            date_str = datetime.now().strftime("%Y%m%d")
            
            try:
                with zipfile.ZipFile('export.zip', 'w') as zipf:
                    with zipf.open('processed_data.json', 'w') as f:
                        f.write(json.dumps(st.session_state.processed_data, indent=2).encode())
                    
                    for uploaded_file in uploaded_files:
                        original_bytes = uploaded_file.getvalue()
                        
                        if uploaded_file.type == "application/pdf":
                            if 'rapport de vérification' in uploaded_file.name.lower():
                                new_name = f"RVD_{code_site}_{date_str}.pdf"
                            else:
                                new_name = f"AED_{st.session_state.dae_type}_{code_site}_{date_str}.pdf"
                        else:
                            new_name = f"IMAGE_{code_site}_{date_str}_{uploaded_file.name}"
                        
                        zipf.writestr(new_name, original_bytes)
                
                with open("export.zip", "rb") as f:
                    st.download_button(
                        label="Download Export Package",
                        data=f,
                        file_name=f"Inspection_{code_site}_{date_str}.zip",
                        mime="application/zip"
                    )
            
            except Exception as e:
                st.error(f"Error creating export package: {str(e)}")

if __name__ == "__main__":
    main()
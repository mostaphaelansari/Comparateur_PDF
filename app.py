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
import pdfplumber

import tempfile

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

def classify_image(image_path):
    return client.infer(image_path, model_id=MODEL_ID)

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""  # Use `or ""` to handle pages without text
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
        # Gestion spéciale pour le numéro de série de la batterie
        if keyword == "N° série nouvelle batterie":
            # Utilisation d'un motif plus précis pour capturer le numéro de série
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
            # Nettoyage supplémentaire pour le numéro de série de la batterie
            if keyword == "N° série nouvelle batterie":
                value = value.split()[0]  # Prendre la première partie si séparée par un espace
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

def main():
    st.set_page_config(page_title="Inspecteur de dispositifs médicaux", layout="wide")
    
    # Section d'en-tête
    st.title("Système d'inspection des dispositifs médicaux")
    st.markdown("---")
    
    # Barre latérale de configuration
    with st.sidebar:
        st.header("Configuration")
        st.session_state.dae_type = st.radio("Type d'AED", ("G5", "G3"), index=0)
        st.session_state.enable_ocr = st.checkbox("Activer le traitement OCR", True)
        st.markdown("---")
        st.write("Développé par Locacoeur]")
    
    # Section de téléversement des fichiers
    with st.expander("Téléverser des documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Glissez et déposez des fichiers ici",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help="Téléverser des rapports PDF et des images de dispositifs"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                    if 'rapport de vérification' in uploaded_file.name.lower():
                        st.session_state.processed_data['RVD'] = extract_rvd_data(text)
                        st.success(f"RVD traité : {uploaded_file.name}")
                    elif 'aed' in uploaded_file.name.lower():
                        if st.session_state.dae_type == "G5":
                            st.session_state.processed_data['AEDG5'] = extract_aed_g5_data(text)
                        else:
                            st.session_state.processed_data['AEDG3'] = extract_aed_g3_data(text)
                        st.success(f"Rapport AED {st.session_state.dae_type} traité : {uploaded_file.name}")
                
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
                            st.success(f"Image {detected_classes[0]} traitée : {uploaded_file.name}")
                        
                        else:
                            st.warning(f"Aucune classification trouvée pour : {uploaded_file.name}")
                        os.unlink(temp_file_path)
                    
                    except Exception as e:
                        st.error(f"Erreur lors du traitement de {uploaded_file.name} : {str(e)}")
    
    # Affichage des données traitées
    with st.expander("Données traitées", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Données RVD")
            st.json(st.session_state.processed_data['RVD'], expanded=False)
        
        with col2:
            st.subheader(f"Données AED {st.session_state.dae_type}")
            aed_type = f'AEDG{st.session_state.dae_type[-1]}'
            aed_data = st.session_state.processed_data.get(aed_type, {})
            st.json(aed_data if aed_data else {"status": "Aucune donnée AED trouvée"}, expanded=False)
    
    # Affichage des résultats d'analyse d'images
    if st.session_state.processed_data['images']:
        with st.expander("Résultats d'analyse d'images", expanded=True):
            cols = st.columns(3)
            for idx, img_data in enumerate(st.session_state.processed_data['images']):
                with cols[idx % 3]:
                    st.image(img_data['image'], use_column_width=True)
                    st.markdown(f"""
                    **Type:** {img_data['type']}  
                    **Numéro de série:** {img_data.get('serial', 'N/A')}  
                    **Date:** {img_data.get('date', 'N/A')}
                    """)
    
    # Section de comparaison améliorée
    with st.expander("Comparaison des documents", expanded=True):
        if st.button("Lancer l'analyse complète"):
            try:
                aed_results = compare_rvd_aed()
                image_results = compare_rvd_images()
                
                display_comparison("Comparaison RVD vs Rapport AED", aed_results)
                display_comparison("Comparaison RVD vs Données d'images", image_results)
                
                all_matches = all(
                    item.get('match', False)
                    for comp in [aed_results, image_results] 
                    for item in comp.values() 
                )
                
                if all_matches:
                    st.success("Tous les contrôles sont réussis ! Le dispositif est conforme.")
                else:
                    failed = [
                        k for comp in [aed_results, image_results] 
                        for k, v in comp.items() 
                        if not v.get('match', True)
                    ]
                    st.error(f"Échec de validation pour : {', '.join(failed)}")
            
            except Exception as e:
                st.error(f"Échec de l'analyse : {str(e)}")
    
    # Section de gestion des fichiers
    with st.expander("Exportation des fichiers", expanded=True):
        if st.button("Générer un package d'export"):
            if not st.session_state.processed_data.get('RVD'):
                st.warning("Aucune donnée RVD disponible pour le nommage")
                return
            
            code_site = st.session_state.processed_data['RVD'].get('Code site', 'INCONNU')
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
                        label="Télécharger le package d'export",
                        data=f,
                        file_name=f"Inspection_{code_site}_{date_str}.zip",
                        mime="application/zip"
                    )
            
            except Exception as e:
                st.error(f"Erreur lors de la création du package d'export : {str(e)}")

if __name__ == "__main__":
    main()
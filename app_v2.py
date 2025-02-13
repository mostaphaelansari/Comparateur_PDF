import streamlit as st
from PIL import Image
from datetime import datetime
import os

# Constants
ALLOWED_EXTENSIONS = ['pdf', 'png', 'jpg', 'jpeg']
THEME_COLORS = {
    'primary': '#006699',
    'success': '#4CAF50',
    'warning': '#FFA500',
    'error': '#DC3545',
    'info': '#17a2b8'
}

def set_page_config():
    st.set_page_config(
        page_title="Système d'inspection des dispositifs médicaux",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    st.markdown("""
        <style>
            /* Main container styling */
            .main {
                background-color: #f8f9fa;
                padding: 1rem;
            }
            
            /* Header styling */
            .header-container {
                background: linear-gradient(90deg, #006699 0%, #004466 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            /* Card styling */
            .card {
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                border-left: 5px solid #006699;
            }
            
            /* Stats card styling */
            .stat-card {
                background: white;
                padding: 1.5rem;
                border-radius: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.2s;
            }
            .stat-card:hover {
                transform: translateY(-5px);
            }
            
            /* Button styling */
            .stButton>button {
                background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 10px;
                font-weight: 500;
                transition: all 0.3s;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            
            /* File uploader styling */
            .uploadedFile {
                border: 2px dashed #006699;
                border-radius: 10px;
                padding: 1rem;
                background: rgba(0,102,153,0.05);
            }
            
            /* Alert box styling */
            .success-box {
                background-color: #d4edda;
                border-left: 5px solid #4CAF50;
                color: #155724;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            .warning-box {
                background-color: #fff3cd;
                border-left: 5px solid #ffc107;
                color: #856404;
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
            }
            
            /* Sidebar styling */
            .sidebar .sidebar-content {
                background-color: #ffffff;
                padding: 1rem;
            }
            
            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 1rem 2rem;
                background-color: #f8f9fa;
                border-radius: 10px 10px 0 0;
            }
            .stTabs [data-baseweb="tab-panel"] {
                padding: 1rem;
                border-radius: 0 0 10px 10px;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Progress bar styling */
            .stProgress > div > div {
                background-color: #4CAF50;
            }
        </style>
    """, unsafe_allow_html=True)

def render_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("https://www.locacoeur.com/wp-content/uploads/2020/04/Locacoeur_Logo.png", width=100)
    with col2:
        st.title("Système d'inspection des dispositifs médicaux")
        st.markdown("*Solution intégrée pour la gestion et l'inspection des DAE*")

def render_sidebar():
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
        if st.button("📋 Signaler un problème"):
            st.info("Le formulaire de signalement va s'ouvrir dans un nouvel onglet")

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return
    
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_placeholder.info(f"Traitement de {uploaded_file.name}...")
        
        try:
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
                process_pdf_content(text, uploaded_file.name)
            else:
                process_image_file(uploaded_file)
                
            status_placeholder.success(f"✅ {uploaded_file.name} traité avec succès")
        except Exception as e:
            status_placeholder.error(f"❌ Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
    
    progress_bar.empty()
    status_placeholder.empty()
    st.success(f"✨ Traitement terminé - {len(uploaded_files)} fichiers analysés")

def render_upload_tab():
    st.markdown("### 📤 Téléversement des documents")
    
    with st.expander("📋 Instructions de téléversement", expanded=False):
        st.markdown("""
            - Formats acceptés : PDF, JPG, PNG
            - Taille maximale par fichier : 10 MB
            - Les documents doivent être lisibles et non endommagés
        """)
    
    uploaded_files = st.file_uploader(
        "Glissez et déposez vos fichiers ici",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        help="Formats acceptés : PDF, JPG, PNG",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)

def render_results_tab():
    st.markdown("### 📊 Synthèse des résultats")
    
    # Summary Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container():
            st.markdown("""
                <div class="stat-card">
                    <h4>📄 Documents traités</h4>
                    <h2>0</h2>
                </div>
            """, unsafe_allow_html=True)
    with col2:
        with st.container():
            st.markdown("""
                <div class="stat-card">
                    <h4>✅ Validés</h4>
                    <h2>0</h2>
                </div>
            """, unsafe_allow_html=True)
    with col3:
        with st.container():
            st.markdown("""
                <div class="stat-card">
                    <h4>⚠️ En attente</h4>
                    <h2>0</h2>
                </div>
            """, unsafe_allow_html=True)
    
    # Detailed Results
    st.markdown("#### 🔍 Données extraites")
    
    tab1, tab2, tab3 = st.tabs(["📋 Rapports", "📸 Images", "📊 Analyses"])
    
    with tab1:
        render_reports_subtab()
    
    with tab2:
        render_images_subtab()
    
    with tab3:
        render_analysis_subtab()

def render_reports_subtab():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Rapport de vérification (RVD)")
        if st.session_state.processed_data.get('RVD'):
            rvd_data = st.session_state.processed_data['RVD']
            st.markdown("""
                <div class="card">
                    <h6>Informations principales</h6>
                    <ul>
                        <li><strong>Code site:</strong> {}</li>
                        <li><strong>Date inspection:</strong> {}</li>
                        <li><strong>Statut:</strong> {}</li>
                    </ul>
                </div>
            """.format(
                rvd_data.get('Code site', 'N/A'),
                rvd_data.get('Date', 'N/A'),
                rvd_data.get('Statut', 'N/A')
            ), unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Aucune donnée RVD disponible</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"##### Rapport AED {st.session_state.dae_type}")
        aed_type = f'AEDG{st.session_state.dae_type[-1]}'
        if aed_data := st.session_state.processed_data.get(aed_type):
            st.markdown("""
                <div class="card">
                    <h6>Caractéristiques du dispositif</h6>
                    <ul>
                        <li><strong>Modèle:</strong> {}</li>
                        <li><strong>Numéro série:</strong> {}</li>
                        <li><strong>Dernière maintenance:</strong> {}</li>
                    </ul>
                </div>
            """.format(
                aed_data.get('Modèle', 'N/A'),
                aed_data.get('Série', 'N/A'),
                aed_data.get('Maintenance', 'N/A')
            ), unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">Aucune donnée AED disponible</div>', unsafe_allow_html=True)

def render_images_subtab():
    if images := st.session_state.processed_data.get('images', []):
        cols = st.columns(3)
        for idx, img_data in enumerate(images):
            with cols[idx % 3]:
                st.markdown("""
                    <div class="card">
                        <img src="{}" style="width: 100%; border-radius: 8px;">
                        <div style="margin-top: 1rem;">
                            <p><strong>Type:</strong> {}</p>
                            <p><strong>Série:</strong> {}</p>
                            <p><strong>Date:</strong> {}</p>
                        </div>
                    </div>
                """.format(
                    img_data['image'],
                    img_data.get('type', 'N/A'),
                    img_data.get('serial', 'N/A'),
                    img_data.get('date', 'N/A')
                ), unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">Aucune image disponible</div>', unsafe_allow_html=True)

def render_analysis_subtab():
    st.markdown("##### 📊 Analyse comparative")
    if st.session_state.processed_data.get('RVD') and st.session_state.processed_data.get(f'AEDG{st.session_state.dae_type[-1]}'):
        st.markdown('<div class="success-box">Analyse comparative disponible</div>', unsafe_allow_html=True)
        # Add your analysis visualization here
    else:
        st.markdown('<div class="warning-box">Données insuffisantes pour l\'analyse comparative</div>', unsafe_allow_html=True)

def render_export_tab():
    st.markdown("### 📤 Export des résultats")
    
    with st.form("export_config"):
        st.markdown("#### Configuration de l'export")
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox(
                "Format d'export",
                ["ZIP", "PDF", "CSV"],
                help="Sélectionnez le format de sortie souhaité"
            )
        with col2:
            include_images = st.checkbox(
                "Inclure les images",
                True,
                help="Inclure les images dans l'export"
            )
            include_analysis = st.checkbox(
                "Inclure l'analyse comparative",
                True,
                help="Inclure les résultats de l'analyse comparative"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            report_name = st.text_input(
                "Nom du rapport",
                value=f"Inspection_{datetime.now().strftime('%Y%m%d')}",
                help="Personnalisez le nom du fichier d'export"
            )
        with col2:
            compression = st.select_slider(
                "Niveau de compression",
                options=["Aucune", "Normale", "Maximum"],
                value="Normale",
                help="Définissez le niveau de compression pour l'export"
            )
        
        st.markdown("#### Options avancées")
        with st.expander("⚙️ Configuration avancée", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                include_metadata = st.checkbox(
                    "Inclure les métadonnées",
                    True,
                    help="Ajouter les métadonnées des fichiers dans l'export"
                )
                include_raw_data = st.checkbox(
                    "Inclure les données brutes",
                    False,
                    help="Ajouter les données non traitées dans l'export"
                )
            with col2:
                export_format_options = {
                    "ZIP": st.checkbox("Créer une archive structurée", True),
                    "PDF": st.checkbox("Ajouter une table des matières", True),
                    "CSV": st.checkbox("Inclure les en-têtes", True)
                }
        
        submitted = st.form_submit_button("Générer l'export", use_container_width=True)
        if submitted:
            try:
                with st.spinner("Génération de l'export en cours..."):
                    # Simulate export processing
                    st.success("✅ Export généré avec succès!")
                    
                    # Generate dummy export file for demonstration
                    if os.path.exists('export.zip'):
                        with open("export.zip", "rb") as f:
                            st.download_button(
                                label="📥 Télécharger l'export",
                                data=f,
                                file_name=f"{report_name}.zip",
                                mime="application/zip",
                                help="Cliquez pour télécharger le package complet",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Erreur lors de la génération de l'export: {str(e)}")

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {
            'RVD': {},
            'images': [],
            'AEDG5': {},
            'AEDG3': {}
        }
    if 'export_history' not in st.session_state:
        st.session_state.export_history = []

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    # Implement PDF text extraction logic here
    return "Sample extracted text"

def process_pdf_content(text, filename):
    """Process extracted PDF content"""
    if 'rapport de vérification' in filename.lower():
        st.session_state.processed_data['RVD'] = extract_rvd_data(text)
    elif 'aed' in filename.lower():
        if st.session_state.dae_type == "G5":
            st.session_state.processed_data['AEDG5'] = extract_aed_g5_data(text)
        else:
            st.session_state.processed_data['AEDG3'] = extract_aed_g3_data(text)

def extract_rvd_data(text):
    """Extract data from RVD report"""
    # Implement RVD data extraction logic here
    return {
        'Code site': 'SITE001',
        'Date': '2024-02-13',
        'Statut': 'Conforme'
    }

def extract_aed_g5_data(text):
    """Extract data from AED G5 report"""
    # Implement AED G5 data extraction logic here
    return {
        'Modèle': 'G5-001',
        'Série': 'SN123456',
        'Maintenance': '2024-01-15'
    }

def extract_aed_g3_data(text):
    """Extract data from AED G3 report"""
    # Implement AED G3 data extraction logic here
    return {
        'Modèle': 'G3-001',
        'Série': 'SN789012',
        'Maintenance': '2024-01-20'
    }

def process_image_file(file):
    """Process uploaded image file"""
    image = Image.open(file)
    # Implement image processing logic here
    if st.session_state.enable_ocr:
        # Implement OCR logic here
        pass
    
    # Add processed image data to session state
    img_data = {
        'type': 'AED',
        'serial': 'SN123456',
        'date': '2024-02-13',
        'image': image
    }
    st.session_state.processed_data['images'].append(img_data)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Set page configuration
    set_page_config()
    
    # Apply custom CSS
    apply_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Téléversement", "📊 Résultats", "📤 Export"])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_results_tab()
    
    with tab3:
        render_export_tab()

if __name__ == "__main__":
    main()
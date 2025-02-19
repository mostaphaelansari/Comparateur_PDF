def compare_rvd_images():
    if  {Changement batterie} =="Non" && {Changement électrodes pédiatriques} =="Non" && {Changement électrodes adultes} =="Non"  :

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
                    'rvd': rvd.get('Numéro de série Batterie', 'N/A'),
                    'image': battery_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série Batterie', '')) ==
                            normalize_serial(battery_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("Numéro de série ELECTRODES ADULTES", 'N/A'),
                    'image': electrode_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série ELECTRODES ADULTES', '')) ==
                            normalize_serial(electrode_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date de péremption ELECTRODES ADULTES', ''))
                img_date, img_err = parse_date(electrode_data.get('date', ''))
                results['electrode_date'] = {
                    'rvd': rvd.get('Date de péremption ELECTRODES ADULTES', 'N/A'),
                    'image': electrode_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison du défibrillateur
            defibrillator_data = next((i for i in images if i['type'] == 'Defibrillateur G5'), None)
            if defibrillator_data:
                results['defibrillator_serial'] = {
                    'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) ==
                            normalize_serial(defibrillator_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication DEFIBRILLATEUR', ''))
                img_date, img_err = parse_date(defibrillator_data.get('date', ''))
                results['defibrillator_date'] = {
                    'rvd': rvd.get('Date fabrication DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('date', 'N/A'),
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

    if  {Changement batterie} =="Oui" && {Changement électrodes pédiatriques} =="Non" && {Changement électrodes adultes} =="Non"  :
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
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("Numéro de série ELECTRODES ADULTES", 'N/A'),
                    'image': electrode_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série ELECTRODES ADULTES', '')) ==
                            normalize_serial(electrode_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date de péremption ELECTRODES ADULTES', ''))
                img_date, img_err = parse_date(electrode_data.get('date', ''))
                results['electrode_date'] = {
                    'rvd': rvd.get('Date de péremption ELECTRODES ADULTES', 'N/A'),
                    'image': electrode_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison du défibrillateur
            defibrillator_data = next((i for i in images if i['type'] == 'Defibrillateur G5'), None)
            if defibrillator_data:
                results['defibrillator_serial'] = {
                    'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) ==
                            normalize_serial(defibrillator_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication DEFIBRILLATEUR', ''))
                img_date, img_err = parse_date(defibrillator_data.get('date', ''))
                results['defibrillator_date'] = {
                    'rvd': rvd.get('Date fabrication DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('date', 'N/A'),
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
    
    if  {Changement batterie} =="Non" && {Changement électrodes pédiatriques} =="Oui" && {Changement électrodes adultes} =="Non"  :

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
                    'rvd': rvd.get('Numéro de série Batterie', 'N/A'),
                    'image': battery_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série Batterie', '')) ==
                            normalize_serial(battery_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("N° série nouvelles électrodes pédiatriques", 'N/A'),
                    'image': electrode_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('N° série nouvelles électrodes pédiatriques', '')) ==
                            normalize_serial(electrode_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', ''))
                img_date, img_err = parse_date(electrode_data.get('date', ''))
                results['electrode_date'] = {
                    'rvd': rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', 'N/A'),
                    'image': electrode_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison du défibrillateur
            defibrillator_data = next((i for i in images if i['type'] == 'Defibrillateur G5'), None)
            if defibrillator_data:
                results['defibrillator_serial'] = {
                    'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) ==
                            normalize_serial(defibrillator_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication DEFIBRILLATEUR', ''))
                img_date, img_err = parse_date(defibrillator_data.get('date', ''))
                results['defibrillator_date'] = {
                    'rvd': rvd.get('Date fabrication DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('date', 'N/A'),
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

    if  {Changement batterie} =="Non" && {Changement électrodes pédiatriques} =="Non" && {Changement électrodes adultes} =="Oui"  :

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
                    'rvd': rvd.get('Numéro de série Batterie', 'N/A'),
                    'image': battery_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série Batterie', '')) ==
                            normalize_serial(battery_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("N° série nouvelles électrodes", 'N/A'),
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
            
            # Comparaison du défibrillateur
            defibrillator_data = next((i for i in images if i['type'] == 'Defibrillateur G5'), None)
            if defibrillator_data:
                results['defibrillator_serial'] = {
                    'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) ==
                            normalize_serial(defibrillator_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication DEFIBRILLATEUR', ''))
                img_date, img_err = parse_date(defibrillator_data.get('date', ''))
                results['defibrillator_date'] = {
                    'rvd': rvd.get('Date fabrication DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('date', 'N/A'),
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

    if  {Changement batterie} =="Oui" && {Changement électrodes pédiatriques} =="Non" && {Changement électrodes adultes} =="Oui"  :
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
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("N° série nouvelles électrodes", 'N/A'),
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
            
            # Comparaison du défibrillateur
            defibrillator_data = next((i for i in images if i['type'] == 'Defibrillateur G5'), None)
            if defibrillator_data:
                results['defibrillator_serial'] = {
                    'rvd': rvd.get('Numéro de série DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('Numéro de série DEFIBRILLATEUR', '')) ==
                            normalize_serial(defibrillator_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication DEFIBRILLATEUR', ''))
                img_date, img_err = parse_date(defibrillator_data.get('date', ''))
                results['defibrillator_date'] = {
                    'rvd': rvd.get('Date fabrication DEFIBRILLATEUR', 'N/A'),
                    'image': defibrillator_data.get('date', 'N/A'),
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

    if  {Changement batterie} =="Oui" && {Changement électrodes pédiatriques} =="Oui" && {Changement électrodes adultes} =="Non"  :
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
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("N° série nouvelles électrodes pédiatriques", 'N/A'),
                    'image': electrode_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('N° série nouvelles électrodes pédiatriques', '')) ==
                            normalize_serial(electrode_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', ''))
                img_date, img_err = parse_date(electrode_data.get('date', ''))
                results['electrode_date'] = {
                    'rvd': rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', 'N/A'),
                    'image': electrode_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
    if {Changement batterie} =="Oui" && {Changement électrodes pédiatriques} =="Oui" && {Changement électrodes adultes} =="Oui" :
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
                
                rvd_date, rvd_err = parse_date(rvd.get('Date fabrication BATTERIE', ''))
                img_date, img_err = parse_date(battery_data.get('date', ''))
                results['battery_date'] = {
                    'rvd': rvd.get('Date fabrication BATTERIE', 'N/A'),
                    'image': battery_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
            
            # Comparaison des électrodes
            electrode_data = next((i for i in images if i['type'] == 'Electrodes'), None)
            if electrode_data:
                results['electrode_serial'] = {
                    'rvd': rvd.get("N° série nouvelles électrodes pédiatriques", 'N/A'),
                    'image': electrode_data.get('serial', 'N/A'),
                    'match': normalize_serial(rvd.get('N° série nouvelles électrodes pédiatriques', '')) ==
                            normalize_serial(electrode_data.get('serial', ''))
                }
                
                rvd_date, rvd_err = parse_date(rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', ''))
                img_date, img_err = parse_date(electrode_data.get('date', ''))
                results['electrode_date'] = {
                    'rvd': rvd.get('Date péremption des nouvelles éléctrodes pédiatriques', 'N/A'),
                    'image': electrode_data.get('date', 'N/A'),
                    'match': rvd_date == img_date if not (rvd_err or img_err) else False,
                    'errors': [e for e in [rvd_err, img_err] if e]
                }
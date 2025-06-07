import gradio as gr
import whisperx
import torch
import tempfile
import json
import os
from datetime import datetime
from pathlib import Path
import gc

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if torch.cuda.is_available() else "float32"

# Variables globales pour les modèles
whisper_model = None
diarize_model = None
align_model = None
align_metadata = None

def load_models():
    """Charge tous les modèles nécessaires"""
    global whisper_model, diarize_model, align_model, align_metadata
    
    try:
        # 1. Modèle Whisper
        print("Chargement du modèle Whisper...")
        whisper_model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        
        # 2. Modèle d'alignement (sera chargé dynamiquement selon la langue)
        print("Modèles chargés avec succès!")
        
        return True
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        return False

def load_alignment_model(language_code):
    """Charge le modèle d'alignement pour une langue spécifique"""
    global align_model, align_metadata
    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language_code, 
            device=device
        )
        return True
    except Exception as e:
        print(f"Erreur chargement alignement {language_code}: {e}")
        return False

def load_diarization_model():
    """Charge le modèle de diarisation"""
    global diarize_model
    try:
        # Utilise le token HF si disponible
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=device
            )
            return True
        else:
            print("Token HF manquant pour la diarisation")
            return False
    except Exception as e:
        print(f"Erreur chargement diarisation: {e}")
        return False

def format_timestamp(seconds):
    """Convertit les secondes en format timestamp SRT"""
    if seconds is None:
        return "00:00:00,000"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

def generate_srt_content(segments):
    """Génère le contenu SRT avec speakers si disponible"""
    srt_content = ""
    segment_counter = 1
    
    for segment in segments:
        start_time = format_timestamp(segment.get('start'))
        end_time = format_timestamp(segment.get('end'))
        text = segment.get('text', '').strip()
        speaker = segment.get('speaker', '')
        
        if text:
            srt_content += f"{segment_counter}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            
            # Ajoute le speaker si disponible
            if speaker:
                srt_content += f"[{speaker}] {text}\n\n"
            else:
                srt_content += f"{text}\n\n"
            
            segment_counter += 1
    
    return srt_content

def transcribe_with_whisperx(audio_file, language="auto", enable_diarization=True):
    """Transcription complète avec WhisperX"""
    
    if not whisper_model:
        return "Erreur: Modèles non chargés", "", "", ""
    
    try:
        # 1. Transcription de base
        print("Étape 1: Transcription...")
        audio = whisperx.load_audio(audio_file)
        
        if language == "auto":
            result = whisper_model.transcribe(audio, batch_size=16)
            detected_language = result["language"]
        else:
            result = whisper_model.transcribe(audio, batch_size=16, language=language)
            detected_language = language
        
        print(f"Langue détectée/utilisée: {detected_language}")
        
        # 2. Alignement pour améliorer les timestamps
        print("Étape 2: Alignement des timestamps...")
        if load_alignment_model(detected_language):
            result = whisperx.align(result["segments"], align_model, align_metadata, 
                                 audio, device, return_char_alignments=False)
        
        segments = result["segments"]
        
        # 3. Diarisation (séparation des locuteurs) si demandée
        speakers_info = ""
        if enable_diarization:
            print("Étape 3: Diarisation (identification des locuteurs)...")
            if load_diarization_model():
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
                segments = result["segments"]
                
                # Compte les locuteurs
                speakers = set()
                for segment in segments:
                    if 'speaker' in segment:
                        speakers.add(segment['speaker'])
                
                if speakers:
                    speakers_info = f"\n🎙️ Locuteurs identifiés: {', '.join(sorted(speakers))}\n"
        
        # Génération des différents formats
        full_text = ""
        timestamped_text = ""
        
        for segment in segments:
            text = segment.get('text', '').strip()
            start = segment.get('start')
            end = segment.get('end')
            speaker = segment.get('speaker', '')
            
            if text:
                # Texte brut
                full_text += text + " "
                
                # Texte horodaté
                start_str = format_timestamp(start).replace(',', '.')
                end_str = format_timestamp(end).replace(',', '.')
                
                if speaker:
                    timestamped_text += f"[{start_str} → {end_str}] [{speaker}] {text}\n\n"
                else:
                    timestamped_text += f"[{start_str} → {end_str}] {text}\n\n"
        
        # Génération SRT
        srt_content = generate_srt_content(segments)
        
        # JSON détaillé
        json_output = {
            "language": detected_language,
            "duration": max([seg.get('end', 0) for seg in segments]) if segments else 0,
            "speakers_detected": len(set(seg.get('speaker', '') for seg in segments if seg.get('speaker'))),
            "segments": segments,
            "full_text": full_text.strip(),
            "transcription_date": datetime.now().isoformat(),
            "model_info": {
                "whisper_model": "large-v3",
                "diarization_enabled": enable_diarization,
                "alignment_enabled": True
            }
        }
        
        json_content = json.dumps(json_output, indent=2, ensure_ascii=False)
        
        # Nettoyage mémoire
        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        success_msg = f"✅ Transcription terminée!\n🌍 Langue: {detected_language}{speakers_info}"
        return success_msg + "\n" + full_text.strip(), timestamped_text, srt_content, json_content
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de la transcription: {str(e)}"
        return error_msg, "", "", ""

def create_download_files(text_content, srt_content, json_content):
    """Crée les fichiers temporaires pour téléchargement"""
    files = []
    
    try:
        if text_content and not text_content.startswith("❌"):
            # Fichier texte
            txt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                                 delete=False, encoding='utf-8')
            # Nettoie le texte des émojis et infos pour le fichier
            clean_text = text_content.split('\n', 1)[-1] if '\n' in text_content else text_content
            txt_file.write(clean_text)
            txt_file.close()
            files.append(txt_file.name)
            
            # Fichier SRT
            if srt_content:
                srt_file = tempfile.NamedTemporaryFile(mode='w', suffix='.srt', 
                                                     delete=False, encoding='utf-8')
                srt_file.write(srt_content)
                srt_file.close()
                files.append(srt_file.name)
            
            # Fichier JSON
            if json_content:
                json_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', 
                                                      delete=False, encoding='utf-8')
                json_file.write(json_content)
                json_file.close()
                files.append(json_file.name)
    
    except Exception as e:
        print(f"Erreur création fichiers: {e}")
    
    return files

def process_audio(audio_file, language, enable_diarization, progress=gr.Progress()):
    """Traite l'audio avec barre de progression"""
    
    if audio_file is None:
        return "❌ Veuillez sélectionner un fichier audio", "", "", "", []
    
    progress(0.1, desc="Chargement du fichier...")
    
    # Transcription
    progress(0.3, desc="Transcription en cours...")
    result_text, timestamped_text, srt_content, json_content = transcribe_with_whisperx(
        audio_file, language, enable_diarization
    )
    
    progress(0.9, desc="Génération des fichiers...")
    
    # Création des fichiers de téléchargement
    download_files = create_download_files(result_text, srt_content, json_content)
    
    progress(1.0, desc="Terminé!")
    
    return result_text, timestamped_text, srt_content, json_content, download_files

# Interface Gradio
def create_interface():
    with gr.Blocks(title="WhisperX Pro - Transcription avec Diarisation", 
                   theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎙️ WhisperX Pro - Transcription Avancée
        
        **Transcription professionnelle avec séparation des locuteurs**
        
        ✨ Fonctionnalités :
        - 🗣️ **Diarisation** : Identification automatique des locuteurs  
        - ⏰ **Alignement précis** : Timestamps optimisés
        - 🌍 **Multilingue** : Plus de 100 langues supportées
        - 📁 **Multi-export** : TXT, SRT, JSON
        """)
        
        # Status de chargement des modèles
        model_status = gr.Markdown("🔄 Chargement des modèles en cours...")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="📁 Fichier Audio/Vidéo",
                    type="filepath"
                )
                
                language_select = gr.Dropdown(
                    label="🌍 Langue",
                    choices=[
                        ("Détection automatique", "auto"),
                        ("Français", "fr"),
                        ("Anglais", "en"),
                        ("Espagnol", "es"),
                        ("Italien", "it"),
                        ("Allemand", "de"),
                        ("Portugais", "pt"),
                        ("Chinois", "zh"),
                        ("Japonais", "ja"),
                        ("Arabe", "ar")
                    ],
                    value="auto"
                )
                
                diarization_checkbox = gr.Checkbox(
                    label="🎭 Activer la diarisation (séparation des locuteurs)",
                    value=True,
                    info="Identifie et sépare les différents locuteurs"
                )
                
                transcribe_btn = gr.Button(
                    "🚀 Transcrire avec WhisperX",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("📝 Résultat"):
                        text_output = gr.Textbox(
                            label="Transcription",
                            lines=12,
                            max_lines=20,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("⏰ Horodatage"):
                        timestamped_output = gr.Textbox(
                            label="Transcription horodatée avec locuteurs",
                            lines=12,
                            max_lines=20,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🎬 SRT"):
                        srt_output = gr.Textbox(
                            label="Sous-titres SRT",
                            lines=12,
                            max_lines=20,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("📊 JSON"):
                        json_output = gr.Textbox(
                            label="Export JSON détaillé",
                            lines=12,
                            max_lines=20,
                            show_copy_button=True
                        )
                
                download_files = gr.File(
                    label="📥 Télécharger les fichiers",
                    file_count="multiple",
                    visible=False
                )
        
        # Événements
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, language_select, diarization_checkbox],
            outputs=[text_output, timestamped_output, srt_output, json_output, download_files]
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[download_files]
        )
        
        gr.Markdown("""
        ## ⚙️ Configuration requise
        
        - **Token Hugging Face** : Nécessaire pour la diarisation (modèles pyannote)
        - **GPU recommandé** : Pour de meilleures performances
        - **Formats supportés** : MP3, WAV, MP4, M4A, FLAC, etc.
        
        ## 🎯 Optimisé pour
        
        - Interviews et entretiens
        - Réunions et conférences  
        - Podcasts multi-locuteurs
        - Tests utilisateurs
        """)
        
        # Mise à jour du status au chargement
        interface.load(
            fn=lambda: "✅ Modèles chargés - Prêt à transcrire!" if load_models() else "❌ Erreur de chargement des modèles",
            outputs=[model_status]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)

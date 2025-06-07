# Importations (inchangées)
import gradio as gr
import whisperx
import torch
import os
import time
from pyannote.audio import Pipeline
import logging
import pandas as pd

# --- CONFIGURATION (inchangée) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16 # Vous pouvez augmenter ceci à 32 si vous avez beaucoup de VRAM
compute_type = "float16" if torch.cuda.is_available() else "int8"
logging.info(f"Utilisation du device: {device} avec le compute_type: {compute_type}")

# --- GESTION DES MODÈLES (inchangée) ---
model_cache = {}
diarize_pipeline = None
alignment_models_cache = {} # Cache pour les modèles d'alignement

def load_transcription_model(model_size):
    """Charge un modèle Whisper de taille spécifique s'il n'est pas déjà en cache."""
    if model_size in model_cache:
        logging.info(f"Modèle '{model_size}' chargé depuis le cache.")
        return model_cache[model_size]
    
    logging.info(f"Chargement du modèle Whisper '{model_size}'...")
    try:
        # On spécifie le type de calcul ici
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        model_cache[model_size] = model
        logging.info("Modèle chargé avec succès.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_size}: {e}", exc_info=True)
        if model_size in model_cache: del model_cache[model_size]
        raise

def load_diarization_model():
    """Charge le modèle de diarisation si nécessaire."""
    global diarize_pipeline
    if HF_TOKEN and diarize_pipeline is None:
        logging.info("Chargement du modèle de diarisation Pyannote...")
        try:
            diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(device))
            logging.info("Modèle de diarisation chargé.")
        except Exception as e:
            logging.error(f"Erreur chargement modèle diarisation: {e}", exc_info=True)
            diarize_pipeline = None
            logging.warning("La diarisation sera désactivée.")

def load_alignment_model(language_code):
    """Charge un modèle d'alignement s'il n'est pas déjà en cache."""
    if language_code in alignment_models_cache:
        logging.info(f"Modèle d'alignement pour '{language_code}' chargé depuis le cache.")
        return alignment_models_cache[language_code]
    
    logging.info(f"Chargement du modèle d'alignement pour '{language_code}'...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        alignment_models_cache[language_code] = (model_a, metadata)
        logging.info("Modèle d'alignement chargé.")
        return model_a, metadata
    except Exception as e:
        logging.error(f"Erreur chargement modèle d'alignement: {e}", exc_info=True)
        raise

# --- FONCTION PRINCIPALE (RECONFIGURÉE POUR LA VITESSE) ---
def transcribe_and_diarize(audio_file, model_size, enable_diarization, progress=gr.Progress(track_tqdm=True)):
    
    # --- Chargement des modèles ---
    try:
        yield "Chargement des modèles...", gr.update(interactive=False), ""
        progress(0, desc=f"Chargement du modèle {model_size}...")
        model = load_transcription_model(model_size)
        if enable_diarization:
            load_diarization_model()
    except Exception as e:
        yield f"Erreur de chargement: {e}", gr.update(interactive=True), ""
        return

    try:
        # --- Étape 1: Chargement audio ---
        progress(0.1, desc="Chargement audio...")
        yield "Étape 1/3 : Chargement audio...", gr.update(interactive=False), ""
        audio = whisperx.load_audio(audio_file)
        
        # --- Étape 2: Transcription ---
        progress(0.2, desc=f"Transcription avec {model_size}...")
        yield f"Étape 2/3 : Transcription ({model_size})...", gr.update(interactive=False), ""
        result = model.transcribe(audio, batch_size=batch_size)
        language_code = result["language"]

        # --- Étape 3: Alignement & Diarisation (Optionnel) ---
        if enable_diarization and diarize_pipeline:
            progress(0.6, desc="Alignement & Diarisation...")
            yield "Étape 3/3 : Alignement & Diarisation...", gr.update(interactive=False), ""
            
            model_a, metadata = load_alignment_model(language_code)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

            diarize_segments = diarize_pipeline(audio_file, min_speakers=2, max_speakers=5)
            diarization_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
            diarization_df['start'] = diarization_df['segment'].apply(lambda x: x.start)
            diarization_df['end'] = diarization_df['segment'].apply(lambda x: x.end)
            result = whisperx.assign_word_speakers(diarization_df[['start', 'end', 'speaker']], result)
        
        # --- Formatage final ---
        progress(0.95, desc="Formatage...")
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker_prefix = f"{segment.get('speaker', 'LOCUTEUR')}: " if enable_diarization else ""
            text = segment['text']
            output_text += f"[{start_time}] {speaker_prefix}{text.strip()}\n"

        progress(1, "Terminé !")
        yield "Terminé !", gr.update(interactive=True), output_text

    except Exception as e:
        logging.error(f"Erreur pendant le traitement: {e}", exc_info=True)
        yield "Une erreur est survenue.", gr.update(interactive=True), f"Erreur: {e}"

# --- INTERFACE UTILISATEUR AVEC OPTIONS PAR DÉFAUT ---
with gr.Blocks(title="TranscribeMe", theme=gr.themes.Default()) as app:
    gr.Markdown("# TranscribeMe : Votre Assistant de Transcription Audio")
    gr.Markdown("Choisissez vos options de performance puis lancez la transcription.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="1. Chargez votre fichier audio")
            
            gr.Markdown("### 2. Choisissez vos options")
            
            # --- MODIFICATION: Utilisation de gr.Radio pour le choix du modèle ---
            model_size_radio = gr.Radio(
                ["tiny", "base", "small", "medium", "large-v3"], 
                value="medium", # <-- DÉFAUT: medium
                label="Qualité du Modèle (Vitesse vs Précision)",
                info="Medium est un bon équilibre. Large est plus précis mais beaucoup plus lent."
            )
            
            # --- MODIFICATION: Diarisation activée par défaut ---
            diarize_checkbox = gr.Checkbox(
                value=True, # <-- DÉFAUT: Activé
                label="Identifier les locuteurs (diarisation)",
                info="Décochez pour une transcription beaucoup plus rapide mais sans attribution des locuteurs."
            )
            
            transcribe_button = gr.Button("3. Lancer la Transcription", variant="primary")

        with gr.Column(scale=2):
            status_textbox = gr.Textbox(label="Statut du Traitement", interactive=False, placeholder="En attente...")
            output_transcription = gr.Textbox(label="Transcription", interactive=True, lines=20, placeholder="Le résultat apparaîtra ici...")

    transcribe_button.click(
        fn=transcribe_and_diarize,
        # On passe les nouveaux composants 'radio' et 'checkbox'
        inputs=[audio_input, model_size_radio, diarize_checkbox],
        outputs=[status_textbox, transcribe_button, output_transcription]
    )

if __name__ == "__main__":
    app.launch(debug=True)

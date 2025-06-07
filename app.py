# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os
import time
from pyannote.audio import Pipeline
import logging
import pandas as pd

# --- 1. CONFIGURATION DU LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration Initiale ---
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if torch.cuda.is_available() else "int8"
logging.info(f"Utilisation du device: {device} avec le compute_type: {compute_type}")

# --- GESTION DES MODÈLES (AMÉLIORÉE POUR LE CHOIX DYNAMIQUE ET LE LAZY LOADING) ---
# Dictionnaires pour mettre en cache les modèles déjà chargés
model_cache = {}
diarize_pipeline = None
alignment_models_cache = {}

def load_transcription_model(model_size):
    """Charge un modèle Whisper de la taille spécifiée s'il n'est pas déjà en cache."""
    if model_size in model_cache:
        logging.info(f"Modèle '{model_size}' déjà en cache.")
        return model_cache[model_size]
    
    logging.info(f"Chargement du modèle Whisper '{model_size}'...")
    try:
        model = whisperx.load_model(model_size, device, compute_type=compute_type)
        model_cache[model_size] = model
        logging.info("Modèle chargé.")
        return model
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle {model_size}: {e}", exc_info=True)
        raise

def load_diarization_model():
    """Charge le modèle de diarisation si nécessaire."""
    global diarize_pipeline
    if HF_TOKEN and diarize_pipeline is None:
        logging.info("Chargement du modèle de diarisation Pyannote...")
        try:
            diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN).to(torch.device(device))
        except Exception as e:
            logging.error(f"Erreur chargement modèle diarisation: {e}", exc_info=True)
            diarize_pipeline = None
            logging.warning("Diarisation désactivée en raison d'une erreur de chargement.")

def load_alignment_model(language_code):
    """Charge ou récupère depuis le cache le modèle d'alignement pour une langue donnée."""
    if language_code in alignment_models_cache:
        return alignment_models_cache[language_code]
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    alignment_models_cache[language_code] = (model_a, metadata)
    return model_a, metadata

# --- Fonction principale de transcription (Générateur) ---
def transcribe_and_diarize(audio_file, model_size, progress=gr.Progress(track_tqdm=True)):
    
    # --- Chargement des modèles à la volée ---
    try:
        # CORRECTION BUG AFFICHAGE : On met à jour 3 composants
        yield "Chargement des modèles...", gr.update(interactive=False), ""
        progress(0, desc=f"Chargement du modèle {model_size}...")
        model = load_transcription_model(model_size)
        # On ne charge le modèle de diarisation que si on a un token
        if HF_TOKEN:
            load_diarization_model()
    except Exception as e:
        error_message = f"Erreur critique lors du chargement des modèles: {e}"
        logging.error(error_message, exc_info=True)
        yield error_message, gr.update(interactive=True), ""
        return

    logging.info(f"Début du traitement pour le fichier : {audio_file}")
    
    try:
        # --- Étape 1: Chargement audio ---
        progress(0.1, desc="Chargement audio...")
        yield "Étape 1/3: Chargement...", gr.update(interactive=False), ""
        audio = whisperx.load_audio(audio_file)
        
        # --- Étape 2: Transcription ---
        progress(0.3, desc="Transcription...")
        yield f"Étape 2/3: Transcription ({model_size})...", gr.update(interactive=False), ""
        result = model.transcribe(audio, batch_size=batch_size)
        language_code = result["language"]

        # --- Étape 3: Alignement et Diarisation (si possible) ---
        progress(0.7, desc="Alignement & Diarisation...")
        yield "Étape 3/3: Alignement & Diarisation...", gr.update(interactive=False), ""
        model_a, metadata = load_alignment_model(language_code)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # La diarisation ne se fait que si le modèle a pu être chargé
        if diarize_pipeline:
            diarize_segments = diarize_pipeline(audio_file, min_speakers=2, max_speakers=5)
            diarization_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
            diarization_df['start'] = diarization_df['segment'].apply(lambda x: x.start)
            diarization_df['end'] = diarization_df['segment'].apply(lambda x: x.end)
            result = whisperx.assign_word_speakers(diarization_df[['start', 'end', 'speaker']], result)
        else:
            logging.info("Pipeline de diarisation non disponible, l'étape est ignorée.")

        # --- Formatage final ---
        progress(0.95, desc="Formatage du texte...")
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            # Le préfixe 'speaker' n'est ajouté que si l'information existe
            speaker = segment.get('speaker')
            speaker_prefix = f"{speaker}: " if speaker else ""
            text = segment['text']
            output_text += f"[{start_time}] {speaker_prefix}{text.strip()}\n"
        
        progress(1, "Terminé !")
        # CORRECTION BUG AFFICHAGE : Le dernier yield met à jour les 3 composants avec le résultat final
        yield "Terminé !", gr.update(interactive=True), output_text
        
    except Exception as e:
        logging.error(f"Erreur pendant la transcription: {e}", exc_info=True)
        # CORRECTION BUG AFFICHAGE : On met à jour les 3 composants en cas d'erreur
        yield "Une erreur est survenue.", gr.update(interactive=True), f"Erreur: {e}"

# --- Interface Utilisateur (robuste et avec options) ---
description = """
**Comment ça marche ?**
1.  **Choisissez la qualité du modèle.** `tiny` est le plus rapide, `large-v3` est le plus précis.
2.  **Chargez un fichier audio** ou enregistrez-vous.
3.  Cliquez sur **"Lancer la Transcription"**. Le résultat s'affichera ici.
"""

with gr.Blocks(title="TranscribeMe", theme=gr.themes.Soft()) as app:
    gr.Markdown("# TranscribeMe : Votre Assistant de Transcription Audio")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            # AJOUT: Sélecteur de modèle
            model_size_dropdown = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="tiny", # Par défaut le plus petit/rapide
                label="Qualité du Modèle (vitesse vs précision)"
            )
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Chargez votre fichier audio")
            transcribe_button = gr.Button("Lancer la Transcription", variant="primary")
        
        with gr.Column(scale=2):
            status_textbox = gr.Textbox(label="Statut du Traitement", interactive=False)
            output_transcription = gr.Textbox(label="Transcription", interactive=True, lines=20, placeholder="Le résultat apparaîtra ici...")
    
    # CORRECTION BUG AFFICHAGE : La logique de click est simplifiée
    # Elle prend maintenant le sélecteur de modèle en entrée
    # et met à jour les 3 composants de sortie directement.
    transcribe_button.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input, model_size_dropdown],
        outputs=[status_textbox, transcribe_button, output_transcription]
    )

if __name__ == "__main__":
    app.launch(debug=True)

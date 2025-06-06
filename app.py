# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os
import time
from pyannote.audio import Pipeline
import logging # <-- NOUVEL IMPORT pour le logging

# --- 1. CONFIGURATION DU LOGGING ---
# On configure le logger pour qu'il affiche la date, le niveau du log, et le message.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Configuration Initiale ---
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if torch.cuda.is_available() else "int8"

# --- Chargement des modèles ---
model = None
diarize_pipeline = None

# On utilise un bloc try/except pour attraper les erreurs de chargement de modèle
try:
    if HF_TOKEN:
        logging.info("Chargement du modèle Whisper...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        
        logging.info("Chargement du modèle de diarisation depuis Pyannote...")
        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        ).to(torch.device(device))
        
        logging.info("Modèles chargés avec succès.")
    else:
        logging.warning("Le token Hugging Face (HF_TOKEN) n'est pas configuré. La diarisation sera désactivée.")
except Exception as e:
    logging.error(f"Erreur lors du chargement des modèles : {e}", exc_info=True)


# --- Fonction principale de transcription ---
def transcribe_and_diarize(audio_file):
    if not model:
        logging.error("Le modèle Whisper n'est pas chargé. Impossible de continuer.")
        return "Erreur critique : Le modèle de transcription n'a pas pu être chargé. Vérifiez les logs."

    logging.info(f"Début du traitement pour le fichier : {audio_file}")
    
    try:
        # Étape 1: Charger l'audio
        logging.info("Étape 1/6 : Chargement du fichier audio...")
        audio = whisperx.load_audio(audio_file)
        logging.info("Fichier audio chargé.")

        # Étape 2: Transcription avec WhisperX
        logging.info("Étape 2/6 : Lancement de la transcription Whisper...")
        result = model.transcribe(audio, batch_size=batch_size)
        logging.info("Transcription terminée.")

        # Étape 3: Alignement des segments
        logging.info("Étape 3/6 : Alignement des segments de la transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        logging.info("Alignement terminé.")

        # Étape 4: Diarisation (si disponible)
        if diarize_pipeline:
            logging.info("Étape 4/6 : Identification des locuteurs (diarisation)...")
            diarize_segments = diarize_pipeline(audio_file)
            logging.info("Diarisation terminée.")
            
            # Étape 5: Assignation des locuteurs
            logging.info("Étape 5/6 : Assignation des locuteurs aux mots...")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logging.info("Assignation des locuteurs terminée.")
        else:
            logging.info("Étape 4/6 & 5/6 : Diarisation et assignation des locuteurs ignorées (pas de token).")


        # Étape 6: Formatage de la sortie
        logging.info("Étape 6/6 : Formatage du texte final...")
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            # Si le locuteur n'a pas été assigné, on met 'LOCUTEUR' par défaut
            speaker = segment.get('speaker', 'LOCUTEUR') 
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
        logging.info("Formatage terminé. Traitement réussi !")
        
        return output_text
        
    except Exception as e:
        # --- AMÉLIORATION MAJEURE DE LA CAPTURE D'ERREUR ---
        logging.error(f"Une erreur inattendue est survenue pendant la transcription: {e}", exc_info=True)
        return "Une erreur est survenue durant le traitement. Veuillez consulter les logs pour plus de détails."

# --- Interface Utilisateur (inchangée) ---
description = "..." # Le reste du fichier est identique

with gr.Blocks(title="TranscribeMe") as app:
   # ...
   # Le reste du fichier est identique...
   # ...
    gr.Markdown("# TranscribeMe : Votre Assistant de Transcription Audio")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Chargez votre fichier audio")
            transcribe_button = gr.Button("Lancer la Transcription", variant="primary")
        with gr.Column(scale=2):
            output_transcription = gr.Textbox(label="Transcription", interactive=False, lines=20, placeholder="Le résultat apparaîtra ici...")
    
    transcribe_button.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input],
        outputs=output_transcription
    )

if __name__ == "__main__":
    app.launch(debug=True)

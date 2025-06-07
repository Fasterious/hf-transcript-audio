# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os
import time
from pyannote.audio import Pipeline
import logging
import pandas as pd # <-- CORRECTION : Importation de pandas

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

# --- Chargement des modèles ---
model = None
diarize_pipeline = None

try:
    if HF_TOKEN:
        logging.info("Chargement du modèle Whisper...")
        model = whisperx.load_model("tiny", device, compute_type=compute_type)
        
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
        logging.info("Étape 1/6 : Chargement du fichier audio...")
        audio = whisperx.load_audio(audio_file)

        logging.info("Étape 2/6 : Lancement de la transcription Whisper...")
        result = model.transcribe(audio, batch_size=batch_size)

        logging.info("Étape 3/6 : Alignement des segments de la transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        if diarize_pipeline:
            logging.info("Étape 4/6 : Identification des locuteurs (diarisation)...")
            diarize_segments = diarize_pipeline(audio_file, min_speakers=2, max_speakers=5) # Ajout de min/max speakers est une bonne pratique
            logging.info("Diarisation terminée.")
            
            # --- CORRECTION : Conversion de la sortie de pyannote en DataFrame ---
            # On crée un DataFrame avec les colonnes 'start', 'end', 'speaker' que whisperx comprend.
            diarization_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
            diarization_df['start'] = diarization_df['segment'].apply(lambda x: x.start)
            diarization_df['end'] = diarization_df['segment'].apply(lambda x: x.end)
            # On ne garde que les colonnes nécessaires pour whisperx
            diarization_df = diarization_df[['start', 'end', 'speaker']]
            # --- FIN DE LA CORRECTION ---
            
            logging.info("Étape 5/6 : Assignation des locuteurs aux mots...")
            # CORRECTION : On passe le DataFrame formaté au lieu de l'objet pyannote brut
            result = whisperx.assign_word_speakers(diarization_df, result)
            logging.info("Assignation des locuteurs terminée.")
        else:
            logging.info("Étape 4/6 & 5/6 : Diarisation et assignation des locuteurs ignorées (pas de token).")

        logging.info("Étape 6/6 : Formatage du texte final...")
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker = segment.get('speaker', 'LOCUTEUR') 
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
        logging.info("Formatage terminé. Traitement réussi !")
        
        return output_text
        
    except Exception as e:
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

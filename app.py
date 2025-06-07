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

# --- AMÉLIORATION : LAZY LOADING & CACHE ---
# Les modèles ne sont pas chargés au démarrage pour une UI plus rapide.
model = None
diarize_pipeline = None
# Le cache évite de recharger le modèle d'alignement pour une même langue.
alignment_models_cache = {}

def load_models():
    """
    Charge les modèles Whisper et Pyannote s'ils ne sont pas déjà en mémoire.
    Cette fonction est appelée "paresseusement" (lazy) lors du premier clic.
    """
    global model, diarize_pipeline
    
    # Charger le modèle Whisper s'il n'est pas déjà en mémoire
    if model is None:
        try:
            logging.info("Chargement du modèle Whisper (large-v3)...")
            model = whisperx.load_model("large-v3", device, compute_type=compute_type)
            logging.info("Modèle Whisper chargé.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle Whisper : {e}", exc_info=True)
            raise  # Propage l'erreur pour l'afficher dans l'UI et arrêter le processus

    # Charger la pipeline de diarisation si un token est disponible et qu'elle n'est pas déjà en mémoire
    if HF_TOKEN and diarize_pipeline is None:
        try:
            logging.info("Chargement du modèle de diarisation Pyannote...")
            diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN
            ).to(torch.device(device))
            logging.info("Modèle de diarisation chargé.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la pipeline de diarisation : {e}", exc_info=True)
            # Ne pas propager, la diarisation est optionnelle. On s'assure juste que la pipeline reste à None.
            diarize_pipeline = None
            logging.warning("La diarisation sera désactivée à cause d'une erreur de chargement.")

# --- Fonction principale de transcription (Générateur avec toutes les améliorations) ---
def transcribe_and_diarize(audio_file, progress=gr.Progress(track_tqdm=True)):
    
    # Étape 0: Vérification et chargement des modèles (Lazy Loading)
    try:
        # On met à jour l'UI pour indiquer le chargement.
        # Le troisième output (pour la transcription) est une chaîne vide pour l'instant.
        yield "Chargement des modèles...", gr.Button(interactive=False, value="Chargement..."), ""
        progress(0, desc="Chargement des modèles...")
        load_models()
    except Exception as e:
        error_message = f"Erreur critique lors du chargement des modèles: {e}"
        logging.error(error_message, exc_info=True)
        yield error_message, gr.Button(interactive=True, value="Lancer la Transcription"), ""
        return

    if not model:
        yield "Erreur: Le modèle Whisper n'a pas pu être chargé.", gr.Button(interactive=True, value="Lancer la Transcription"), ""
        return

    logging.info(f"Début du traitement pour le fichier : {audio_file}")
    
    try:
        # Étape 1: Charger l'audio
        progress(0.1, desc="Étape 1/6 : Chargement du fichier audio...")
        yield "Étape 1/6 : Chargement...", gr.Button(interactive=False, value="Traitement..."), ""
        audio = whisperx.load_audio(audio_file)
        
        # Étape 2: Transcription avec WhisperX
        progress(0.25, desc="Étape 2/6 : Transcription (peut être long)...")
        yield "Étape 2/6 : Transcription...", gr.Button(interactive=False, value="Transcription..."), ""
        result = model.transcribe(audio, batch_size=batch_size)
        language_code = result["language"]

        # Étape 3: Alignement des segments (avec cache)
        progress(0.6, desc="Étape 3/6 : Alignement des segments...")
        yield "Étape 3/6 : Alignement...", gr.Button(interactive=False, value="Alignement..."), ""
        
        if language_code in alignment_models_cache:
            model_a, metadata = alignment_models_cache[language_code]
            logging.info(f"Modèle d'alignement pour '{language_code}' chargé depuis le cache.")
        else:
            logging.info(f"Chargement du modèle d'alignement pour la langue : {language_code}")
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            alignment_models_cache[language_code] = (model_a, metadata)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Étapes 4 & 5: Diarisation et Assignation (avec correction `KeyError`)
        if diarize_pipeline:
            progress(0.75, desc="Étape 4/6 : Identification des locuteurs...")
            yield "Étape 4/6 : Diarisation...", gr.Button(interactive=False, value="Diarisation..."), ""
            diarize_segments = diarize_pipeline(audio_file, min_speakers=2, max_speakers=5)

            # Correction de la KeyError: Conversion de la sortie pyannote en DataFrame pandas
            diarization_df = pd.DataFrame(diarize_segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
            diarization_df['start'] = diarization_df['segment'].apply(lambda x: x.start)
            diarization_df['end'] = diarization_df['segment'].apply(lambda x: x.end)
            diarization_df = diarization_df[['start', 'end', 'speaker']]

            progress(0.9, desc="Étape 5/6 : Assignation des locuteurs...")
            yield "Étape 5/6 : Assignation...", gr.Button(interactive=False, value="Assignation..."), ""
            result = whisperx.assign_word_speakers(diarization_df, result)
        else:
            logging.info("Diarisation et assignation ignorées (pipeline non disponible).")
            
        # Étape 6: Formatage de la sortie
        progress(0.95, desc="Étape 6/6 : Formatage du texte final...")
        yield "Étape 6/6 : Formatage...", gr.Button(interactive=False, value="Finalisation..."), ""
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker = segment.get('speaker', 'LOCUTEUR_INCONNU') 
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
        
        logging.info("Formatage terminé. Traitement réussi !")
        progress(1, "Terminé !")
        
        # Le dernier 'yield' renvoie le résultat final et met à jour tous les composants.
        yield "Terminé !", gr.Button(interactive=True, value="Lancer la Transcription"), output_text
        
    except Exception as e:
        logging.error(f"Une erreur inattendue est survenue pendant la transcription: {e}", exc_info=True)
        # En cas d'erreur, on affiche un message et on réactive le bouton.
        error_details = f"Erreur: {e}"
        yield "Une erreur est survenue. Consultez les logs.", gr.Button(interactive=True, value="Lancer la Transcription"), error_details


# --- Interface Utilisateur (robuste et informative) ---
description = """
**Bienvenue sur TranscribeMe !**

1.  **Chargez un fichier audio** ou enregistrez-vous directement via votre microphone.
2.  Cliquez sur **"Lancer la Transcription"**.
3.  Suivez la progression du traitement grâce à la barre de statut. Les modèles se chargent au premier lancement, cela peut prendre un moment.
4.  Le texte transcrit et identifié par locuteur (si possible) apparaîtra dans la zone de résultat.

**Technologies utilisées :**
-   **Transcription :** `Whisper large-v3` par OpenAI (via `whisperx`).
-   **Diarisation (Locuteurs) :** `pyannote/speaker-diarization-3.1` par `pyannote.audio`.
-   **Alignement :** `wav2vec2` par Facebook AI.
"""

with gr.Blocks(title="TranscribeMe", theme=gr.themes.Soft()) as app:
    gr.Markdown("# TranscribeMe : Votre Assistant de Transcription Audio")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Chargez votre fichier audio")
            transcribe_button = gr.Button("Lancer la Transcription", variant="primary")
        with gr.Column(scale=2):
            status_textbox = gr.Textbox(label="Statut du Traitement", interactive=False)
            output_transcription = gr.Textbox(label="Transcription", interactive=True, lines=20, placeholder="Le résultat apparaîtra ici...")
    
    # La logique est maintenant simple et directe : un clic déclenche la fonction,
    # qui met à jour les 3 composants listés dans 'outputs'.
    transcribe_button.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input],
        outputs=[status_textbox, transcribe_button, output_transcription]
    )

if __name__ == "__main__":
    app.launch(debug=True)

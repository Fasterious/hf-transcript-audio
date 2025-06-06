# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os
import time
from pyannote.audio import Pipeline # <--- NOUVEL IMPORT

# --- Configuration Initiale ---
HF_TOKEN = os.getenv("HF_TOKEN")
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if torch.cuda.is_available() else "int8"

# --- Chargement des modèles ---
model = None
diarize_pipeline = None

if HF_TOKEN:
    print("Chargement du modèle Whisper...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    
    print("Chargement du modèle de diarisation directement depuis Pyannote...")
    # MODIFICATION CLÉ : On utilise Pipeline.from_pretrained de pyannote
    diarize_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    ).to(torch.device(device))
    
    print("Modèles chargés avec succès.")
else:
    print("Avertissement : Le token Hugging Face (HF_TOKEN) n'est pas configuré dans les secrets.")

# --- Fonction principale de transcription ---
def transcribe_and_diarize(audio_file):
    if not model or not diarize_pipeline:
        return "Erreur : L'application n'a pas pu démarrer car le HF_TOKEN n'est pas configuré dans les secrets du Space."

    try:
        # 1. Charger l'audio
        audio = whisperx.load_audio(audio_file)
        
        # 2. Transcription avec WhisperX
        result = model.transcribe(audio, batch_size=batch_size)
        
        # 3. Alignement des segments
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # 4. Diarisation avec Pyannote
        print("Identification des locuteurs (diarisation)...")
        diarize_segments = diarize_pipeline(audio_file)
        
        # 5. Assignation des locuteurs aux mots
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 6. Formatage de la sortie
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker = segment.get('speaker', 'LOCUTEUR_INCONNU')
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
            
        return output_text
        
    except Exception as e:
        return f"Une erreur est survenue durant le traitement : {e}"

# --- Interface Utilisateur (inchangée) ---
description = """
Bienvenue sur **TranscribeMe** 🎙️
<br>
Chargez simplement un fichier audio et obtenez une transcription complète avec identification des locuteurs et horodatage.
"""

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate").set(body_background_fill_dark='*neutral_950')

with gr.Blocks(theme=theme, title="TranscribeMe") as app:
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

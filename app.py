# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os # Importé pour accéder aux variables d'environnement
import time
from huggingface_hub import HfApi, HfFolder

# --- Configuration Initiale ---

# Lire le token d'accès depuis les "Secrets" de Hugging Face
# os.getenv("HF_TOKEN") va chercher la variable que vous avez définie.
HF_TOKEN = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
compute_type = "float16" if torch.cuda.is_available() else "int8"

# --- Chargement des modèles ---

# On ne charge les modèles que si un token est bien présent.
if HF_TOKEN:
    print("Chargement du modèle Whisper...")
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    print("Chargement du modèle de diarisation...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    print("Modèles chargés avec succès.")
else:
    print("Avertissement : Le token Hugging Face n'a pas été trouvé dans les secrets. La diarisation ne fonctionnera pas.")
    model = None # On met le modèle à None pour gérer l'erreur dans l'interface
    diarize_model = None

# --- Fonction principale de transcription (simplifiée) ---

def transcribe_and_diarize(audio_file):
    """
    Cette fonction ne prend plus le token en argument.
    Elle utilise directement les modèles chargés au démarrage.
    """
    if not model or not diarize_model:
        return "Erreur : L'application n'a pas pu démarrer car le HF_TOKEN n'est pas configuré dans les secrets du Space."

    try:
        # 1. Charger l'audio
        audio = whisperx.load_audio(audio_file)
        
        # 2. Transcription
        result = model.transcribe(audio, batch_size=batch_size)
        
        # 3. Alignement
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # 4. Diarisation
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 5. Formatage de la sortie
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker = segment.get('speaker', 'LOCUTEUR_INCONNU')
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
            
        return output_text
        
    except Exception as e:
        return f"Une erreur est survenue durant le traitement : {e}"

# --- Interface Utilisateur (épurée) ---

description = """
Bienvenue sur **TranscribeMe** 🎙️
<br>
Chargez simplement un fichier audio et obtenez une transcription complète avec identification des locuteurs et horodatage.
<br>
*Cette application est sécurisée : le propriétaire a configuré la clé d'accès en arrière-plan.*
"""

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
).set(
    body_background_fill_dark='*neutral_950'
)

with gr.Blocks(theme=theme, title="TranscribeMe") as app:
    gr.Markdown("# TranscribeMe : Votre Assistant de Transcription Audio")
    gr.Markdown(description)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Chargez votre fichier audio ou enregistrez-vous"
            )
            transcribe_button = gr.Button("Lancer la Transcription", variant="primary")

        with gr.Column(scale=2):
            output_transcription = gr.Textbox(
                label="Transcription",
                interactive=False,
                lines=20,
                placeholder="Le résultat de la transcription apparaîtra ici..."
            )
    
    # On a simplifié l'appel : plus besoin de fournir le token depuis l'interface
    transcribe_button.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input],
        outputs=output_transcription
    )

if __name__ == "__main__":
    app.launch(debug=True)

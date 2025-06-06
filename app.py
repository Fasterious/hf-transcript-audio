# Importation des bibliothèques nécessaires
import gradio as gr
import whisperx
import torch
import os
import time
from huggingface_hub import HfApi, HfFolder

# --- Configuration Initiale ---

# Vérifier si une carte graphique (GPU) est disponible pour un traitement plus rapide
# Si oui, on utilise 'cuda', sinon 'cpu'.
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16  # Réduit la taille du lot si vous avez des problèmes de mémoire
compute_type = "float16" if torch.cuda.is_available() else "int8"

# Charger le modèle de transcription Whisper. 
# "large-v3" est le plus puissant. Vous pouvez utiliser "base" ou "medium" pour des tests plus rapides.
print("Chargement du modèle Whisper...")
model = whisperx.load_model("large-v3", device, compute_type=compute_type)
print("Modèle Whisper chargé.")

# --- Fonction principale de transcription et diarisation ---

def transcribe_and_diarize(audio_file, hf_token):
    """
    Cette fonction prend un fichier audio en entrée, le transcrit, identifie les locuteurs (diarisation)
    et retourne un texte formaté avec timestamps et locuteurs.
    """
    if not hf_token:
        return "Erreur : Veuillez fournir un token d'accès Hugging Face pour la diarisation.", ""
        
    try:
        # 1. Sauvegarder le token Hugging Face pour utiliser le modèle de diarisation
        HfFolder.save_token(hf_token)
        print("Token Hugging Face sauvegardé temporairement.")
        
        # 2. Charger l'audio depuis le chemin du fichier
        print(f"Chargement du fichier audio : {audio_file}")
        if audio_file is None:
            return "Erreur : Aucun fichier audio fourni.", ""
            
        audio = whisperx.load_audio(audio_file)
        
        # 3. Transcription avec Whisper
        print("Début de la transcription...")
        result = model.transcribe(audio, batch_size=batch_size)
        
        # 4. Aligner la transcription avec le modèle d'alignement
        print("Alignement de la transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # 5. Diarisation (identification des locuteurs)
        print("Identification des locuteurs (diarisation)...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("Diarisation terminée.")
        
        # 6. Formater la sortie pour une lecture facile
        output_text = ""
        for segment in result["segments"]:
            start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
            speaker = segment.get('speaker', 'LOCUTEUR_INCONNU')
            text = segment['text']
            output_text += f"[{start_time}] {speaker}:{text.strip()}\n"
            
        print("Formatage de la sortie terminé.")
        return output_text
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
        return f"Une erreur est survenue durant le traitement : {e}", ""

# --- Création de l'Interface Utilisateur avec Gradio ---

# Description en Markdown pour l'interface
description = """
Bienvenue sur **TranscribeMe** 🎙️
<br>
Chargez simplement un fichier audio et obtenez une transcription complète avec identification des locuteurs et horodatage.
<br>
**Important** : Pour l'identification des locuteurs, vous devez fournir un [**token d'accès Hugging Face**](https://huggingface.co/settings/tokens).
"""

# Thème pour un design moderne et épuré
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
            hf_token_input = gr.Textbox(
                label="Token d'accès Hugging Face",
                placeholder="Collez votre token 'read' ici...",
                type="password",
                info="Nécessaire pour l'identification des locuteurs."
            )
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

    transcribe_button.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input, hf_token_input],
        outputs=output_transcription
    )
    
    gr.Examples(
        examples=[
            [os.path.join(os.path.dirname(__file__), "audio_example.mp3"), "HF_TOKEN_ICI"],
        ],
        inputs=[audio_input, hf_token_input],
        outputs=output_transcription,
        fn=transcribe_and_diarize,
        cache_examples=False, # Mettre à True pour le déploiement final
        label="Exemples (cliquez pour essayer)"
    )

# --- Lancement de l'application ---
if __name__ == "__main__":
    app.launch(debug=True)

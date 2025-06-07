import gradio as gr
import whisperx
import torch
import os
import time
import tempfile
import zipfile

# --- 1. Configuration et Chargement des Modèles ---

# Déterminer le device (GPU si disponible, sinon CPU)
# Sur les Spaces HF, le type de hardware est défini dans les paramètres du Space.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
print(f"Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}")

# Récupérer le token HF depuis les secrets du Space
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    print("Avertissement : Le token Hugging Face n'est pas configuré. La diarisation peut échouer.")
    # On peut continuer sans token si on n'utilise pas la diarisation, 
    # mais pyannote en a besoin pour télécharger ses modèles.

# Charger le modèle de diarisation une seule fois au démarrage de l'app
# Cela évite de le recharger à chaque appel, ce qui est très lent.
diarize_model = None
if HF_TOKEN:
    try:
        print("Chargement du modèle de diarisation...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("Modèle de diarisation chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de diarisation : {e}")
        # L'app pourra continuer sans la fonctionnalité de diarisation
        diarize_model = None
else:
    print("Pas de token HF, la diarisation sera désactivée.")

# Dictionnaire pour garder en cache les modèles Whisper chargés
# Clé: taille du modèle (ex: 'large-v3'), Valeur: objet modèle
loaded_models = {}

def get_whisper_model(model_size):
    """Charge un modèle Whisper ou le récupère depuis le cache."""
    if model_size in loaded_models:
        print(f"Récupération du modèle {model_size} depuis le cache.")
        return loaded_models[model_size]
    
    print(f"Chargement du modèle Whisper : {model_size}")
    # Le chargement du modèle peut prendre du temps la première fois
    model = whisperx.load_model(model_size, DEVICE, compute_type=COMPUTE_TYPE)
    loaded_models[model_size] = model
    print(f"Modèle {model_size} chargé.")
    return model

# --- 2. La Fonction de Transcription ---

def transcribe_and_diarize(audio_file_path, language_code, model_size, enable_diarization, progress=gr.Progress(track_ τότε=True)):
    """
    Fonction principale qui prend un fichier audio et retourne les fichiers de transcription.
    """
    if audio_file_path is None:
        return "Veuillez téléverser un fichier audio.", None

    progress(0, desc="Chargement de l'audio...")
    
    try:
        # Charger le modèle Whisper demandé
        model = get_whisper_model(model_size)

        # Charger l'audio depuis le chemin temporaire fourni par Gradio
        audio = whisperx.load_audio(audio_file_path)

        # --- Transcription ---
        progress(0.2, desc=f"Transcription avec {model_size}...")
        result = model.transcribe(audio, batch_size=16, language=language_code)
        
        transcription_text = "Transcription:\n" + "\n".join([seg['text'] for seg in result['segments']])

        # --- Diarisation (si activée et si le modèle est chargé) ---
        if enable_diarization and diarize_model:
            progress(0.6, desc="Alignement du modèle...")
            # Aligner les timings des mots
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            
            progress(0.8, desc="Identification des locuteurs (diarisation)...")
            # Assigner les locuteurs
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Formatter le texte avec les locuteurs
            transcription_text = "Transcription avec locuteurs:\n"
            current_speaker = ""
            for segment in result["segments"]:
                if "speaker" in segment and segment["speaker"] != current_speaker:
                    current_speaker = segment["speaker"]
                    transcription_text += f"\n--- {current_speaker} ---\n"
                transcription_text += segment.get("text", "").strip() + " "

        progress(0.9, desc="Génération des fichiers de sortie...")
        
        # Créer un dossier temporaire pour les fichiers de sortie
        output_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        # Écrire les fichiers de sortie
        writer = whisperx.utils.get_writer("all", output_dir)
        writer(result, base_name) # 'all' crée .txt, .srt, .vtt, .tsv, .json

        # Zipper tous les fichiers de sortie pour un téléchargement facile
        zip_path = os.path.join(output_dir, f"{base_name}_transcription_files.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in os.listdir(output_dir):
                if file.endswith(('.txt', '.srt', '.vtt', '.tsv', '.json')):
                    zf.write(os.path.join(output_dir, file), arcname=file)
        
        progress(1, desc="Terminé !")
        return transcription_text, zip_path

    except Exception as e:
        return f"Une erreur est survenue : {str(e)}", None


# --- 3. L'Interface Gradio ---

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Outil de Transcription Audio avec WhisperX & Diarisation")
    gr.Markdown(
        "Déposez un fichier audio ou vidéo, choisissez les options, et lancez la transcription. "
        "Vous obtiendrez le texte transcrit ainsi qu'un fichier `.zip` contenant les formats `.txt`, `.srt`, `.json`, etc."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Fichier Audio/Vidéo",
                type="filepath" # 'filepath' est plus stable pour les gros fichiers
            )
            
            model_size_dropdown = gr.Dropdown(
                label="Taille du modèle Whisper",
                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                value="large-v3"
            )

            language_dropdown = gr.Dropdown(
                label="Langue de l'audio (Code ISO 639-1)",
                choices=["fr", "en", "es", "de", "it", "auto"],
                value="fr",
                info="Mettez 'auto' pour la détection automatique."
            )
            
            diarization_checkbox = gr.Checkbox(
                label="Activer la diarisation (identification des locuteurs)",
                value=True if diarize_model else False,
                interactive=True if diarize_model else False,
                info="Nécessite un modèle pyannote.audio. Désactivé si le token HF est manquant."
            )

            submit_btn = gr.Button("Lancer la Transcription", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Résultat de la Transcription", lines=15, interactive=False)
            output_files = gr.File(label="Télécharger les fichiers (.zip)", interactive=False)

    submit_btn.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input, language_dropdown, model_size_dropdown, diarization_checkbox],
        outputs=[output_text, output_files]
    )
    
    gr.Examples(
        examples=[
            ["./examples/test_audio_fr.mp3", "fr", "base", True],
            ["./examples/test_audio_en.wav", "en", "small", False],
        ],
        inputs=[audio_input, language_dropdown, model_size_dropdown, diarization_checkbox],
        outputs=[output_text, output_files],
        fn=transcribe_and_diarize,
        cache_examples=True # Mettre en cache pour des démos rapides
    )
    # Pour que les exemples fonctionnent, créez un dossier "examples" dans votre Space
    # et placez-y des fichiers audio.

# Lancer l'application
app.launch(debug=True)

import gradio as gr
import whisperx
import torch
import os
import tempfile
import zipfile
import traceback

# --- 1. Configuration et Chargement des Modèles ---

# Déterminer le device (GPU si disponible, sinon CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Utiliser float16 sur GPU pour la vitesse, int8 sur CPU pour la compatibilité
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
print(f"Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}")

# Récupérer le token HF depuis les secrets du Space
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    print("Avertissement : Le token Hugging Face n'est pas configuré. La diarisation peut échouer.")

# Charger le modèle de diarisation une seule fois au démarrage de l'app
diarize_model = None
if HF_TOKEN:
    try:
        print("Chargement du modèle de diarisation...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=DEVICE)
        print("Modèle de diarisation chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle de diarisation : {e}")
        diarize_model = None
else:
    print("Pas de token HF, la diarisation sera désactivée.")

# Dictionnaire pour garder en cache les modèles Whisper chargés
loaded_models = {}

def get_whisper_model(model_size):
    """Charge un modèle Whisper ou le récupère depuis le cache."""
    if model_size in loaded_models:
        print(f"Récupération du modèle {model_size} depuis le cache.")
        return loaded_models[model_size]
    
    print(f"Chargement du modèle Whisper : {model_size}")
    model = whisperx.load_model(model_size, DEVICE, compute_type=COMPUTE_TYPE)
    loaded_models[model_size] = model
    print(f"Modèle {model_size} chargé.")
    return model

# --- 2. La Fonction de Transcription ---

def transcribe_and_diarize(audio_file_path, language_code, model_size, enable_diarization, progress=gr.Progress(track_tqdm=True)):
    """
    Fonction principale qui prend un fichier audio et retourne les fichiers de transcription.
    """
    if audio_file_path is None:
        return "Veuillez téléverser un fichier audio.", None

    if language_code == "auto":
        language_code = None # WhisperX attend None pour la détection auto

    try:
        progress(0, desc="Chargement du modèle Whisper...")
        model = get_whisper_model(model_size)
        
        progress(0.1, desc="Chargement de l'audio...")
        audio = whisperx.load_audio(audio_file_path)

        progress(0.2, desc=f"Transcription avec {model_size}...")
        result = model.transcribe(audio, batch_size=16, language=language_code)
        
        detected_language = result["language"]
        transcription_text = f"Langue détectée: {detected_language}\n\nTranscription:\n" + "\n".join([seg['text'].strip() for seg in result['segments']])

        if enable_diarization and diarize_model:
            progress(0.6, desc="Alignement du modèle...")
            model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)
            
            progress(0.8, desc="Identification des locuteurs (diarisation)...")
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            transcription_text = f"Langue détectée: {detected_language}\n\nTranscription avec locuteurs:\n"
            current_speaker = None
            full_text_list = []
            for segment in result["segments"]:
                spk = segment.get("speaker", "LOCUTEUR_INCONNU")
                if spk != current_speaker:
                    current_speaker = spk
                    full_text_list.append(f"\n--- {current_speaker} ---\n")
                full_text_list.append(segment.get("text", "").strip())
            transcription_text = "".join(full_text_list)

        progress(0.9, desc="Génération des fichiers de sortie...")
        
        output_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(audio_file_path))[0]
        
        writer = whisperx.utils.get_writer("all", output_dir)
        
        # CORRECTION FINALE : On passe un dictionnaire 'options' vide car il est requis.
        options = {}
        writer(result, base_name, options)

        zip_path = os.path.join(output_dir, f"{base_name}_transcription_files.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for file in os.listdir(output_dir):
                if file.endswith(('.txt', '.srt', '.vtt', '.tsv', '.json')):
                    zf.write(os.path.join(output_dir, file), arcname=file)
        
        progress(1, desc="Terminé !")
        return transcription_text, zip_path

    except Exception as e:
        # Affiche l'erreur complète dans les logs du Space pour un débogage facile
        traceback.print_exc()
        return f"Une erreur critique est survenue : {e}", None


# --- 3. L'Interface Gradio ---

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("<h1>Outil de Transcription Audio avec WhisperX & Diarisation</h1>")
    gr.Markdown(
        "Déposez un fichier audio ou vidéo, choisissez les options, et lancez la transcription. "
        "Vous obtiendrez le texte transcrit ainsi qu'un fichier `.zip` contenant les formats `.txt`, `.srt`, `.json`, etc."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(label="Fichier Audio/Vidéo", type="filepath")
            
            model_size_dropdown = gr.Dropdown(
                label="Taille du modèle Whisper",
                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                value="base", # 'base' est rapide pour tester, l'utilisateur peut choisir 'large-v3' pour la meilleure qualité
                info="Pour une meilleure qualité, utilisez 'large-v3'."
            )

            language_dropdown = gr.Dropdown(
                label="Langue de l'audio (Code ISO 639-1)",
                choices=["fr", "en", "es", "de", "it", "auto"],
                value="auto",
                info="Laissez sur 'auto' pour la détection automatique."
            )
            
            diarization_checkbox = gr.Checkbox(
                label="Activer la diarisation (identification des locuteurs)",
                value=True if diarize_model else False,
                interactive=True if diarize_model else False,
                info="Désactivé si le token HF est manquant ou si le modèle n'a pas pu être chargé."
            )

            submit_btn = gr.Button("Lancer la Transcription", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Résultat de la Transcription", lines=20, interactive=False, show_copy_button=True)
            output_files = gr.File(label="Télécharger les fichiers (.zip)", interactive=False)

    submit_btn.click(
        fn=transcribe_and_diarize,
        inputs=[audio_input, language_dropdown, model_size_dropdown, diarization_checkbox],
        outputs=[output_text, output_files]
    )

# Lancer l'application
app.launch()

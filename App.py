import gradio as gr
import whisperx
import gc
import torch
import os
import tempfile
from typing import Optional, Dict, Any

class WhisperXTranscriber:
    def __init__(self):
        self.model = None
        self.align_model = None
        self.metadata = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16 if self.device == "cuda" else 8
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
    def load_model(self, model_size: str = "base"):
        """Charge le modèle WhisperX"""
        try:
            # Libère la mémoire si un modèle est déjà chargé
            if self.model is not None:
                del self.model
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Charge le nouveau modèle
            self.model = whisperx.load_model(
                model_size, 
                self.device, 
                compute_type=self.compute_type
            )
            return f"✅ Modèle {model_size} chargé avec succès sur {self.device}"
        except Exception as e:
            return f"❌ Erreur lors du chargement du modèle: {str(e)}"
    
    def transcribe_audio(self, audio_file, model_size: str = "base", 
                        align_speakers: bool = False, language: str = "auto"):
        """Transcrit un fichier audio"""
        if audio_file is None:
            return "❌ Veuillez sélectionner un fichier audio", ""
        
        try:
            # Charge le modèle si nécessaire
            if self.model is None:
                load_status = self.load_model(model_size)
                if "❌" in load_status:
                    return load_status, ""
            
            # Transcription initiale
            if language == "auto":
                audio = whisperx.load_audio(audio_file)
                result = self.model.transcribe(audio, batch_size=self.batch_size)
            else:
                audio = whisperx.load_audio(audio_file)
                result = self.model.transcribe(
                    audio, 
                    batch_size=self.batch_size, 
                    language=language
                )
            
            # Alignement temporel (optionnel)
            if align_speakers:
                try:
                    # Charge le modèle d'alignement
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result["language"], 
                        device=self.device
                    )
                    
                    # Aligne les segments
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio, 
                        self.device, 
                        return_char_alignments=False
                    )
                    
                    # Diarisation des locuteurs (optionnel)
                    # Nécessite HF_TOKEN dans les variables d'environnement
                    if os.getenv("HF_TOKEN"):
                        diarize_model = whisperx.DiarizationPipeline(
                            use_auth_token=os.getenv("HF_TOKEN"), 
                            device=self.device
                        )
                        diarize_segments = diarize_model(audio)
                        result = whisperx.assign_word_speakers(
                            diarize_segments, result
                        )
                    
                    # Libère la mémoire
                    del model_a
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Erreur lors de l'alignement: {e}")
            
            # Formate le résultat
            transcription_text = self.format_transcription(result)
            detailed_result = self.format_detailed_result(result)
            
            return f"✅ Transcription terminée (Langue détectée: {result.get('language', 'N/A')})", transcription_text, detailed_result
            
        except Exception as e:
            return f"❌ Erreur lors de la transcription: {str(e)}", "", ""
    
    def format_transcription(self, result) -> str:
        """Formate la transcription en texte simple"""
        if isinstance(result, dict) and "segments" in result:
            segments = result["segments"]
        else:
            segments = result
            
        return " ".join([segment["text"].strip() for segment in segments])
    
    def format_detailed_result(self, result) -> str:
        """Formate le résultat détaillé avec timestamps"""
        if isinstance(result, dict) and "segments" in result:
            segments = result["segments"]
        else:
            segments = result
            
        formatted = []
        for segment in segments:
            start_time = f"{segment.get('start', 0):.2f}s"
            end_time = f"{segment.get('end', 0):.2f}s"
            text = segment.get('text', '').strip()
            speaker = segment.get('speaker', '')
            
            if speaker:
                line = f"[{start_time} - {end_time}] {speaker}: {text}"
            else:
                line = f"[{start_time} - {end_time}] {text}"
            formatted.append(line)
        
        return "\n".join(formatted)

# Initialise le transcripteur
transcriber = WhisperXTranscriber()

def transcribe_interface(audio_file, model_size, align_speakers, language):
    """Interface pour Gradio"""
    status, transcription, detailed = transcriber.transcribe_audio(
        audio_file, model_size, align_speakers, language
    )
    return status, transcription, detailed

def load_model_interface(model_size):
    """Interface pour charger un modèle"""
    return transcriber.load_model(model_size)

# Interface Gradio
with gr.Blocks(title="🎙️ Transcripteur Audio WhisperX", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🎙️ Transcripteur Audio avec WhisperX
    
    Uploadez un fichier audio pour obtenir une transcription précise avec timestamps.
    
    **Formats supportés:** MP3, WAV, M4A, FLAC, OGG
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Configuration
            gr.Markdown("### ⚙️ Configuration")
            
            model_size = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                value="base",
                label="Taille du modèle",
                info="Plus grand = plus précis mais plus lent"
            )
            
            language = gr.Dropdown(
                choices=["auto", "fr", "en", "es", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                value="auto",
                label="Langue",
                info="Détection automatique ou langue spécifique"
            )
            
            align_speakers = gr.Checkbox(
                label="Alignement avancé",
                info="Active l'alignement temporel précis (plus lent)",
                value=False
            )
            
            load_btn = gr.Button("🔄 Charger le modèle", variant="secondary")
            
        with gr.Column(scale=2):
            # Upload et transcription
            gr.Markdown("### 📁 Fichier Audio")
            
            audio_input = gr.Audio(
                label="Sélectionnez votre fichier audio",
                type="filepath"
            )
            
            transcribe_btn = gr.Button("🎯 Transcrire", variant="primary", size="lg")
    
    # Résultats
    gr.Markdown("### 📝 Résultats")
    
    status_output = gr.Textbox(
        label="Statut",
        interactive=False,
        show_copy_button=True
    )
    
    with gr.Tab("Transcription Simple"):
        transcription_output = gr.Textbox(
            label="Transcription",
            lines=10,
            show_copy_button=True,
            interactive=False
        )
    
    with gr.Tab("Transcription Détaillée"):
        detailed_output = gr.Textbox(
            label="Transcription avec timestamps",
            lines=15,
            show_copy_button=True,
            interactive=False
        )
    
    # Événements
    load_btn.click(
        fn=load_model_interface,
        inputs=[model_size],
        outputs=[status_output]
    )
    
    transcribe_btn.click(
        fn=transcribe_interface,
        inputs=[audio_input, model_size, align_speakers, language],
        outputs=[status_output, transcription_output, detailed_output]
    )
    
    # Exemple d'utilisation
    gr.Markdown("""
    ### 💡 Conseils d'utilisation
    
    - **Modèles recommandés:** `base` pour un bon équilibre, `large-v3` pour la meilleure qualité
    - **Alignement avancé:** Améliore la précision des timestamps mais augmente le temps de traitement
    - **GPU recommandé:** Pour de meilleures performances avec les gros modèles
    - **Diarisation:** Définissez `HF_TOKEN` dans vos variables d'environnement pour identifier les locuteurs
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )

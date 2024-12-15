import streamlit as st
import whisper
import tempfile
from transformers import pipeline


st.title("Summerizer App")


# Upload audio
audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a"])
if audio_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name  # Store the temporary file path

# Load Models
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
st.sidebar.markdown("Models loaded")


if st.sidebar.button("Play Audio"):
    if temp_audio_path is not None:
        st.audio(temp_audio_path)
        st.sidebar.success("Playing audio" + str(temp_audio_path))
    else:
        st.sidebar.warning("Please upload an audio file first.")

if st.sidebar.button("Transcribe Audio"):
    if temp_audio_path is not None:
        st.sidebar.success("Transcribing Audio")
        transcription = whisper_model.transcribe(
            temp_audio_path
        )  # Use the stored temporary file path
        st.sidebar.success("Transcription completed")
        st.header("Transcript")
        st.markdown(transcription["text"])

        st.sidebar.success("Summerizing Audio")

        # Split the transcript into chunks of manageable size
        chunk_size = 4000  # Adjust the chunk size as needed
        text_chunks = [
            transcription["text"][i : i + chunk_size]
            for i in range(0, len(transcription["text"]), chunk_size)
        ]

        # Summarize each chunk and combine the summaries
        summaries = []
        for chunk in text_chunks:
            summary = summarizer(chunk, max_length=50, min_length=10, do_sample=False)
            summaries.append(summary[0]["summary_text"])

        # Combine the summaries into a single summary
        summary = " ".join(summaries)
        st.header("Summary")
        st.markdown(summary)
        st.sidebar.success("Summerys completed")
    else:
        st.sidebar.warning(
            "Please upload an audio file and play it first." + str(temp_audio_path)
        )

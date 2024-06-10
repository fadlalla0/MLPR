import streamlit as st
import torch
import torchaudio.transforms as T
import numpy as np
import joblib
import torchaudio

@st.cache_data
def load_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('efficient_netV2s.pt', map_location=device)
    model.eval()  # Set the model to evaluation mode
    return model, device

def pad_or_trim(audio, max_length):
    if audio.shape[1] > max_length:
        audio = audio[:, :max_length]
    else:
        padding = max_length - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, padding))
    return audio

def load_one_hot_encoder(encoder_path):
    return joblib.load(encoder_path)

def get_class_name(prediction, encoder):
    class_idx = np.argmax(prediction, axis=1)
    class_names = encoder.categories_[0]
    return class_names[class_idx][0]

st.title('Bird Classifier using Audio')
model, device = load_data()
sr = 32000
max_length = 480000
transform = T.MelSpectrogram(sr, n_fft=2028, n_mels=128, hop_length=max_length // (384 - 1),
                             window_fn=torch.hann_window, f_max=16000, f_min=20)
encoder = load_one_hot_encoder('one_hot_encoder.pkl')

uploaded_file = st.file_uploader("Choose an OGG file", type=["ogg"])
if uploaded_file is not None:
    if uploaded_file.type == 'audio/ogg':
        st.write('You have uploaded an OGG file.')

        # Load the audio file
        waveform, sample_rate = torchaudio.load(uploaded_file)
        
        # Resample if necessary
        if sample_rate != sr:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=sr)
            waveform = resample_transform(waveform)
        
        # Pad or trim the waveform to the max length
        waveform = pad_or_trim(waveform, max_length)
        
        # Apply the MelSpectrogram transformation
        mel_spectrogram = transform(waveform)

        # Model expects input of shape (batch_size, channels, n_mels, time)
        mel_spectrogram = mel_spectrogram.unsqueeze(0)  # Add batch dimension
        
        # Move the mel spectrogram to the same device as the model
        mel_spectrogram = mel_spectrogram.to(device)
        
        # Get prediction from the model
        with torch.no_grad():
            prediction = model(mel_spectrogram)
        
        # Convert prediction to human-readable format using OneHotEncoder
        predicted_class_name = get_class_name(prediction.cpu().numpy(), encoder)
        
        st.write('Prediction:', predicted_class_name)
        st.write('MelSpectrogram shape:', mel_spectrogram.shape)
        st.audio(uploaded_file, format='audio/ogg')

    else:
        st.write('Please upload an OGG file.')
else:
    st.write('No file uploaded yet.')

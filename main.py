import os
import librosa
import torch
from transformers import HubertModel,Wav2Vec2FeatureExtractor,Wav2Vec2CTCTokenizer,Wav2Vec2Processor
import time
import psutil
import socket
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import fire


 # Override de this dataloader class
class MusicDataset(Dataset):
    def __init__(self, music_segment_info, mp3_folder, target_sr, processor, model, concatenated_segments, tensors_folder):
        self.music_segment_info = music_segment_info
        self.length = len(music_segment_info)
        self.mp3_folder = mp3_folder
        self.target_sr = target_sr
        self.processor = processor
        self.model = model
        
        def __len__(self):
          return self.length # return how many musics there is
        
        def __getitem__(self, idx): # This class expects a X (data) and y (labels) to be returned.
            #However, we don't have them before finding hidden states -> model(input_values)
            music_index, segments_number = music_segment_info[idx]
            for i in range(segments_number):
                segment = concatenated_segments[music_index][i]
                with torch.no_grad():  # Apply torch.no_grad() to disable gradient computation
                    audio, _ = librosa.load(segment, sr=self.target_sr)  # Load the audio segment
                    input_values = self.processor(audio, sampling_rate=self.target_sr, return_tensors="pt").input_values
                    hidden_states = self.model(input_values).last_hidden_state
                concatenated_hidden_states.append(hidden_states)  # Append hidden states of each segment
            concatenated_hidden_states = torch.cat(concatenated_hidden_states, dim=1)
            tensor_file = os.path.join(tensors_folder, f"tensor{idx}.pt")
            torch.save(concatenated_hidden_states, tensor_file)

            length = len(concatenated_hidden_states[0][0][:-1]) + 1
            reshaped_tensor = concatenated_hidden_states.view(-1, length)

            return reshaped_tensor # X, y



def process_audio(mp3_folder, tensors_folder):
    # Get the hostname of the machine
    hostname = socket.gethostname()

    # Get the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # List all files directly in the folder (ignoring ".")
    mp3_files = [f for f in os.listdir(mp3_folder) if ("left" in f.lower() or "right" in f.lower())]

    # Adding the full path to the mp3 files
    mp3_files = [mp3_folder + '/' + f for f in mp3_files]

    # Configuration of the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Specify the desired target sample rate
    target_sr = 16000

    # Segment duration in seconds
    segment_duration = 30

    # Array to define list of musics and number of segments
    music_segment_info = []
    segments = []
    concatenated_segments = []
    def segment_audio(audio, i):

        num_segments = int(audio.shape[0] / (target_sr * segment_duration)) # Calculate number of segments
        print(f"\nAudio {i}. Number of segments: {num_segments}\n")

        music_segment_info.append((i, num_segments))
        print(len(music_segment_info))

        for segment_index in range(num_segments):
            start_index = segment_index * target_sr * segment_duration
            end_index = start_index + target_sr * segment_duration

            segment_audio = audio[start_index:end_index]

            # Convert it to a 2D array so we can convert it to a torch tensor
            segment_audio = segment_audio.reshape(1, -1)

            # Convert numpy array to torch tensor (if needed)
            segment_tensor = torch.from_numpy(segment_audio).float()

            segments.append(segment_tensor)

        concatenated_segments.append(segments)

    start_time = time.time()  # Record the starting time

    for i in range(len(mp3_files)):

        audio, sample_rate = librosa.load(mp3_files[i], sr=target_sr)
        segment_audio(audio, i)
    end_time = time.time()  # Record the ending time of segmentation
    execution_time = end_time - start_time
    print("Total execution time of segmentation part: ", execution_time, " seconds")
    # Now that we have the segments, we can use the Hubert model to extract the hidden states


    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base")

    processor = Wav2Vec2Processor(feature_extractor,tokenizer)
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    # The processor handles tokenization and feature extraction

    music_dataset = MusicDataset(music_segment_info, mp3_folder, target_sr, processor, model, concatenated_segments, tensors_folder)
    batch_size = 64
    dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)
    
    for batch in dataloader:
        print(batch)

    # Print memory allocated by CUDA
    if device.type == 'cuda':
        print("Memory allocated by CUDA:", torch.cuda.memory_allocated(device=device) / (1024 * 1024), "MB")
        print("Device properties:", torch.cuda.get_device_properties(device=device))
    else:
        # Print CPU usage
        print("CPU usage (percentage):", psutil.cpu_percent())
        # Print virtual memory usage
        virtual_memory = psutil.virtual_memory()
        print("Total virtual memory:", virtual_memory.total / (1024 * 1024), "MB")
        print("Available virtual memory:", virtual_memory.available / (1024 * 1024), "MB")

    end_time2 = time.time()  # Record the ending time
    execution_time2 = end_time2 - end_time
    print("Total execution time of finding hidden units: ", execution_time2, " seconds")


    # Get memory usage

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss  # in bytes
    print("Memory usage:", memory_usage / (1024 * 1024), "MB")

    # Print hostname and current date
    print("Hostname:", hostname)
    print("Date:", current_date)


if __name__ == "__main__":
    fire.Fire(process_audio)
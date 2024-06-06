import random
import midiutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import librosa
import sounddevice as sd
from midi2audio import FluidSynth
POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
MELODY_LENGTH = 16  



def midi_to_audio(midi_file, sample_rate=22050):
    fs=FluidSynth()
    fs.midi_to_audio('midi_file','generated_song.wav')
    
def genre_classifier():
    data=pd.read_excel("genres.xlsx")
    X=data.drop("label",axis=1)
    y=data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    
    return model

def feature_extract(audio_file):

    y, sr = librosa.load(audio_file)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = chroma_stft.mean(axis=1)  # Calculate mean along columns
    chroma_stft_var = chroma_stft.var(axis=1)    # Calculate variance along columns


    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    rms_var = rms.var()

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = spectral_centroid.mean()
    spectral_centroid_var = spectral_centroid.var()


    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = spectral_bandwidth.mean()
    spectral_bandwidth_var = spectral_bandwidth.var()


    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()


    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = zero_crossing_rate.mean()
    zero_crossing_rate_var = zero_crossing_rate.var()


    harmony, perceptr = librosa.effects.hpss(y)
    harmony_mean = harmony.mean()
    harmony_var = harmony.var()
    perceptr_mean = perceptr.mean()
    perceptr_var = perceptr.var()


    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features=[chroma_stft_mean,chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,perceptr_mean,perceptr_var,tempo]
    return features

def fitness(melody,genre,tempo):
    midi = melody_to_midi(melody, tempo)
    with open("generated_song.mid", "wb") as midi_file:
        midi.writeFile(midi_file)
    midi_to_audio('generated_song.mid', 'generated_song.wav')
    features = feature_extract('generated_song.wav')
    genre_probabilities = genre_classifier.predict_proba([features])[0]
    genre_probability = genre_probabilities[genre_classifier.classes_.tolist().index(genre)]
    return genre_probability
    
def generate_population(pop_size, melody_length,MIN_NOTE,MAX_NOTE):
    return [[random.randint(MIN_NOTE, MAX_NOTE) for _ in range(melody_length)] for _ in range(pop_size)]

def select(population, fitness_fn):
    selected = sorted(population, key=fitness_fn, reverse=True)
    return selected


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(melody,MIN_NOTE,MAX_NOTE):
    mutated_melody = []
    for note in melody:
        if random.random() < MUTATION_RATE:
            mutated_melody.append(random.randint(MIN_NOTE, MAX_NOTE))
        else:
            mutated_melody.append(note)
    return mutated_melody


def melody_to_midi(melody,TEMPO):
    track = 0
    channel = 0
    time = 0
    duration = 1  
    tempo = TEMPO
    volume = 100

    midi = midiutil.MIDIFile(1, adjust_origin=True)
    midi.addTempo(track, time, tempo)

    for note in melody:
        pitch = note
        midi.addNote(track, channel, pitch, time, duration, volume)
        time += duration

    return midi


def genetic_algorithm(population_size, num_generations,MIN_NOTE,MAX_NOTE,genre,tempo):
    population = generate_population(population_size, MELODY_LENGTH,MIN_NOTE,MAX_NOTE)
    for _ in range(num_generations):
        population = sorted(population, key=lambda x: fitness(x,genre,tempo), reverse=True)
        selected = select(population, fitness)
        offspring = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1,MIN_NOTE,MAX_NOTE)
            child2 = mutate(child2,MIN_NOTE,MAX_NOTE)
            offspring.extend([child1, child2])
        population = offspring[:population_size]
    return population


def main():
    genre=input("Enter the genre:")
    if genre=="Jazz":
        MIN_NOTE=36
        MAX_NOTE=79
        TEMPO=random.randint(130,170)
    elif genre=="EDM":
        MIN_NOTE=60
        MAX_NOTE=127
        TEMPO=random.randint(100,130)
    elif genre=="Hip Hop":
        MIN_NOTE=60
        MAX_NOTE=96
        TEMPO=random.randint(60,100)
    elif genre=="Pop":
        MIN_NOTE=60
        MAX_NOTE=108
        TEMPO=random.randint(100,130)
    elif genre=="Rock":
        MIN_NOTE=40
        MAX_NOTE=108
        TEMPO=random.randint(100,130)
    elif genre=="Classical":
        MIN_NOTE=36
        MAX_NOTE=127
        TEMPO=random.randint(30,60)
    final_population = genetic_algorithm(POPULATION_SIZE, NUM_GENERATIONS,MIN_NOTE,MAX_NOTE,genre,TEMPO)
    fitnesses=[]
    # for x in final_population:
    #     midi=melody_to_midi(x,TEMPO)
    #     with open("generated_song.mid", "wb") as midi_file:
    #         midi.writeFile(midi_file)
    #     midi_to_audio('generated_song.mid')
    #     fitnesses.append(fitness("output.wav",genre,TEMPO))
    best_melody = max(final_population, key=fitness)
    print("Best Melody:", best_melody)
    print("Fitness Score:", fitness(best_melody))
    midi = melody_to_midi(best_melody,TEMPO)
    with open("generated_song.mid", "wb") as midi_file:
        midi.writeFile(midi_file)
        
main()
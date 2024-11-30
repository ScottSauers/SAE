import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, regularizers
import seaborn as sns
import time
from datetime import datetime

class SparseAutoencoder:
    def __init__(self, input_dim, encoding_dim=32, sparsity_reg=0.01):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing Sparse Autoencoder...")
        print(f"├── Input dimension: {input_dim}")
        print(f"├── Encoding dimension: {encoding_dim}")
        print(f"└── Sparsity regularization: {sparsity_reg}")
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.sparsity_reg = sparsity_reg
        self.model = self._build_model()
        
    def _build_model(self):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building model architecture...")
        
        print("├── Creating input layer...")
        input_layer = layers.Input(shape=(self.input_dim,))
        
        print("├── Building encoder with regularization...")
        encoded = layers.Dense(
            self.encoding_dim,
            activation='relu',
            activity_regularizer=regularizers.l1(self.sparsity_reg),
            name='encoder'
        )(input_layer)
        
        print("├── Building decoder layer...")
        decoded = layers.Dense(
            self.input_dim,
            activation='sigmoid',
            name='decoder'
        )(encoded)
        
        print("├── Assembling full autoencoder...")
        autoencoder = Model(input_layer, decoded)
        
        print("├── Creating separate encoder model...")
        self.encoder = Model(input_layer, encoded)
        
        print("└── Compiling model with Adam optimizer...")
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder

class ProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting epoch {epoch + 1}...")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"└── Epoch {epoch + 1} complete - Loss: {logs['loss']:.6f}, MAE: {logs['mae']:.6f}")
        if 'val_loss' in logs:
            print(f"    └── Validation - Loss: {logs['val_loss']:.6f}, MAE: {logs['val_mae']:.6f}")

def preprocess_text(file_path, max_features=1000):
    """Read and preprocess text data with detailed progress reporting."""
    start_time = time.time()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting text preprocessing pipeline...")
    print(f"├── Reading file: {file_path}")
    
    with open(file_path, 'r') as file:
        text = file.read()
    
    print(f"├── Raw text loaded: {len(text):,} characters")
    
    # Split into documents
    print("├── Splitting text into documents...")
    documents = [p.strip() for p in text.split('\n') if p.strip()]
    print(f"│   └── Found {len(documents):,} documents")
    
    print(f"├── Initializing TF-IDF vectorizer (max_features={max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2
    )
    
    print("├── Converting text to TF-IDF vectors...")
    X = vectorizer.fit_transform(documents)
    vocab_size = len(vectorizer.get_feature_names_out())
    print(f"│   ├── Vocabulary size: {vocab_size:,} terms")
    print(f"│   └── Matrix shape: {X.shape}")
    
    print("├── Converting sparse matrix to dense...")
    X = X.toarray()
    
    print("├── Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    elapsed = time.time() - start_time
    print(f"└── Preprocessing complete! (Time: {elapsed:.2f}s)")
    
    return X_scaled, vectorizer, documents, scaler

def train_autoencoder(X, encoding_dim=32, epochs=100, batch_size=32):
    """Train autoencoder with detailed progress reporting."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Initializing autoencoder training...")
    print(f"├── Data shape: {X.shape}")
    print(f"├── Encoding dimension: {encoding_dim}")
    print(f"├── Epochs: {epochs}")
    print(f"├── Batch size: {batch_size}")
    
    autoencoder = SparseAutoencoder(X.shape[1], encoding_dim)
    
    # Create callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    progress_callback = ProgressCallback()
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training...")
    history = autoencoder.model.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stopping, progress_callback]
    )
    
    return autoencoder, history

def analyze_features(autoencoder, X, vectorizer):
    """Analyze features with detailed progress reporting."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting feature analysis...")
    
    print("├── Getting encoder layer weights...")
    encoder_weights = autoencoder.model.get_layer('encoder').get_weights()[0]
    print(f"│   └── Weight matrix shape: {encoder_weights.shape}")
    
    print("├── Computing encoded features...")
    encoded_features = autoencoder.encoder.predict(X, verbose=0)
    print(f"│   └── Encoded shape: {encoded_features.shape}")
    
    print("├── Calculating mean activations...")
    mean_activations = np.mean(encoded_features, axis=0)
    
    print("├── Analyzing feature-word relationships...")
    feature_analysis = []
    vocab = vectorizer.get_feature_names_out()
    
    total_features = encoder_weights.shape[1]
    for i in range(total_features):
        if i % 5 == 0:  # Progress update every 5 features
            print(f"│   ├── Analyzing feature {i+1}/{total_features} ({(i+1)/total_features*100:.1f}%)")
        
        feature_weights = encoder_weights[:, i]
        top_word_indices = np.argsort(np.abs(feature_weights))[-5:]
        top_words = [(vocab[idx], feature_weights[idx]) for idx in top_word_indices]
        
        feature_analysis.append({
            'feature_id': i,
            'mean_activation': mean_activations[i],
            'top_words': top_words
        })
    
    print("└── Feature analysis complete!")
    return feature_analysis, encoded_features

def visualize_results(history, encoded_features, feature_analysis):
    """Create visualizations with detailed progress reporting."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Generating visualizations...")
    
    print("├── Creating figure...")
    plt.figure(figsize=(15, 10))
    
    print("├── Plotting training history...")
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    print("├── Generating feature activation heatmap...")
    plt.subplot(2, 2, 2)
    sns.heatmap(
        encoded_features[:min(50, encoded_features.shape[0])].T,
        cmap='viridis'
    )
    plt.title('Feature Activations (First 50 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    
    print("├── Creating mean activation plot...")
    plt.subplot(2, 2, 3)
    mean_activations = [f['mean_activation'] for f in feature_analysis]
    plt.bar(range(len(mean_activations)), mean_activations)
    plt.title('Mean Feature Activations')
    plt.xlabel('Feature')
    plt.ylabel('Mean Activation')
    
    print("├── Adjusting layout...")
    plt.tight_layout()
    
    print("├── Saving visualization to 'autoencoder_analysis.png'...")
    plt.savefig('autoencoder_analysis.png', dpi=300)
    plt.close()
    
    print("└── Visualization complete!")

def main():
    total_start_time = time.time()
    print(f"\n{'='*80}")
    print(f"SPARSE AUTOENCODER ANALYSIS - Started at {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*80}")
    
    # Parameters
    max_features = 1000
    encoding_dim = 32
    epochs = 100
    
    try:
        # Preprocessing
        X, vectorizer, documents, scaler = preprocess_text(
            'L405BB_wake.txt',
            max_features=max_features
        )
        
        # Training
        autoencoder, history = train_autoencoder(
            X,
            encoding_dim=encoding_dim,
            epochs=epochs
        )
        
        # Analysis
        feature_analysis, encoded_features = analyze_features(
            autoencoder, X, vectorizer
        )
        
        # Visualization
        visualize_results(history, encoded_features, feature_analysis)
        
        # Results
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ANALYSIS RESULTS")
        print("═" * 50)
        print("\nTop activated features and their associated words:")
        for idx, feat in enumerate(sorted(feature_analysis, 
                                        key=lambda x: x['mean_activation'], 
                                        reverse=True)[:10]):
            print(f"\n{idx+1}. Feature {feat['feature_id']} "
                  f"(activation: {feat['mean_activation']:.4f}):")
            for word, weight in feat['top_words']:
                print(f"   {word}: {weight:.4f}")
        
        total_time = time.time() - total_start_time
        print(f"\n{'='*80}")
        print(f"Analysis complete! Total time: {total_time:.2f} seconds")
        print(f"Check 'autoencoder_analysis.png' for visualizations")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()

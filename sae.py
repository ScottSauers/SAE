import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, regularizers
import seaborn as sns

class SparseAutoencoder:
    def __init__(self, input_dim, encoding_dim=32, sparsity_reg=0.01):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.sparsity_reg = sparsity_reg
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(
            self.encoding_dim,
            activation='relu',
            activity_regularizer=regularizers.l1(self.sparsity_reg),
            name='encoder'
        )(input_layer)
        
        # Decoder
        decoded = layers.Dense(
            self.input_dim,
            activation='sigmoid',
            name='decoder'
        )(encoded)
        
        # Full autoencoder
        autoencoder = Model(input_layer, decoded)
        
        # Separate encoder model for feature extraction
        self.encoder = Model(input_layer, encoded)
        
        # Compile model
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder

def preprocess_text(file_path, max_features=1000):
    """Read and preprocess text data."""
    # Read text file
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Split into documents (adjust splitting logic based on your data)
    documents = [p.strip() for p in text.split('\n') if p.strip()]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=2
    )
    X = vectorizer.fit_transform(documents).toarray()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, vectorizer, documents, scaler

def train_autoencoder(X, encoding_dim=32, epochs=100, batch_size=32):
    """Train the autoencoder."""
    # Initialize and train model
    autoencoder = SparseAutoencoder(X.shape[1], encoding_dim)
    
    history = autoencoder.model.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    return autoencoder, history

def analyze_features(autoencoder, X, vectorizer):
    """Analyze learned features and their relationships to input words."""
    # Get encoder layer weights
    encoder_weights = autoencoder.model.get_layer('encoder').get_weights()[0]
    
    # Get feature activations
    encoded_features = autoencoder.encoder.predict(X)
    mean_activations = np.mean(encoded_features, axis=0)
    
    # Analyze feature-word relationships
    feature_analysis = []
    vocab = vectorizer.get_feature_names_out()
    
    for i in range(encoder_weights.shape[1]):
        # Get weights connecting to this feature
        feature_weights = encoder_weights[:, i]
        
        # Find top words for this feature
        top_word_indices = np.argsort(np.abs(feature_weights))[-5:]
        top_words = [(vocab[idx], feature_weights[idx]) for idx in top_word_indices]
        
        feature_analysis.append({
            'feature_id': i,
            'mean_activation': mean_activations[i],
            'top_words': top_words
        })
    
    return feature_analysis, encoded_features

def visualize_results(history, encoded_features, feature_analysis):
    """Create visualizations of training and learned features."""
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training history
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot 2: Feature activations heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(
        encoded_features[:min(50, encoded_features.shape[0])].T,
        cmap='viridis'
    )
    plt.title('Feature Activations (First 50 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    
    # Plot 3: Mean feature activations
    plt.subplot(2, 2, 3)
    mean_activations = [f['mean_activation'] for f in feature_analysis]
    plt.bar(range(len(mean_activations)), mean_activations)
    plt.title('Mean Feature Activations')
    plt.xlabel('Feature')
    plt.ylabel('Mean Activation')
    
    plt.tight_layout()
    plt.savefig('autoencoder_analysis.png')
    plt.close()

def main():
    # Parameters
    max_features = 1000
    encoding_dim = 32
    epochs = 100
    
    print("Loading and preprocessing text data...")
    X, vectorizer, documents, scaler = preprocess_text(
        'L405BB_wake.txt',
        max_features=max_features
    )
    print(f"Processed {len(documents)} documents with {X.shape[1]} features")
    
    print("\nTraining autoencoder...")
    autoencoder, history = train_autoencoder(
        X,
        encoding_dim=encoding_dim,
        epochs=epochs
    )
    
    print("\nAnalyzing learned features...")
    feature_analysis, encoded_features = analyze_features(
        autoencoder, X, vectorizer
    )
    
    print("\nGenerating visualizations...")
    visualize_results(history, encoded_features, feature_analysis)
    
    print("\nTop activated features and their associated words:")
    for feat in sorted(feature_analysis, key=lambda x: x['mean_activation'], reverse=True)[:10]:
        print(f"\nFeature {feat['feature_id']} (activation: {feat['mean_activation']:.4f}):")
        for word, weight in feat['top_words']:
            print(f"  {word}: {weight:.4f}")
    
    print("\nAnalysis complete! Check 'autoencoder_analysis.png' for visualizations.")

if __name__ == "__main__":
    main()

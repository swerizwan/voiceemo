import wav2vec2

def extract_voice_features(raw_voice_snippets):
    """
    Extracts features from raw voice snippets using the Wav2Vec 2.0 model.
    
    Args:
        raw_voice_snippets (list): A list of raw voice snippets.
        
    Returns:
        features: Extracted features from the raw voice snippets.
    """
    # Instantiate the Wav2Vec 2.0 model for feature extraction
    wav2vec_model = wav2vec2.Wav2Vec2Model()
    
    # Extract features from each raw voice snippet using the model
    features = []
    for snippet in raw_voice_snippets:
        # Call the extract_features method of the model to obtain features for the current snippet
        snippet_features = wav2vec_model.extract_features(snippet)
        # Append the features of the current snippet to the list of features
        features.append(snippet_features)
    
    # Return the extracted features for all voice snippets
    return features

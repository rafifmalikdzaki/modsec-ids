import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
path = 'results/tokenizer.pkl'
if os.path.exists(path):
    with open(path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Tokenizer loaded.")
    print(f"Filters: '{tokenizer.filters}'")
    print(f"Lower: {tokenizer.lower}")
    print(f"Split: '{tokenizer.split}'")
    print(f"Char Level: {tokenizer.char_level}")
    
    test_text = "GET /profile.php?name=<img src=x onerror=alert(1)>"
    sequences = tokenizer.texts_to_sequences([test_text])
    
    print(f"\nInput: {test_text}")
    print(f"Sequence: {sequences}")
    
    # Check specific tokens
    keywords = ['union', 'select', 'alert', 'script', '<', '>', '=', 'get', '/', '?', '-', "'"]
    print("\nChecking keywords in vocabulary:")
    for word in keywords:
        idx = tokenizer.word_index.get(word)
        print(f"  '{word}': {idx}")
    
    if not sequences[0]:
        print("\n❌ WARNING: Sequence is empty! The tokenizer filtered everything out.")
else:
    print("Tokenizer not found.")


#!/usr/bin/env python3
"""
Atomic Test Tool for ModSec-IDS

This script allows you to test a single log line or attack sample against the
Semantic LSTM model to verify detection capabilities.

Usage:
    python tools/test_sample.py "GET /wp-admin/admin-ajax.php?action=revslider_show_image&img=../wp-config.php HTTP/1.1"
    python tools/test_sample.py --interactive
"""

import argparse
import sys
import os

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from detectors.tensorflow_semantic_inference import TensorFlowSemanticInference
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Error importing detector: {e}")
    MODEL_AVAILABLE = False

def test_sample(detector, sample):
    """Run prediction on a single sample."""
    print(f"\n📝 Analyzing Sample:")
    print(f"   {sample}")
    print(f"DEBUG: Raw sample sent to detector.predict: {sample[:100]}...")
    
    try:
        label, conf, probs = detector.predict(sample)
        
        # Determine status icon and color
        if label == 'normal':
            status = "🟢 SAFE"
        else:
            status = f"🔴 ATTACK ({label.upper()})"
            
        print(f"\n🧠 Prediction Result:")
        print(f"   Status:     {status}")
        print(f"   Confidence: {conf:.2%}")
        
        print("\n📊 Class Probabilities:")
        # Sort probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for cls, prob in sorted_probs:
            if prob > 0.01: # Show only significant probabilities
                bar = "█" * int(prob * 20)
                print(f"   {cls:20s}: {bar} {prob:.2%}")
                
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def interactive_mode(detector):
    """Run in interactive loop."""
    print("\n🔵 Interactive Mode (Ctrl+C to exit)")
    print("   Type a log line and press Enter to test.")
    
    try:
        while True:
            sample = input("\n> ").strip()
            if sample:
                test_sample(detector, sample)
    except KeyboardInterrupt:
        print("\n\nExiting...")

def main():
    parser = argparse.ArgumentParser(description="Atomic Test Tool for ModSec-IDS")
    parser.add_argument('sample', nargs='?', help="The log line or attack string to test")
    parser.add_argument('--interactive', '-i', action='store_true', help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if not MODEL_AVAILABLE:
        print("❌ TensorFlow Semantic Model could not be imported.")
        print("   Make sure you are in the project root and dependencies are installed.")
        return 1
        
    print("🔄 Loading Semantic Model...")
    detector = TensorFlowSemanticInference()
    
    if args.interactive:
        interactive_mode(detector)
    elif args.sample:
        test_sample(detector, args.sample)
    else:
        # No arguments provided, show help
        parser.print_help()
        
        # Run a default example
        print("\n" + "="*50)
        print("Running default example:")
        example = "GET /login.php?user=admin' OR '1'='1 HTTP/1.1"
        test_sample(detector, example)

if __name__ == "__main__":
    main()

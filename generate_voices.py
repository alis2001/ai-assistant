#!/usr/bin/env python3
"""
Generate Voice Prompts for Call Center
Run this script to create all voice prompts
"""

import subprocess
import sys

def install_requirements():
    """Install required packages"""
    try:
        import gtts
        print("✅ gTTS already installed")
    except ImportError:
        print("📦 Installing gTTS...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gtts"], check=True)
        print("✅ gTTS installed")

def generate_voices():
    """Generate all voice prompts"""
    print("🎤 Generating Call Center Voice Prompts")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Run voice prompts generator
    try:
        from voice_prompts import create_voice_prompts, list_prompts, PROMPTS
        
        print("Creating Italian voice prompts...")
        create_voice_prompts()
        
        print("\n✅ Voice prompts generated successfully!")
        print("📁 Directory: voice_prompts/")
        print("📋 Available prompts:")
        
        for prompt_id in list_prompts():
            text = PROMPTS.get(prompt_id, "Unknown")
            print(f"   • {prompt_id}: {text}")
        
        print("\n🎯 Voice prompts ready for call center integration!")
        
    except Exception as e:
        print(f"❌ Error generating voice prompts: {e}")
        return False
    
    return True

if __name__ == "__main__":
    generate_voices()

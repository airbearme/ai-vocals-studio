#!/usr/bin/env python3
"""
Simple validation script for precision voice cloning system
"""

import sys
import os

def validate_imports():
    """Validate that all modules can be imported"""
    print("🎤 Validating Precision Voice Cloning System Imports")
    print("=" * 50)
    
    modules = [
        ('advanced_audio_processor', 'AdvancedAudioProcessor'),
        ('voice_feature_extractor', 'VoiceFeatureExtractor'),
        ('data_augmentation', 'VoiceDataAugmentor'),
        ('advanced_trainer', 'AdvancedVoiceTrainer'),
        ('voice_quality_assurance', 'VoiceQualityAssurance'),
        ('precision_voice_cloning_system', 'PrecisionVoiceCloningSystem')
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} - OK")
            passed += 1
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} - FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"⚠️ {module_name}.{class_name} - WARNING: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All modules imported successfully!")
        return True
    else:
        print("⚠️ Some modules failed to import. Check dependencies.")
        return False

def validate_file_structure():
    """Validate that all files exist"""
    print("\n📁 Validating File Structure")
    print("=" * 50)
    
    required_files = [
        'advanced_audio_processor.py',
        'voice_feature_extractor.py',
        'data_augmentation.py',
        'advanced_trainer.py',
        'voice_quality_assurance.py',
        'precision_voice_cloning_system.py',
        'app_modern.py'
    ]
    
    passed = 0
    failed = 0
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename} - exists")
            passed += 1
        else:
            print(f"❌ {filename} - missing")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    return failed == 0

def validate_integration():
    """Validate integration with main app"""
    print("\n🔗 Validating Integration")
    print("=" * 50)
    
    try:
        # Check if app_modern has the precision system integration
        with open('app_modern.py', 'r') as f:
            content = f.read()
            
        checks = [
            ('PrecisionVoiceCloningSystem import', 'from precision_voice_cloning_system import PrecisionVoiceCloningSystem'),
            ('Precision system initialization', 'self.precision_system = PrecisionVoiceCloningSystem'),
            ('Precision cloning button', 'start_precision_cloning'),
            ('Precision cloning thread', '_precision_cloning_thread')
        ]
        
        passed = 0
        failed = 0
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"✅ {check_name} - found")
                passed += 1
            else:
                print(f"❌ {check_name} - not found")
                failed += 1
        
        print("=" * 50)
        print(f"Results: {passed} passed, {failed} failed")
        
        return failed == 0
        
    except Exception as e:
        print(f"❌ Integration validation failed: {e}")
        return False

def main():
    """Run all validations"""
    print("🚀 PRECISION VOICE CLONING SYSTEM VALIDATION")
    print("=" * 50)
    
    # Validate file structure
    files_ok = validate_file_structure()
    
    # Validate imports (this will fail if dependencies are missing)
    try:
        imports_ok = validate_imports()
    except Exception as e:
        print(f"⚠️ Import validation skipped due to: {e}")
        imports_ok = False
    
    # Validate integration
    integration_ok = validate_integration()
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"File Structure: {'✅ OK' if files_ok else '❌ FAILED'}")
    print(f"Module Imports: {'✅ OK' if imports_ok else '⚠️ SKIPPED/FAILED'}")
    print(f"Integration: {'✅ OK' if integration_ok else '❌ FAILED'}")
    
    if files_ok and integration_ok:
        print("\n🎉 PRECISION SYSTEM STRUCTURALLY VALID!")
        print("💡 Note: Full functionality requires installing dependencies:")
        print("   pip install numpy librosa soundfile scipy")
        return True
    else:
        print("\n❌ VALIDATION FAILED - CHECK ISSUES ABOVE")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
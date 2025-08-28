#!/usr/bin/env python3
"""
Test script to verify all functionality works correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from app import calculate_data_quality_score, generate_manual_analysis

def test_data_loader():
    """Test the DataLoader functionality"""
    print("🔍 Testing DataLoader...")
    loader = DataLoader()
    
    # Test with sample data
    test_data = """name,age,city
John,25,NYC
Jane,30,LA
Bob,35,Chicago"""
    
    from io import StringIO
    df = loader.load_csv(StringIO(test_data))
    
    assert len(df) == 3, "Should load 3 rows"
    assert len(df.columns) == 3, "Should have 3 columns"
    print("✅ DataLoader works correctly")
    return df

def test_data_quality_score(df):
    """Test data quality scoring"""
    print("📊 Testing Data Quality Score...")
    score = calculate_data_quality_score(df)
    assert 0 <= score <= 100, f"Score should be between 0-100, got {score}"
    print(f"✅ Data Quality Score: {score:.1f}/100")
    return score

def test_manual_analysis(df):
    """Test manual analysis generation"""
    print("🤖 Testing Manual Analysis...")
    analysis = generate_manual_analysis(df, "Test Dataset")
    assert "DATA QUALITY SCORE" in analysis, "Should contain quality score"
    assert "KEY INSIGHTS" in analysis, "Should contain insights"
    assert "RECOMMENDATIONS" in analysis, "Should contain recommendations"
    print("✅ Manual Analysis generated successfully")
    return True

def test_all():
    """Run all tests"""
    print("🚀 Starting Comprehensive Testing...")
    print("="*50)
    
    try:
        # Test 1: Data Loading
        df = test_data_loader()
        
        # Test 2: Data Quality
        score = test_data_quality_score(df)
        
        # Test 3: Manual Analysis
        test_manual_analysis(df)
        
        print("="*50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Application is ready for use")
        print(f"✅ Access your app at: http://localhost:8502")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    test_all()

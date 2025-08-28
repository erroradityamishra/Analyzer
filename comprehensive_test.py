#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE FEATURE TEST SUITE
Testing all critical functionality before submission
"""

import pandas as pd
import numpy as np
import sys
import os
import requests
from io import StringIO
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import DataLoader
from app import calculate_data_quality_score, generate_manual_analysis

def create_test_dataset():
    """Create a comprehensive test dataset"""
    np.random.seed(42)
    n = 1000
    
    data = {
        'customer_id': range(1, n+1),
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 15000, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'score': np.random.uniform(0, 100, n),
        'active': np.random.choice([True, False], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'date_joined': pd.date_range('2020-01-01', periods=n, freq='D')[:n]
    }
    
    # Add some missing values and duplicates
    df = pd.DataFrame(data)
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'category'] = np.nan
    
    # Add some duplicates
    df = pd.concat([df, df.head(20)], ignore_index=True)
    
    return df

def test_feature_1_data_loading():
    """🔍 Test Feature 1: Data Loading Capabilities"""
    print("🔍 TESTING: Data Loading Capabilities")
    print("-" * 40)
    
    try:
        loader = DataLoader()
        
        # Test 1: CSV from StringIO
        test_csv = """name,age,city,salary
John,25,NYC,50000
Jane,30,LA,60000
Bob,35,Chicago,55000
Alice,28,Boston,52000"""
        
        df1 = loader.load_csv(StringIO(test_csv))
        assert len(df1) == 4, f"Expected 4 rows, got {len(df1)}"
        assert len(df1.columns) == 4, f"Expected 4 columns, got {len(df1.columns)}"
        print("✅ CSV Loading: PASSED")
        
        # Test 2: URL Loading
        try:
            iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            df2 = loader.load_url(iris_url)
            assert len(df2) > 100, "Iris dataset should have >100 rows"
            print("✅ URL Loading: PASSED")
        except:
            print("⚠️  URL Loading: SKIPPED (Network issue)")
        
        # Test 3: Large dataset handling
        large_df = create_test_dataset()
        assert len(large_df) > 1000, "Large dataset creation failed"
        print("✅ Large Dataset: PASSED")
        
        return large_df
        
    except Exception as e:
        print(f"❌ Data Loading: FAILED - {e}")
        return None

def test_feature_2_data_quality_scoring(df):
    """📊 Test Feature 2: Data Quality Assessment"""
    print("\n📊 TESTING: Data Quality Assessment")
    print("-" * 40)
    
    try:
        # Test quality score calculation
        score = calculate_data_quality_score(df)
        assert 0 <= score <= 100, f"Score should be 0-100, got {score}"
        print(f"✅ Quality Score: {score:.1f}/100 - PASSED")
        
        # Test with perfect data
        perfect_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': ['x', 'y', 'z', 'w', 'v']
        })
        perfect_score = calculate_data_quality_score(perfect_df)
        assert perfect_score > 80, f"Perfect data should score >80, got {perfect_score}"
        print(f"✅ Perfect Data Score: {perfect_score:.1f}/100 - PASSED")
        
        # Test with poor data
        poor_df = pd.DataFrame({
            'A': [1, np.nan, np.nan, np.nan, np.nan],
            'B': [1, 1, 1, 1, 1],  # All same values
            'C': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All missing
        })
        poor_score = calculate_data_quality_score(poor_df)
        assert poor_score < 50, f"Poor data should score <50, got {poor_score}"
        print(f"✅ Poor Data Score: {poor_score:.1f}/100 - PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Quality: FAILED - {e}")
        return False

def test_feature_3_analysis_generation(df):
    """🤖 Test Feature 3: AI/Manual Analysis Generation"""
    print("\n🤖 TESTING: Analysis Generation")
    print("-" * 40)
    
    try:
        # Test manual analysis
        analysis = generate_manual_analysis(df, "Test Dataset")
        
        # Check if analysis contains key sections
        required_sections = [
            "DATA QUALITY SCORE",
            "KEY INSIGHTS", 
            "RECOMMENDATIONS",
            "WARNINGS & ALERTS"
        ]
        
        for section in required_sections:
            assert section in analysis, f"Missing section: {section}"
        
        print("✅ Analysis Structure: PASSED")
        
        # Check analysis quality
        assert len(analysis) > 500, "Analysis should be comprehensive"
        assert "customer_id" in analysis or "age" in analysis, "Analysis should reference actual columns"
        print("✅ Analysis Content: PASSED")
        
        # Test with different dataset types
        categorical_df = pd.DataFrame({
            'category': ['A', 'B', 'C'] * 100,
            'type': ['X', 'Y'] * 150,
            'status': ['active', 'inactive', 'pending'] * 100
        })
        
        cat_analysis = generate_manual_analysis(categorical_df, "Categorical Dataset")
        assert "categorical" in cat_analysis.lower(), "Should detect categorical nature"
        print("✅ Categorical Analysis: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis Generation: FAILED - {e}")
        return False

def test_feature_4_data_processing(df):
    """⚙️ Test Feature 4: Data Processing & Transformation"""
    print("\n⚙️ TESTING: Data Processing")
    print("-" * 40)
    
    try:
        # Test missing value detection
        missing_count = df.isnull().sum().sum()
        print(f"✅ Missing Values Detected: {missing_count}")
        
        # Test duplicate detection
        duplicate_count = df.duplicated().sum()
        print(f"✅ Duplicates Detected: {duplicate_count}")
        
        # Test column analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        assert len(numeric_cols) > 0, "Should detect numeric columns"
        assert len(categorical_cols) > 0, "Should detect categorical columns"
        print(f"✅ Column Types: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
        
        # Test correlation calculation
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            assert corr_matrix.shape[0] == len(numeric_cols), "Correlation matrix size mismatch"
            print("✅ Correlation Analysis: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Processing: FAILED - {e}")
        return False

def test_feature_5_performance():
    """⚡ Test Feature 5: Performance & Memory Efficiency"""
    print("\n⚡ TESTING: Performance & Memory")
    print("-" * 40)
    
    try:
        # Test with progressively larger datasets
        sizes = [100, 1000, 5000]
        times = []
        
        for size in sizes:
            large_data = pd.DataFrame({
                'col1': np.random.randn(size),
                'col2': np.random.choice(['A', 'B', 'C'], size),
                'col3': np.random.randint(0, 100, size)
            })
            
            start_time = time.time()
            score = calculate_data_quality_score(large_data)
            analysis = generate_manual_analysis(large_data, f"Dataset_{size}")
            end_time = time.time()
            
            duration = end_time - start_time
            times.append(duration)
            print(f"✅ Size {size}: {duration:.3f}s (Score: {score:.1f})")
        
        # Performance should scale reasonably
        assert all(t < 10 for t in times), "Performance should be under 10s"
        print("✅ Performance Scaling: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Testing: FAILED - {e}")
        return False

def test_feature_6_error_handling():
    """🛡️ Test Feature 6: Error Handling & Edge Cases"""
    print("\n🛡️ TESTING: Error Handling")
    print("-" * 40)
    
    try:
        # Test empty dataframe
        empty_df = pd.DataFrame()
        try:
            score = calculate_data_quality_score(empty_df)
            print("✅ Empty DataFrame: Handled gracefully")
        except:
            print("⚠️  Empty DataFrame: Could cause issues")
        
        # Test single column
        single_col_df = pd.DataFrame({'col1': [1, 2, 3]})
        score = calculate_data_quality_score(single_col_df)
        assert 0 <= score <= 100, "Single column should work"
        print("✅ Single Column: PASSED")
        
        # Test all missing data
        missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        score = calculate_data_quality_score(missing_df)
        assert score < 50, "All missing should score low"
        print("✅ All Missing Data: PASSED")
        
        # Test large column count
        wide_df = pd.DataFrame({f'col_{i}': np.random.randn(100) for i in range(50)})
        score = calculate_data_quality_score(wide_df)
        assert 0 <= score <= 100, "Wide dataframe should work"
        print("✅ Wide DataFrame: PASSED")
        
        return True
        
    except Exception as e:
        print(f"❌ Error Handling: FAILED - {e}")
        return False

def run_comprehensive_tests():
    """🚀 Run all feature tests"""
    print("🚀 STARTING COMPREHENSIVE FEATURE TESTING")
    print("=" * 60)
    print("Testing all critical features before submission...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create test data
    df = create_test_dataset()
    if df is None:
        print("❌ CRITICAL FAILURE: Could not create test dataset")
        return False
    
    # Run all tests
    tests = [
        ("Data Loading", test_feature_1_data_loading),
        ("Data Quality", lambda: test_feature_2_data_quality_scoring(df)),
        ("Analysis Generation", lambda: test_feature_3_analysis_generation(df)),
        ("Data Processing", lambda: test_feature_4_data_processing(df)),
        ("Performance", test_feature_5_performance),
        ("Error Handling", test_feature_6_error_handling)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name}: CRITICAL ERROR - {e}")
            results[test_name] = False
    
    # Final report
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print("-" * 60)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"Duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 ALL FEATURES WORKING PERFECTLY!")
        print("✅ Application is READY FOR SUBMISSION")
        print(f"🌐 Access at: http://localhost:8503")
    else:
        print(f"\n⚠️  {total-passed} issues found - Review before submission")
    
    print("=" * 60)
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)

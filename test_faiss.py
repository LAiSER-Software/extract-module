import torch
import pandas as pd
import time
import argparse
import json
import os
from datetime import datetime
from laiser.skill_extractor import Skill_Extractor

def test_faiss_performance():
    """
    Test the performance of the FAISS-based taxonomy-aware AI vs. traditional approach
    """
    # Create results directory if it doesn't exist
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging to file
    log_file = os.path.join(results_dir, f"faiss_test_{timestamp}.log")
    with open(log_file, 'w') as f:
        f.write(f"===== LAiSER v0.2.25 Taxonomy-Aware AI Performance Test =====\n")
        f.write(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    print(f"\n===== LAiSER v0.2.25 Taxonomy-Aware AI Performance Test =====\n")
    print(f"Logging results to: {log_file}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test FAISS integration performance")
    parser.add_argument("--use_gpu", type=str, default=str(torch.cuda.is_available()), 
                        help="Enable or disable GPU use")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of job samples to process")
    args = parser.parse_args()
    
    use_gpu = True if args.use_gpu == "True" else False
    
    log(log_file, f"GPU Usage: {'Enabled' if use_gpu else 'Disabled'}")
    log(log_file, f"Number of samples: {args.samples}")
    
    # Initialize the Skill Extractor
    print("Initializing Skill Extractor...")
    log(log_file, "Initializing Skill Extractor...")
    
    se = Skill_Extractor(use_gpu=use_gpu)
    log(log_file, "Skill Extractor initialized successfully!")
    print("Skill Extractor initialized successfully!")
    
    # Load test data
    print("\nLoading test data...")
    log(log_file, "\nLoading test data...")
    
    job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')
    job_sample = job_sample[['description', 'job_id']]
    job_sample = job_sample[1:min(args.samples+1, len(job_sample))]
    
    log(log_file, f"Test data loaded: {len(job_sample)} job descriptions")
    print(f"Test data loaded: {len(job_sample)} job descriptions")
    
    # Test standard approach
    print("\n----- Testing Standard Approach (without FAISS) -----")
    log(log_file, "\n----- Testing Standard Approach (without FAISS) -----")
    
    start_time = time.time()
    standard_output = se.extractor(job_sample, 'job_id', text_columns=['description'], 
                                  batch_size=1, use_faiss=False, warnings=False)
    standard_time = time.time() - start_time
    
    log(log_file, f"Standard approach completed in {standard_time:.2f} seconds")
    log(log_file, f"Found {len(standard_output)} aligned skills")
    print(f"Standard approach completed in {standard_time:.2f} seconds")
    print(f"Found {len(standard_output)} aligned skills")
    
    # Test FAISS approach
    print("\n----- Testing FAISS Approach -----")
    log(log_file, "\n----- Testing FAISS Approach -----")
    
    start_time = time.time()
    faiss_output = se.extractor(job_sample, 'job_id', text_columns=['description'], 
                               batch_size=1, use_faiss=True, warnings=False)
    faiss_time = time.time() - start_time
    
    log(log_file, f"FAISS approach completed in {faiss_time:.2f} seconds")
    log(log_file, f"Found {len(faiss_output)} aligned skills")
    print(f"FAISS approach completed in {faiss_time:.2f} seconds")
    print(f"Found {len(faiss_output)} aligned skills")
    
    # Performance comparison
    print("\n===== Performance Comparison =====")
    log(log_file, "\n===== Performance Comparison =====")
    
    performance_data = {
        "standard_time": f"{standard_time:.2f}s",
        "faiss_time": f"{faiss_time:.2f}s",
        "standard_skills_count": len(standard_output),
        "faiss_skills_count": len(faiss_output),
    }
    
    if standard_time > 0:
        speedup = standard_time / faiss_time
        performance_data["speedup"] = f"{speedup:.2f}x"
        log(log_file, f"Standard approach: {standard_time:.2f} seconds")
        log(log_file, f"FAISS approach:    {faiss_time:.2f} seconds")
        log(log_file, f"Speedup:           {speedup:.2f}x")
        print(f"Standard approach: {standard_time:.2f} seconds")
        print(f"FAISS approach:    {faiss_time:.2f} seconds")
        print(f"Speedup:           {speedup:.2f}x")
    
    # Save detailed performance metrics as JSON
    performance_file = os.path.join(results_dir, f"performance_metrics_{timestamp}.json")
    with open(performance_file, 'w') as f:
        json.dump(performance_data, f, indent=4)
    
    # Save outputs for comparison
    standard_csv = os.path.join(results_dir, f"standard_approach_{timestamp}.csv")
    faiss_csv = os.path.join(results_dir, f"faiss_approach_{timestamp}.csv")
    
    standard_output.to_csv(standard_csv, index=False)
    faiss_output.to_csv(faiss_csv, index=False)
    
    log(log_file, f"\nOutputs saved to:")
    log(log_file, f"- {standard_csv}")
    log(log_file, f"- {faiss_csv}")
    log(log_file, f"- {performance_file}")
    
    print(f"\nOutputs saved to:")
    print(f"- {standard_csv}")
    print(f"- {faiss_csv}")
    print(f"- {performance_file}")
    
    # Save sample outputs for format comparison
    log(log_file, "\n===== Standard Approach Sample Output =====")
    log(log_file, standard_output.head(3).to_string())
    
    log(log_file, "\n===== FAISS Approach Sample Output =====")
    log(log_file, faiss_output.head(3).to_string())
    
    # Print sample comparison to console
    print("\n===== Output Format Comparison (Sample of 3 Records) =====")
    print("\nStandard Approach:")
    print(standard_output.head(3).to_string())
    
    print("\nFAISS Approach:")
    print(faiss_output.head(3).to_string())
    
    print(f"\nTest completed successfully! Full results saved to {log_file}")
    log(log_file, f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(log_file, "Test completed successfully!")

def log(file, message):
    """Append a message to the log file"""
    with open(file, 'a') as f:
        f.write(f"{message}\n")

if __name__ == "__main__":
    test_faiss_performance() 
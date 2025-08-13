import torch
import pandas as pd

# Import the LAiSER extractor
try:
    from laiser import SkillExtractorRefactored
except ImportError:
    try:
        from laiser.skill_extractor_refactored import SkillExtractorRefactored
    except ImportError:
        print("LAiSER not found. Please install or check your environment.")
        raise

print("LAiSER Skill Extraction Example")

# Initialize the extractor (like main.py)
print("\nInitializing the Skill Extractor...")
extractor = SkillExtractorRefactored(
    model_id="marcsun13/gemma-2-9b-it-GPTQ",
    use_gpu=True
)
print("The Skill Extractor has been initialized successfully!\n")

# Load sample data (like main.py)
print("Loading a sample dataset...")
try:
    job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')
    print("The sample jobs dataset has been loaded successfully!\n")
except:
    # Fallback to sample data if URL fails
    job_sample = pd.DataFrame({
        'description': [
            'We are looking for a Python developer with machine learning experience and SQL skills',
            'Seeking a data scientist proficient in R, statistics, and data visualization'
        ],
        'job_id': ['job_001', 'job_002']
    })
    print("Fallback sample data has been loaded successfully!\n")

# Filter data (like main.py)
job_sample = job_sample[['description', 'job_id']]
job_sample = job_sample[1:3]
print("The sample dataset has been filtered successfully!\n")
print("Head of the sample:\n", job_sample.head())

# Extract skills using the same approach as main.py
print("\nExtracting skills from jobs data...")
output = extractor.extract_and_align(
    job_sample,
    id_column='job_id',
    text_columns=['description'],
    input_type='job_desc',
    batch_size=32
)
print("The skills have been extracted from jobs data successfully...\n")

# Display and save results (like main.py)
print(output)
file_name = f'extracted_skills_for_{len(job_sample)}Jobs.csv'
output.to_csv(file_name, index=False)
print("The extracted skills have been saved to the file named:", file_name)

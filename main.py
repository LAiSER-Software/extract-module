import torch
import pandas as pd
import argparse
from laiser.skill_extractor import Skill_Extractor

# TODO: verify if everything is working fine with the latest version of the library
# Check with and without GPU availablility

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Skill Extractor on jobs and syllabi data.')
parser.add_argument('--HF_TOKEN', type=str, default=None, help='Hugging Face token for authentication')
parser.add_argument('--AI_MODEL_ID', type=str, default=None, help='Model name for Skill Extractor')
parser.add_argument('--use_gpu', type=str, default=str(torch.cuda.is_available()), help='Enable or disable GPU use.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for skills extraction')
args = parser.parse_args()


use_gpu = True if args.use_gpu == "True" else False

# based on the above arguments, to run this script:
# python main.py --HF_TOKEN <your_hf_token> --AI_MODEL_ID <your_model_id> --use_gpu True --batch_size 32

print('\n\nInitializing the Skill Extractor...')
se = Skill_Extractor(AI_MODEL_ID=args.AI_MODEL_ID, HF_TOKEN=args.HF_TOKEN, use_gpu=use_gpu)
print('The Skill Extractor has been initialized successfully!\n')

# Skill extraction from jobs data
print('\n\nLoading a sample dataset of 50 jobs...')
job_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/jobs-data/linkedin_jobs_sample_36rows.csv')
print('The sample jobs dataset has been loaded successfully!\n')

job_sample = job_sample[['description', 'job_id']]
job_sample = job_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', job_sample.head())

output = se.extractor(job_sample, 'job_id', text_columns=['description'], batch_size=args.batch_size)
print('The skills have been extracted from jobs data successfully...\n')

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(job_sample)}Jobs.csv'
output.to_csv(file_name, index=False)
print('The extracted skills have been saved to the file named:', file_name)

# Skill extraction from syllabi data
print('\n\nLoading a sample dataset of 50 syllabi...')
syllabi_sample = pd.read_csv('https://raw.githubusercontent.com/LAiSER-Software/datasets/refs/heads/master/syllabi-data/preprocessed_50_opensyllabus_syllabi_data.csv')
print('The sample syllabi dataset has been loaded successfully!\n')

syllabi_sample = syllabi_sample[['id', 'description', 'learning_outcomes']]
syllabi_sample = syllabi_sample[1:3]
print('The sample dataset has been filtered successfully!\n')
print('Head of the sample:\n', syllabi_sample.head())

output = se.extractor(syllabi_sample, 'id', text_columns=['description', 'learning_outcomes'], input_type='syllabus', batch_size=args.batch_size)
print('The skills have been extracted from syllabi data successfully...\n')

# Save the extracted skills to a CSV file
print(output)
file_name = f'extracted_skills_for_{len(syllabi_sample)}Syllabi.csv'
output.to_csv(file_name, index=False)
print('The extracted skills have been saved to the file named:', file_name)
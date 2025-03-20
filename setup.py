from setuptools import setup, find_packages

setup(
    name='dev-laiser',
    version='0.2.11', 
    author='Satya Phanindra Kumar Kalaga, Bharat Khandelwal, Prudhvi Chekuri', 
    author_email='phanindra.connect@gmail.com',  
    description='LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a tool designed to help learners, educators, and employers extract and share trusted information about skills. It uses a fine-tuned language model to extract raw skill keywords from text, then aligns them with a predefined taxonomy. You can find more technical details in the project’s paper.md and an overview in the README.md.', 
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/LAiSER-Software/extract-module',  
    packages=find_packages(),  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.2.3',
        'psutil==7.0.0',
        'skillNer==1.0.3',
        'spacy==3.8.4',
        'tokenizers==0.21.1',
        'accelerate==1.5.2',
        'bitsandbytes==0.45.3',
        'datasets==3.4.1',
        'huggingface_hub',
        'peft==0.15.0',
        'torch==2.6.0',
        'trl==0.15.2',
        'ipython==9.0.2',
        'python-dotenv==1.0.1',
        'vllm==0.8.1',
        'tqdm==4.67.1'
    ],
)
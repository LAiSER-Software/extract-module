from setuptools import setup, find_packages

setup(
    name='laiser',
    version='0.2.0', 
    author='Satya Phanindra Kumar Kalaga, Prudhvi Chekuri, Bharat Khandelwal', 
    author_email='phanindra.connect@gmail.com',  
    description='LAiSER (Leveraging Artificial Intelligence for Skill Extraction & Research) is a tool designed to help learners, educators, and employers extract and share trusted information about skills. It uses a fine-tuned language model to extract raw skill keywords from text, then aligns them with a predefined taxonomy. You can find more technical details in the project's paper.md and an overview in the README.md.', 
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    url='https://github.com/LAiSER-Software/extract-module',  
    packages=find_packages(),  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License', 
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.21.0,<2.0.0',
        'pandas>=1.3.0,<2.0.0',
        'psutil>=5.8.0,<6.0.0',
        'skillNer>=0.0.1,<1.0.0',
        'scikit-learn>=1.0.0,<2.0.0',
        'spacy>=3.0.0,<4.0.0',
        'transformers>=4.0.0,<5.0.0',
        'tokenizers>=0.10.0,<1.0.0',
        'torch>=1.10.0,<2.0.0',
        'ipython>=7.0.0,<8.0.0',
        'python-dotenv>=0.19.0,<1.0.0',
        'vllm>=0.1.0,<1.0.0',
        'tqdm>=4.62.0,<5.0.0'
    ],
    extras_require={
        'gpu': [
            'torch>=1.10.0,<2.0.0',
            'vllm>=0.1.0,<1.0.0'
        ],
        'cpu': [
            'skillNer>=0.0.1,<1.0.0',
            'spacy>=3.0.0,<4.0.0'
        ]
    }
)
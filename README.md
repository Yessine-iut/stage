It is a website allowing the use of an AI which detects the different propaganda of an entered text.

# Use the website on your pc
 ### Requirements
 
  - Visual Studio Code
  - NodeJs 12.17.0
  - Python 3.6.13
  - Conda 4.7.12
 
 ### Instructions
 
  - Clone git repository: https://github.com/Yessine-iut/stage.git
  - Create your own condo environment using “conda create -y --name <your_env_name> python==3.6.13”
  - Activate the environment using “conda activate <your_env_name>”
  - Type on terminal to install python dependencies in the created conda using “pip install -r Stage/span_boundary_detection/requirements.txt --user”
  - Install customised transformer model
     - Type on terminal “cd Stage/span_boundary_detection/custom_transformers”
     - Type on terminal “pip install -e .”
  - Type on terminal “cd Stage/span_boundary_detection“
  - Type on terminal “python Stage/span_boundary_detection/pipeline.py”
  - Flask package installation
      - Type on terminal “npm cache clean --force”
      - Go to directory using “cd Stage/propaganda”, if a directory “node_modules” exists, remove the folder manually
      - Type on terminal “npm install”

# Propaganda Snippets Detection (Service 1)
   This service allows you to highlight propaganda in different colors according to their type and their probability (Clearer for low probabilities) .
   
# Propaganda Word Cloud (Service 2)
   This service creates a word cloud, the greater the probability of propaganda the greater the word.
    


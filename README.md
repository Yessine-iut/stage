It is a website allowing the use of an AI which detects the different propaganda of an entered text.

# Use the website on your pc
 ### Requirements
 
  - Visual Studio Code
  - NodeJs 12.17.0
  - Python 3.6.13
  - Conda 4.7.12
 
 ### Instructions
 
  1. Clone git repository: https://github.com/Yessine-iut/stage.git
  2. Create your own condo environment using “conda create -y --name <your_env_name> python==3.6.13”
  3. Activate the environment using “conda activate <your_env_name>”
  4. Type on terminal to install python dependencies in the created conda using “pip install -r Stage/span_boundary_detection/requirements.txt --user”
  5. Install customised transformer model
     5.1 Type on terminal “cd Stage/span_boundary_detection/custom_transformers”
     5.2 Type on terminal “pip install -e .”
  6. Type on terminal “cd Stage/span_boundary_detection“
  7. Type on terminal “python Stage/span_boundary_detection/pipeline.py”
  8. Flask package installation
      8.1 Type on terminal “npm cache clean --force”
      8.2 Go to directory using “cd Stage/propaganda”, if a directory “node_modules” exists, remove the folder manually
      8.3 Type on terminal “npm install”

# Propaganda Snippets Detection (Service 1)
   This service allows you to highlight propaganda in different colors according to their type and their probability (Clearer for low probabilities) .
   
# Propaganda Word Cloud (Service 2)
   This service creates a word cloud, the greater the probability of propaganda the greater the word.
    


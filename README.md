### Team Name - Bottle Cap
### Problem Statement - The challenge is to design an innovative solution that integrates AI technologies to improve learning outcomes and student engagement. Participants should aim to create a system that helps students learn more effectively, boosts participation in educational activities, or assists teachers in delivering personalized content.
### Team Leader Email - giridharan1ads@gmail.com

## Education Chatbot 

This repository contains the code for a chatbot built using the Gemini API and Python. 

### Dependencies

This project requires the following Python libraries:

* `streamlit`
* `google-generativeai`
* `python-dotenv`
* `langchain`
* `PyPDF2`
* `chromadb`
* `faiss-cpu`
* `langchain_google_genai`
* `PIL`

### Usage

1. Set up your GEMINI_API_KEY:

Create a file named .env in the root directory of this project.
Add the following line to the .env file, replacing YOUR_API_KEY with your actual Gemini API key:

```bash
GEMINI_API_KEY=YOUR_API_KEY
```

2.  Run the Script:

```bash
python chat_AI.py
```
This will start the chatbot. You can then interact with it by typing your questions or prompts.


```bash
python chat_img.py
```
This will start the chatbot. First you have to insert the image you want to interact,then by typing your questions or prompts you will get the response.


```bash
python chat_pdf.py
```
This will start the chatbot. First you have to insert the PDF you want to interact,then by typing your questions or prompts you will get the response from the PDF and also from global search too.

### Code Structure

 - The code consists of a three Python script (chat_AI.py,chat_img.py,chat_pdf.py) that performs the following:
 - Imports necessary libraries and loads the API key from the .env file.
 - Configures the GenerativeModel with safety settings, generation configurations, and system instructions.
 - Starts a loop that continuously prompts the user for input and sends it to the model.
 - Receives the model's response and prints it to the console.

## What I Learned:

I have learned how to implement the LLM using langchain, time management and also how to present or execute our idea into real.
teamwork plays a huge role here,were my teammates are supported me to accompishment the work and learned many libraries and api's
and also usage of some modules,I had a wonderful experience with my seniors and they helped me a lot to do so far.


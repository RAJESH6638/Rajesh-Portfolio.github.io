import pandas as pd

# Specify the path to your WhatsApp chat file
file_path = 'WhatsApp Chat with RMG BADA Job 2.txt'

# Reading the file
with open(file_path, 'r', encoding='utf-8') as file:
    chat_data = file.readlines()

# Displaying the first few lines to understand the format
chat_data[:10]  # Change the number to see more lines if needed
import torch
from transformers import pipeline
import pandas as pd
import re

# Load Zero-Shot Classification pipeline from Hugging Face
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for extraction
candidate_labels = ["Role", "Experience", "Location", "Email"]

# Sample job description data (replace with your actual data)
job_descriptions = [
    "Business Analyst with 3-5 years experience required in Bangalore. Please send your resume to hr@example.com.",
    "Looking for a Senior Data Scientist. Experience: 2-4 years. Location: Mumbai. Email: example@company.com.",
    "Role: Frontend Developer. Experience: 1 year. Location: Pune. Apply at jobs@company.in."
]

# Function to extract details
def extract_details(job_descriptions):
    extracted_data = []

    for job in job_descriptions:
        # Classify each line into predefined labels
        classification = classifier(job, candidate_labels)
        
        # Create a dictionary to store extracted information
        job_info = {"Role": None, "Experience": None, "Location": None, "Email": None}
        
        # Assign classification results
        for label, score in zip(classification['labels'], classification['scores']):
            if label == "Role" and score > 0.5:  # Thresholding for confidence score
                job_info['Role'] = job
            elif label == "Experience" and score > 0.5:
                job_info['Experience'] = job
            elif label == "Location" and score > 0.5:
                job_info['Location'] = job
            elif label == "Email" and score > 0.5:
                # Extract email using regex from job description text
                email = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', job)
                if email:
                    job_info['Email'] = email[0]

        # Append the result to the extracted data list
        extracted_data.append(job_info)
    
    return extracted_data

# Extract data from job descriptions
extracted_data = extract_details(job_descriptions)

# Convert the data into a DataFrame
df = pd.DataFrame(extracted_data)

# Display the extracted data
print(df)

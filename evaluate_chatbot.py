import pandas as pd
import numpy as np
from virtual_therpist import get_therapeutic_response, rate_response
from openai import OpenAI
from config import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)

# Load the dataset
df = pd.read_json("hf://datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

# Select prompts and expected responses from the dataset
prompts = df['Context'].tolist()
ideal_responses = df['Response'].tolist()

# Function to evaluate responses against ideal responses
def evaluate_responses(client, prompts, ideal_responses):
    evaluations = []

    for i, prompt in enumerate(prompts):
        # Only evaluate every 10th prompt
        if i % 10 == 0:
            print(f"Evaluating prompt {i} of {len(prompts)}")
            
            # Generate a response from the chatbot
            model_response = get_therapeutic_response(client, prompt)
            print(f"Prompt: {prompt}")
            print(f"Chatbot Response: {model_response}")
            print(f"Ideal Response: {ideal_responses[i]}")

            # Rate the response using the OpenAI API
            ratings = rate_response(client, prompt, model_response, ideal_responses[i])
            print(f"Ratings: {ratings}\n")
            
            # Extract ratings
            accuracy = ratings[0] 
            safety = ratings[1]
            clarity = ratings[2]

            # Create an evaluation summary
            evaluation_summary = f"Accuracy: {accuracy}, Safety: {safety}, Clarity: {clarity}"
            
            evaluations.append({
                "Prompt": prompt,
                "Ideal Response": ideal_responses[i],
                "Chatbot Response": model_response,
                "Evaluation": evaluation_summary
            })
        
    return pd.DataFrame(evaluations)

# Main evaluation process
def main():
    print("Starting evaluation of chatbot responses...")
    evaluation_results = evaluate_responses(client, prompts, ideal_responses)

    # Save results to CSV file
    evaluation_results.to_csv("evaluation_results.csv", index=False)
    print("Evaluation completed. Results saved to 'evaluation_results.csv'.")

if __name__ == "__main__":
    main()
from crewai import Crew, Agent, Task, Process
from dotenv import load_dotenv
from crewai_tools import FileReadTool, BaseTool, SerperDevTool
from rouge_score import rouge_scorer
import os
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from bert_score import BERTScorer
from typing import Tuple
import torch

load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "default_value")
os.environ["AGENTOPS_API_KEY"] = os.getenv("AGENTOPS_API_KEY", "default_value")
os.environ["OPENAI_MODEL_NAME"] = "gpt-4-0125-preview"

# Initialize agentops
import agentops
agentops.init(tags=["crewai-agent"])

class ROUGEScoreTool(BaseTool):
    name: str = "ROUGE Score Tool"
    description: str = "Calculates ROUGE scores for summary evaluation. Takes a single string argument with 'reference_file_path||candidate_file_path'."

    def _run(self, reference_summary: str, candidate_summary: str) -> str:
        try:
            # Calculate ROUGE scores
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference_summary, candidate_summary)
            
            # Formatting scores
            result = {
                'ROUGE-1': {
                    'precision': scores['rouge1'].precision,
                    'recall': scores['rouge1'].recall,
                    'f1': scores['rouge1'].fmeasure
                },
                'ROUGE-2': {
                    'precision': scores['rouge2'].precision,
                    'recall': scores['rouge2'].recall,
                    'f1': scores['rouge2'].fmeasure
                },
                'ROUGE-L': {
                    'precision': scores['rougeL'].precision,
                    'recall': scores['rougeL'].recall,
                    'f1': scores['rougeL'].fmeasure
                }
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
        


class BERTScoreTool(BaseTool):
    name: str = "BERT Score Tool"
    description: str = "Calculates BERT scores for summary evaluation. Takes a single string argument with 'reference_file_path||candidate_file_path'."

    def _run(self, reference_summary: str, candidate_summary: str) -> str:
        try:
            # Initialize BERTScorer
            scorer = BERTScorer(model_type='bert-base-uncased', lang='en')

            # Calculate BERT scores
            P, R, F1 = scorer.score([candidate_summary], [reference_summary])

            # Formatting scores
            result = {
                'BERTScore': {
                    'precision': P.mean().item(),
                    'recall': R.mean().item(),
                    'f1': F1.mean().item()
                }
            }
            
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"

bert_tool = BERTScoreTool()

rouge_tool = ROUGEScoreTool()
file_tool = FileReadTool()
search_tool = SerperDevTool()

# Define the agents
Senior_Document_Analyst = Agent(
    role='Document Review and Initial Analysis',
    goal='To examine and analyze legal document at path: {Original_Document}, identifying key information and preparing them for summarization.',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in understanding the structure and content of legal documents. "
        "Specializes in extracting critical details and preparing outlines for further processing. "
        "Highly experienced in handling various types of legal documents including contracts and court rulings."
    ),
    tools=[file_tool],
    allow_delegation=False
)

Judicial_Assistant = Agent(
    role='Document Summarization',
    goal='To create clear and comprehensive summaries of legal documents based on the analysis of document and key points provided by the Senior Document Analyst.',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in writing coherent, complete, relevant, and fluent summaries of legal documents"
        "Focused on maintaining the accuracy and relevance of the content"
        "Works closely with the Senior Document Analyst to ensure comprehensive summarization."
    ),
    tools=[file_tool],
    allow_delegation=True
)

Evaluation_Specialist = Agent(
    role='Evaluation of Document Summaries',
    goal='To generate a proper rouge score and bert score evaluation table for candidate summary by comparing it with the reference summary.',
    verbose=True,
    memory=True,
    backstory=(
        "Expert in evaluation for legal document summaries. "
    ),
    tools=[file_tool, rouge_tool,bert_tool, search_tool],
    allow_delegation=True
)

Senior_Court_Reporter = Agent(
    role= "Evaluation of Document Summaries",
    goal= "To generate a proper evaluation table for candidate summaries by comparing them with the reference summary and provide reasoning for the best one.",
    verbose=True,
    memory=True,
    backstory= "Expert in evaluating legal document summaries and providing detailed reasoning.",
    tools=[file_tool],
    allow_delegation=True
)

# Define the input, reference, and output directories
input_directory = "D:/NUST/Preply/Courses/XC_GenerativeAI/CrewAI-AgentOps-main/Documents"
reference_directory = "D:/NUST/Preply/Courses/XC_GenerativeAI/CrewAI-AgentOps-main/Summaries"
output_directory = "D:/NUST/Preply/Courses/XC_GenerativeAI/CrewAI-AgentOps-main/Generated_Summaries"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Function to create output paths based on the input file name
def create_output_paths(file_name):
    base_name = os.path.splitext(file_name)[0]
    summary_name = f"G-{base_name}.txt"
    return {
        "main_points": os.path.join(output_directory, f"G-{base_name}", "main_points.md"),
        "summary": os.path.join(output_directory, summary_name),
        "gen_summary": os.path.join(output_directory, f"G-{base_name}", "gen_summary.md"),
        "R_review": os.path.join(output_directory, f"G-{base_name}", "R_review.md"),
        "B_review": os.path.join(output_directory, f"G-{base_name}", "B_review.md"),
        "reasoning": os.path.join(output_directory, f"G-{base_name}", "reasoning.md"),
    }

# Get all text files in the input directory
input_files = [f for f in os.listdir(input_directory) if f.endswith('.txt')]


for input_file in input_files:
    try:
        # Extract the numeric part from the input file name to find the corresponding summary file
        file_number = os.path.splitext(input_file)[0].split('-')[1]
        reference_file = f"hd-{file_number}.txt"

        # Check if the corresponding reference file exists
        if not os.path.exists(os.path.join(reference_directory, reference_file)):
            print(f"Reference summary for {input_file} not found, skipping.")
            continue

        # Create a folder for each input file to store the outputs
        file_output_directory = os.path.join(output_directory, f"G-{file_number}")
        os.makedirs(file_output_directory, exist_ok=True)

        # Create the output paths
        output_paths = create_output_paths(f"{file_number}.txt")

        # Define the tasks with dynamically generated output paths
        Preparation_Task = Task(
            description=(
                "Read the original text of the document from path {Original_Document}. "
                "Prepare the documents for summarization by organizing and structuring the extracted information. "
                "Create outlines or structured notes that facilitate accurate and efficient summarization by the Summarization Specialist."
            ),
            expected_output=(
                'Organized and structured outlines or notes ready for summarization in the following format:\n'
                "Case Details:\n"
                "   - Case Title:\n"
                "   - Case No.: [Number]\n"
                "   - Decision Date:\n"
                "   - Hearing Date:\n"
                "   - Against Judgment Dated: [Date] (if applicable)\n"
                "   - From: [Name of Lower Court/Tribunal or Trial Court] (if applicable)\n"
                "\n"
                "Parties Involved:\n"
                "   - Appellant/Accused:\n"
                "   - Respondent/Prosecution:\n"
                "   - Counsel for Appellant/Accused:\n"
                "   - Counsel for Respondent/Prosecution:\n"
                "\n"
                "Summary:\n"
                "   - Introduction:\n"
                "   - Background:\n"
                "   - Facts of the Case:\n"
                "   - Legal Proceedings and Arguments:\n"
                "   - Counsel Arguments:\n"
                "        - Appellant's/Accused's Counsel:\n"
                "        - Respondent's/Prosecution's Counsel:\n"
                "   - Court's Analysis:\n"
                "   - Final Decision/Order:\n"
                "   - Key Legal Citations:\n"
                "\n"
                "Conclusion:\n"
            ),
            tools=[file_tool],
            agent=Senior_Document_Analyst,
            output_file=output_paths["main_points"]
    )
        Content_Condensation_Task = Task(
            description=(
                "Read the original document from path: {Original_Document}. "
                "Read the main points of the original document obtained from path: {Main_points}. "
                "Summarize the original document while maintaining a high level of detail and precision, using legal terminology appropriately. "
                "Ensure the summary serves the needs of legal professionals effectively. "
                "Include 'Scope' and 'Validity' sections only if the case involves legal interpretation, procedural questions, or the applicability of specific laws or provisions."
            ),
            expected_output=(
                "A professional summary of the legal document formatted as follows:\n"
                "\n"
                "Case Title and Parties:\n"
                "\u2022 Petitioners/Appellants: [Name of the Petitioner(s)/Appellant(s)]\n"
                "\u2022 Respondents/Defendants: [Name of the Respondent(s)/Defendant(s)]\n"
                "\n"
                "Case Information:\n"
                "\u2022 Case No.: [Case Number]\n"
                "\u2022 Decided on: [Date of Decision]\n"
                "\u2022 Against Judgment Dated: [Date of Lower Court/Tribunal Decision] (if applicable)\n"
                "\u2022 From: [Name of Lower Court/Tribunal or Trial Court] (if applicable)\n"
                "\n"
                "Nature of the Case:\n"
                "\u2022 [Brief description of the type of case, e.g., constitutional petition, public procurement dispute, criminal offense, property dispute, etc.]\n"
                "\n"
                "Facts of the Case:\n"
                "\u2022 [Brief summary of the relevant facts leading up to the dispute or legal challenge]\n"
                "\n"
                "Legal Issues:\n"
                "\u2022 [Summary of the key legal issues or questions addressed in the case]\n"
                "\n"
                "Scope and Validity:\n"
                "\u2022 [If applicable, provide details on the scope of the court's ruling and the validity of the legal interpretations applied. This section is included if the case involves legal interpretation, procedural questions, or the applicability of specific laws or provisions.]\n"
                "\n"
                "Applicable Laws and Regulations:\n"
                "\u2022 [List of relevant laws, statutes, or regulations cited in the case]\n"
                "\n"
                "Court's Findings and Rationale:\n"
                "\u2022 [Summary of the court's findings, reasoning, and legal principles applied]\n"
                "\n"
                "Decision:\n"
                "\u2022 [Summary of the court's final decision, including any modifications to the lower court's decision or the outcome for the parties involved]\n"
                "\n"
                "Legal Representatives:\n"
                "\u2022 For Petitioners/Appellants: [Name and title of the legal representative]\n"
                "\u2022 For Respondents/Defendants: [Name and title of the legal representative]\n"
                "\n"
                "Additional Notes:\n"
                "\u2022 [Any additional relevant information or context, e.g., impact of the decision, procedural notes, references to precedents, etc.]\n"
            ),
            tools=[file_tool],
            agent=Judicial_Assistant,
            output_file=output_paths["summary"]
    )

        R_Evaluation_Task = Task(
            description=(
            "Evaluate the quality of the candidate summary by comparing it with the reference summary using rouge scores. "
            "Just Provide a table of rouge score evaluation in output"
            "Don't provide any other information like what does rouge score represent."
        ),
            expected_output=(
                "ROUGE-1 Scores\n"
                    "Precision:\n"
                    "Recall:\n"
                    "F1 Score:\n\n"
                "ROUGE-2 Scores\n"
                    "Precision:\n"
                    "Recall:\n"
                    "F1 Score:\n\n"
                "ROUGE-L Scores\n"
                    "Precision:\n"
                    "Recall:\n"
                    "F1 Score:"
        ),
            tools=[file_tool, rouge_tool],  
            agent=Evaluation_Specialist,
            output_file=output_paths["R_review"]
        )

        B_Evaluation_Task = Task(
            description=(
            "Evaluate the quality of the candidate summary by comparing it with the reference summary using bert scores. "
            "Just Provide a table of bert score evaluation in output"
            "Don't provide any other information like what does bert score represent."
        ),
            expected_output=(
                "BERT Scores\n"
                    "Precision:\n"
                    "Recall:\n"
                    "F1 Score:\n\n"
        ),
            tools=[file_tool, bert_tool],  
            agent=Evaluation_Specialist,
            output_file=output_paths["B_review"]
        )

        Reasoning_Task = Task(
            description=(
                "Read the original document from path: {Original_Document}. "
                "Read Summary1 from path {Summary}. "
                "Read Summary2 from path {Gen_Summary}. "
                "Evaluate the two provided summaries and select the best one based on clarity, accuracy, and completeness. Provide a detailed reasoning for the selection."
            ),
            expected_output="The best summary, a ranking of both summaries, and detailed reasoning highlighting the strengths and weaknesses of each summary.",
            tools=[file_tool],  
            agent=Senior_Court_Reporter,
            output_file=output_paths["reasoning"]
        )

        # Define path_inputs
        path_inputs = {
            "Original_Document": os.path.join(input_directory, input_file),
            "Summary": os.path.join(reference_directory, reference_file),
            "Main_points": output_paths["main_points"],
            "Gen_Summary": output_paths["summary"]
        }
        crew = Crew(
            agents=[
            Senior_Document_Analyst,
            Judicial_Assistant, 
            Evaluation_Specialist,
            Senior_Court_Reporter
            ],
            tasks=[
            Preparation_Task,
            Content_Condensation_Task, 
            R_Evaluation_Task,
            B_Evaluation_Task,
            Reasoning_Task
            ],
            process=Process.sequential,
            memory=True,
            cache=True,
            max_rpm=100
        )

        # Kickoff the crew with the current input file
        result = crew.kickoff(inputs=path_inputs)
        print(f"The outputs have been compiled for {input_file}")
        print("Result => ", result)

    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")

print("Summarization and evaluation process completed.")

agentops.end_session("Success")

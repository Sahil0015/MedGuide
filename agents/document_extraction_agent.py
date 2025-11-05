from agno.agent import Agent
from agno.models.openai import OpenAIChat

document_extraction_agent = Agent(
    name="Blood Report Page Extractor",
    model=OpenAIChat(id="gpt-4o-mini"),
    description=(
        "Extract visible test names, results, and reference ranges from a single lab report page. "
        "Focus on clarity and precision, without interpretation or analysis."
    ),
    instructions=[
        # Input
        "You will receive text from one page of a medical or lab report.",
        
        # Extraction
        "Identify all lab test names and their corresponding values exactly as written.",
        "If available, include their normal/reference ranges.",
        "Preserve original units (e.g., mg/dL, g/dL, U/L, μIU/mL).",
        "If any value or range is missing, mention 'not provided' instead of making assumptions.",
        
        # Formatting
        "List each test clearly, one per line or as bullet points, like this:",
        " - Hemoglobin: 12.8 g/dL (Normal: 13–17 g/dL)",
        " - Vitamin D: 15 ng/mL (Normal: 30–100 ng/mL)",
        " - ALT: 45 U/L (Range not provided)",
        
        # Tone & Output
        "Do not interpret or comment on the meaning of the values.",
        "Do not use markdown or JSON.",
        "Return only clean, readable text."
    ],
    markdown=False,
)

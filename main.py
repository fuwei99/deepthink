import asyncio
import json
import os
import logging
import re
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from openai import AsyncOpenAI, APIStatusError
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Dict, Any, AsyncGenerator

# --- Logger Configuration ---
# Create run_logs directory if it doesn't exist (at project root)
if not os.path.exists("run_logs"):
    os.makedirs("run_logs")

# Create output directory if it doesn't exist (at project root)
if not os.path.exists("output"):
    os.makedirs("output")

# We'll create a new log file for each run with timestamp
# The actual log file will be created in agent_solver function
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # Console output only, file handler will be added per run
    ]
)
logger = logging.getLogger(__name__)


# --- Configuration ---
CONFIG_FILE = "config.json"

app = FastAPI()

# Mount static files & templates using relative paths
# This ensures it works both locally and inside the Docker container
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Prompts (from agent_openaiSDK.py) ---
def create_system_prompt(language: str) -> str:
    """Creates a dynamic system prompt with language and rigor instructions."""
    return f"""### Core Instructions ###

*   **Reply Language:** You MUST reply in {language}.
*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured using the following tags. Ensure the tags are on their own lines and the content for each section is enclosed in curly braces `{{}}`.

[summary]:{{
*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.
}}

[solution]:{{
Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.
}}

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "[summary]" and "[solution]" sections to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.
"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt, including the required [summary]:{{...}} and [solution]:{{...}} format.
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured using the following tags. Ensure the tags are on their own lines and the content for each section is enclosed in curly braces `{{}}`.

[summary]:{{
*   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
*   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
    *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
    *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).
}}

[log]:{{
Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.
}}

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

[summary]:{{
*   **Final Verdict:** The solution is **invalid** because it contains a Critical Error.
*   **List of Findings:**
    *   **Location:** "By interchanging the limit and the integral, we get..."
        *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
    *   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
        *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.
}}
"""


verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""


# --- Helper Functions ---

def load_settings():
    """Loads settings from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"base_url": "", "api_key": ""}

def save_settings(settings: dict):
    """Saves settings to the config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f, indent=2)

# --- Agent Logic (Adapted from agent_openaiSDK.py) ---

def build_request_payload(system_prompt: str, question_prompt: str, other_prompts: List[str] = None) -> List[Dict[str, str]]:
    """Builds the messages payload for the OpenAI API request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question_prompt})
    if other_prompts:
        for prompt in other_prompts:
            messages.append({"role": "user", "content": prompt})
    return messages

async def send_api_request_async(client: AsyncOpenAI, model_name: str, messages: List[Dict[str, str]], max_retries: int = 3, retry_callback=None) -> ChatCompletion:
    """Sends the request to the OpenAI-compatible API asynchronously with retry logic."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Sending API request to model: {model_name} (attempt {attempt + 1}/{max_retries})")
            
            # Special handling for Gemini models if needed (from agent_openaiSDK.py)
            extra_args = {}
            if "gemini" in model_name.lower():
                extra_args = {
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_budget": 32768,
                                "include_thoughts": False
                            }
                        }
                    }
                }
            
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                top_p=1.0,
                **extra_args
            )
            
            # Check if the response has valid content
            if response.choices is None or len(response.choices) == 0:
                raise ValueError(f"No candidates returned from model {model_name}")
            
            if response.choices[0].message is None or response.choices[0].message.content is None:
                raise ValueError(f"Empty response content from model {model_name}")
            
            logger.info(f"API request successful for model: {model_name}")
            return response
            
        except APIStatusError as e:
            logger.warning(f"API status error for {model_name} (attempt {attempt + 1}/{max_retries}): {e.status_code} - {e.message}")
            
            # Retry on specific status codes
            if e.status_code in [429, 500, 502, 503, 504] and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.info(f"Retrying in {wait_time} seconds...")
                if retry_callback:
                    await retry_callback(f"API error ({e.status_code}). Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                # Don't retry on client errors (4xx except 429) or after max retries
                logger.error(f"API request failed for {model_name} after {attempt + 1} attempts: {e.status_code} - {e.message}")
                raise HTTPException(status_code=500, detail=f"API Request Failed: {e.message}")
                
        except ValueError as e:
            # Handle cases where API returns 200 but with empty/invalid content
            logger.warning(f"Invalid response from {model_name} (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                if retry_callback:
                    await retry_callback(f"Empty response received. Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"API request failed for {model_name} after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail=f"API Request Failed: {e}")
                
        except Exception as e:
            logger.warning(f"Unexpected error for {model_name} (attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                if retry_callback:
                    await retry_callback(f"Unexpected error occurred. Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"API request failed for {model_name} after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail=f"API Request Failed: {e}")
    
    # This should never be reached, but just in case
    raise HTTPException(status_code=500, detail=f"API Request Failed: Maximum retries exceeded")

def extract_text_from_response(response: ChatCompletion) -> str:
    """Extracts the generated text from the API response."""
    # At this point, send_api_request_async has already validated the response
    return response.choices[0].message.content

def extract_tagged_section(text: str, tag: str) -> str:
    """Extracts content from a tagged section like [tag]:{...}."""
    # The pattern looks for [tag]:{content}.
    # It handles multiline content with re.DOTALL.
    # It is non-greedy (.*?) to handle multiple sections in one response.
    pattern = re.compile(r"\[" + re.escape(tag) + r"\]:\{(.*?)\}", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    logger.warning(f"Could not find tag '[{tag}]' in the text.")
    return ""


async def verify_solution_async(client: AsyncOpenAI, model_name: str, problem_statement: str, solution: str, retry_callback=None) -> (str, str):
    """Asynchronously verifies the solution."""
    logger.info("Verifying solution...")
    dsol = extract_tagged_section(solution, "solution")
    
    # Fallback: If tagged section is not found, use the whole solution.
    # This can happen if the model fails to follow formatting instructions.
    if not dsol:
        logger.warning("Could not extract [solution] section. Using full response for verification.")
        dsol = solution

    verification_problem = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    messages = build_request_payload(
        system_prompt=verification_system_prompt,
        question_prompt=verification_problem
    )
    
    res = await send_api_request_async(client, model_name, messages, retry_callback=retry_callback)
    verification_output = extract_text_from_response(res)
    logger.info("Verification output received.")

    check_correctness_prompt = f"""Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?\n\n{verification_output}"""
    check_messages = build_request_payload(system_prompt="", question_prompt=check_correctness_prompt)
    
    r = await send_api_request_async(client, model_name, check_messages, retry_callback=retry_callback)
    is_good_verification = extract_text_from_response(r)
    logger.info(f"Verification check result: {is_good_verification.strip()}")
    
    bug_report = ""
    if "yes" not in is_good_verification.lower():
        bug_report = extract_tagged_section(verification_output, "summary")
        # If summary tag is not found in verifier output, use the whole output as bug report
        if not bug_report:
            logger.warning("Could not extract [summary] from verifier output. Using full output as bug report.")
            bug_report = verification_output

        logger.warning("Verification failed. Bug report generated.")
    else:
        logger.info("Verification passed.")
        
    return bug_report, is_good_verification

async def check_if_solution_claimed_complete_async(client: AsyncOpenAI, model_name: str, solution: str) -> bool:
    """Checks if the solution claims to be complete."""
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """
    messages = build_request_payload(system_prompt="", question_prompt=check_complete_prompt)
    r = await send_api_request_async(client, model_name, messages)
    o = extract_text_from_response(r)
    return "yes" in o.lower()

async def agent_solver(
    client: AsyncOpenAI, model_name: str, problem_statement: str, language: str
) -> AsyncGenerator[str, None]:
    """The main agent logic, implemented as an async generator for streaming."""
    from datetime import datetime
    
    # Create a unique log file for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a simple hash from problem statement for identification
    problem_hash = abs(hash(problem_statement[:50])) % 10000
    log_filename = f"solution_{problem_hash:04d}_agent_{timestamp}.log"
    log_filepath = os.path.join("run_logs", log_filename)
    
    # Add file handler for this run
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting agent solver for model {model_name}.")
    logger.info(f"Log file: {log_filepath}")
    
    # Use dynamic system prompt that includes the specified language
    system_prompt = create_system_prompt(language)
    
    solution = "" # Initialize solution to ensure it has a value

    # Define retry callback for streaming retry status to frontend
    async def retry_callback(status_message: str):
        yield json.dumps({"type": "retry_status", "content": status_message})

    # Step 1: Initial Exploration
    yield json.dumps({"type": "status", "content": "Phase 1: Initial solution generation..."})
    logger.info("Phase 1: Initial solution generation...")
    messages = build_request_payload(
        system_prompt=system_prompt,
        question_prompt=problem_statement
    )
    
    # Create an async generator to handle retry messages
    retry_messages = []
    async def collect_retry_callback(message):
        retry_messages.append(message)
    
    response1 = await send_api_request_async(client, model_name, messages, retry_callback=collect_retry_callback)
    
    # Yield any retry messages that were collected
    for retry_msg in retry_messages:
        yield json.dumps({"type": "retry_status", "content": retry_msg})
    retry_messages.clear()
    
    output1 = extract_text_from_response(response1)
    logger.info("Initial solution generated.")
    yield json.dumps({"type": "intermediate_solution", "content": output1, "title": "Initial Solution"})

    # Step 2: Self-Improvement
    yield json.dumps({"type": "status", "content": "Phase 2: Self-correction and refinement..."})
    logger.info("Phase 2: Self-correction and refinement...")
    improvement_messages = messages.copy()
    improvement_messages.append({"role": "assistant", "content": output1})
    improvement_messages.append({"role": "user", "content": self_improvement_prompt})
    
    response2 = await send_api_request_async(client, model_name, improvement_messages, retry_callback=collect_retry_callback)
    
    # Yield any retry messages that were collected
    for retry_msg in retry_messages:
        yield json.dumps({"type": "retry_status", "content": retry_msg})
    retry_messages.clear()
    
    solution = extract_text_from_response(response2)
    logger.info("Self-corrected solution generated.")
    yield json.dumps({"type": "intermediate_solution", "content": solution, "title": "Self-Corrected Solution"})

    # Step 3: Verification and Correction Loop
    error_count = 0
    correct_count = 0
    max_iterations = 5 # Limit iterations to prevent infinite loops

    for i in range(max_iterations):
        yield json.dumps({"type": "status", "content": f"Phase 3: Verification cycle {i + 1}/{max_iterations}..."})
        logger.info(f"Phase 3: Verification cycle {i + 1}/{max_iterations}...")
        
        # is_complete = await check_if_solution_claimed_complete_async(client, model_name, solution)
        # if not is_complete:
        #     yield json.dumps({"type": "status", "content": "Solution does not claim to be complete. Stopping."})
        #     break

        verify, good_verify = await verify_solution_async(client, model_name, problem_statement, solution, retry_callback=collect_retry_callback)
        
        # Yield any retry messages that were collected during verification
        for retry_msg in retry_messages:
            yield json.dumps({"type": "retry_status", "content": retry_msg})
        retry_messages.clear()

        if "yes" in good_verify.lower():
            yield json.dumps({"type": "status", "content": f"Verification {i + 1} passed. Confirming again..."})
            logger.info(f"Verification {i + 1} passed. Confirming again...")
            correct_count += 1
            error_count = 0
            if correct_count >= 2: # Require 2 consecutive successful verifications
                yield json.dumps({"type": "status", "content": "Solution verified successfully multiple times. Finalizing."})
                break
        else:
            yield json.dumps({"type": "status", "content": f"Verification {i + 1} failed. Attempting correction..."})
            logger.warning(f"Verification {i + 1} failed. Attempting correction...")
            yield json.dumps({"type": "bug_report", "content": verify, "title": f"Bug Report {i + 1}"})
            correct_count = 0
            error_count += 1
            if error_count >= 3:
                yield json.dumps({"type": "status", "content": "Failed to correct solution after multiple attempts. Stopping."})
                break

            correction_messages = build_request_payload(
                system_prompt=system_prompt,
                question_prompt=problem_statement,
            )
            correction_messages.append({"role": "assistant", "content": solution})
            correction_messages.append({"role": "user", "content": correction_prompt + "\n\n" + verify})

            response_corrected = await send_api_request_async(client, model_name, correction_messages, retry_callback=collect_retry_callback)
            
            # Yield any retry messages that were collected during correction
            for retry_msg in retry_messages:
                yield json.dumps({"type": "retry_status", "content": retry_msg})
            retry_messages.clear()
            
            solution = extract_text_from_response(response_corrected)
            logger.info(f"Generated corrected solution in cycle {i+1}.")
            yield json.dumps({"type": "intermediate_solution", "content": solution, "title": f"Corrected Solution {i + 1}"})

    yield json.dumps({"type": "final_solution", "content": solution})
    logger.info("Agent solver finished.")
    
    # Save final solution to output directory as markdown
    if solution:
        try:
            output_filename = f"solution_{problem_hash:04d}_{timestamp}.md"
            output_filepath = os.path.join("output", output_filename)
            
            # Create markdown content
            md_content = f"""# Solution Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Model:** {model_name}  
**Language:** {language}  
**Log File:** {log_filename}

---

## Problem Statement

{problem_statement}

---

## Solution

{solution}

---
*This solution was generated by the IMO Agent solver.*
"""
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"Solution saved to: {output_filepath}")
            yield json.dumps({"type": "status", "content": f"Solution saved to: {output_filepath}"})
        except Exception as e:
            logger.error(f"Failed to save solution to output directory: {e}")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    file_handler.close()


# --- Keep-Alive Task ---
async def keep_alive():
    """A background task to prevent the service from sleeping."""
    while True:
        await asyncio.sleep(50 * 60) # Sleep for 50 minutes
        logger.info("Keep-alive: Waking up to perform a routine check.")
        try:
            # This serves as a lightweight API call to keep the service active
            settings = load_settings()
            base_url = settings.get("base_url")
            api_key = settings.get("api_key")
            if base_url and api_key:
                client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
                await client.models.list()
                logger.info("Keep-alive: Successfully pinged the model API.")
            else:
                logger.info("Keep-alive: API settings not configured, skipping API ping.")
        except Exception as e:
            logger.warning(f"Keep-alive: Routine check failed with error: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Starting keep-alive background task.")
    asyncio.create_task(keep_alive())


# --- API Endpoints ---

@app.get("/api/problems", response_class=JSONResponse)
async def get_problems():
    """Fetches the list of available problem files from the problems directory."""
    problem_dir = "problems"
    if not os.path.exists(problem_dir):
        return []
    try:
        problems = [f for f in os.listdir(problem_dir) if f.endswith('.txt')]
        return sorted(problems)
    except Exception as e:
        logger.error(f"Error reading problems directory: {e}")
        return []

@app.get("/api/problems/{filename}", response_class=JSONResponse)
async def get_problem_content(filename: str):
    """Fetches the content of a specific problem file."""
    # Basic security check to prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
        
    problem_filepath = os.path.join("problems", filename)
    if not os.path.exists(problem_filepath):
        raise HTTPException(status_code=404, detail="Problem file not found.")
    
    try:
        with open(problem_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        logger.error(f"Error reading problem file '{filename}': {e}")
        raise HTTPException(status_code=500, detail="Could not read problem file.")

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request):
    """Serves the settings page."""
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/api/settings", response_class=JSONResponse)
async def get_settings():
    """Endpoint to get currently stored settings."""
    return load_settings()

@app.post("/api/settings")
async def update_settings(request: Request):
    """Endpoint to update and save settings."""
    settings = await request.json()
    save_settings(settings)
    return {"status": "success", "message": "Settings saved."}

@app.get("/api/models", response_class=JSONResponse)
async def get_models():
    """Fetches the list of available models from the proxy."""
    settings = load_settings()
    base_url = settings.get("base_url")
    api_key = settings.get("api_key")

    if not base_url or not api_key:
        return [] # Return empty list if not configured

    try:
        # Give this a shorter timeout as it should be a quick request
        client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=60.0)
        models = await client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        # Don't crash the UI if the proxy is down, just return an empty list
        print(f"Error fetching models: {e}")
        return []

@app.post("/api/solve")
async def solve_problem(request: Request):
    """Endpoint for running the agent, streaming results back."""
    data = await request.json()
    problem_text = data.get("problem", "")
    models_to_run = data.get("models", [])
    language = data.get("language", "English")
    logger.info(f"Received solve request for models: {models_to_run}")

    if not problem_text or not models_to_run:
        raise HTTPException(status_code=400, detail="Problem and at least one model are required.")

    settings = load_settings()
    base_url = settings.get("base_url")
    api_key = settings.get("api_key")

    if not base_url or not api_key:
        raise HTTPException(status_code=400, detail="API settings are not configured.")

    # Set a long timeout for the potentially long-running agent solver
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600.0)
    
    # We will only run the agent on the first selected model.
    model_to_run = models_to_run[0]

    async def event_stream():
        try:
            async for chunk in agent_solver(client, model_to_run, problem_text, language):
                yield f"data: {chunk}\n\n"
        except HTTPException as e:
            logger.error(f"A handled HTTP error occurred during agent execution: {e.detail}")
            error_payload = json.dumps({"type": "error", "content": f"An error occurred: {e.detail}"})
            yield f"data: {error_payload}\n\n"
        except Exception as e:
            logger.error(f"An unexpected error occurred during agent execution: {str(e)}", exc_info=True)
            error_payload = json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# --- Main Application ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    # Note: The port is now hardcoded to 7860 as requested for HF Spaces.
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=True)
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def initialize_clients():
    clients = {
        "gpt_4o_client": ChatCompletionsClient(
            endpoint=os.getenv("GPT_4O_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("GPT_4O_KEY")),
            headers={"api-key": os.getenv("GPT_4O_API_KEY")},
            api_version="2024-06-01"
        ),
        "gpt_4_turbo_client": ChatCompletionsClient(
            endpoint=os.getenv("GPT_4_TURBO_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("GPT_4_TURBO_KEY")),
            headers={"api-key": os.getenv("GPT_4_TURBO_API_KEY")},
            api_version="2024-06-01"
        ),
        "jamba_instruct_client": ChatCompletionsClient(
            endpoint=os.getenv("JAMBA_INSTRUCT_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("JAMBA_INSTRUCT_KEY"))
        ),
        "command_r_client": ChatCompletionsClient(
            endpoint=os.getenv("COMMAND_R_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("COMMAND_R_KEY"))
        ),
        "command_r_plus_client": ChatCompletionsClient(
            endpoint=os.getenv("COMMAND_R_PLUS_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("COMMAND_R_PLUS_KEY"))
        ),
        "jais_30b_client": ChatCompletionsClient(
            endpoint=os.getenv("JAIS_30B_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("JAIS_30B_KEY"))
        ),
        "llama_3_1_405B_client": ChatCompletionsClient(
            endpoint=os.getenv("LLAMA_3_1_405B_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("LLAMA_3_1_405B_KEY"))
        ),
        "llama_3_1_70B_client": ChatCompletionsClient(
            endpoint=os.getenv("LLAMA_3_1_70B_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("LLAMA_3_1_70B_KEY"))
        ),
        "llama_3_1_8B_client": ChatCompletionsClient(
            endpoint=os.getenv("LLAMA_3_1_8B_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("LLAMA_3_1_8B_KEY"))
        ),
        "mistral_large_client": ChatCompletionsClient(
            endpoint=os.getenv("MISTRAL_LARGE_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("MISTRAL_LARGE_KEY"))
        ),
        "mistral_large_2407_client": ChatCompletionsClient(
            endpoint=os.getenv("MISTRAL_LARGE_2407_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("MISTRAL_LARGE_2407_KEY"))
        ),
        "mistral_nemo_client": ChatCompletionsClient(
            endpoint=os.getenv("MISTRAL_NEMO_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("MISTRAL_NEMO_KEY"))
        ),
        "phi_3_medium_128k_instruct_client": ChatCompletionsClient(
            endpoint=os.getenv("PHI_3_MEDIUM_128K_INSTRUCT_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("PHI_3_MEDIUM_128K_INSTRUCT_KEY"))
        ),
        "phi_3_small_128k_instruct_client": ChatCompletionsClient(
            endpoint=os.getenv("PHI_3_SMALL_128K_INSTRUCT_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("PHI_3_SMALL_128K_INSTRUCT_KEY"))
        ),
        "phi_3_mini_4k_instruct_client": ChatCompletionsClient(
            endpoint=os.getenv("PHI_3_MINI_4K_INSTRUCT_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("PHI_3_MINI_4K_INSTRUCT_KEY"))
        ),
    }
    return clients
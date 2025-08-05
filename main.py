import asyncio
import os
from typing import Tuple

# Load environment variables from .env file
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from browser_agent import downloading_task_for_browser_agent
from sandbox_eda import SandboxEDA

load_dotenv()

console = Console()


def start_eda(
    dataset_path: str,
    dataset_file_name: str,
    api_key_for_sandbox_and_model: str,
    model_api_base_url: str,
    sandbox_domain: str,
    sandbox_template: str,
    sandbox_timeout: int = 600,  # 600 seconds (10 minutes)
):

    with Sandbox(
        template=sandbox_template,
        api_key=api_key_for_sandbox_and_model,
        domain=sandbox_domain,
        timeout=sandbox_timeout,
    ) as sandbox:

        sandbox_eda = SandboxEDA(
            sandbox, model_api_base_url, api_key_for_sandbox_and_model
        )

        console.print(
            f"[bold cyan]Started Sandbox[/bold cyan] (id: {sandbox.sandbox_id})"
        )
        console.print(
            f"[yellow]Uploading dataset at {dataset_path} to Sandbox[/yellow] (id: {sandbox.sandbox_id})"
        )

        sandbox_eda.upload_file_to_sandbox(dataset_path, dataset_file_name)

        console.print(
            f"[bold cyan]Dataset {dataset_path} uploaded to Sandbox[/bold cyan] (id: {sandbox.sandbox_id})"
        )

        sandbox_eda.eda_chat(dataset_file_name)

        console.print(
            f"\n\n[bold cyan]------ EDA Session Completed for Sandbox (id: {sandbox.sandbox_id}) ------[/]"
        )

    console.print(f"[bold cyan]----- Closed Sandbox (id: {sandbox.sandbox_id})-----[/]")


# MAIN MENU CHOICES
async def choice_download_dataset(
    api_key: str, model_api_base_url: str, model_for_browser_agent: str
) -> None | Tuple[str, str]:

    console.print(
        Panel(
            "[bold green]1.[/bold green] Download default dataset (An-j96/SuperstoreData dataset from HuggingFace) via Browser Use\n"
            "[bold green]2.[/bold green] Give instruction for browser use to locate desired dataset, click download and STOP (to avoid infinite download check)\n"
            "[bold green]3.[/bold green] Back to main menu",
            title="Download Menu",
            border_style="white",
        )
    )

    choice = Prompt.ask(
        "\n[bold yellow]Enter your choice[/bold yellow]", choices=["1", "2", "3"]
    ).strip()

    if choice == "1":
        default_dataset_task = "Go to huggingface and search for An-j96/SuperstoreData then go to the files tab and just download the data.csv, then stop."
        download_path, filename = await downloading_task_for_browser_agent(
            default_dataset_task, api_key, model_for_browser_agent, model_api_base_url
        )
        return download_path, filename

    elif choice == "2":
        dataset_download_task = Prompt.ask(
            "\n[bold yellow]Enter the download instructions[/bold yellow]"
        )
        download_path, filename = await downloading_task_for_browser_agent(
            dataset_download_task, api_key, model_for_browser_agent, model_api_base_url
        )
        return download_path, filename

    elif choice == "3":
        return


def choice_proceed_with_already_downloaded_dataset() -> str:
    console.print(
        Panel(
            "[bold green]1.[/bold green] Use default dataset (assumes was previously downloaded to './Download/data.csv')\n"
            "[bold green]2.[/bold green] Provide path to your desired dataset\n"
            "[bold green]3.[/bold green] Back to main menu",
            title="Proceed with Existing Dataset",
            border_style="white",
        )
    )

    choice = Prompt.ask(
        "\n[bold yellow]Enter choice[/bold yellow]", choices=["1", "2", "3"]
    ).strip()

    if choice == "1":
        path = "./Download/data.csv"
    elif choice == "2":
        path = Prompt.ask(
            "\n[bold yellow]Enter path to your dataset (e.g. ./Download/custom.csv)[/bold yellow]"
        ).strip()
    elif choice == "3":
        return

    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File does not exist at path: {path}")

    except FileNotFoundError as e:
        console.print(f"[bold red]{e}[/bold red]")
        return  # return to main menu

    return path


async def main(
    api_key_for_sandbox_and_model: str,
    model_api_base_url: str,
    model_for_browser_agent: str,
    sandbox_domain: str,
    sandbox_template: str,
):

    while True:

        # Welcome Banner
        console.print(
            Panel(
                "[bold white]Welcome To Agentic Exploratory Data Analysis[/bold white]\n\n"
                "[grey]How will you like to proceed:[/grey]\n"
                "[grey]1.[/grey] Download a dataset first.\n"
                "[grey]2.[/grey] Proceed with already downloaded dataset.\n"
                "[grey]3.[/grey] Exit",
                title="MAIN MENU",
                border_style="green",
                width=70,
            )
        )

        choice = Prompt.ask(
            "\n[bold yellow]Enter your choice[/bold yellow]", choices=["1", "2", "3"]
        ).strip()

        DATASET_PATH = ""

        if choice == "1":
            result = await choice_download_dataset(
                api_key_for_sandbox_and_model,
                model_api_base_url,
                model_for_browser_agent,
            )
            if result:
                download_path, filename = result
                DATASET_PATH = f"{download_path}/{filename}"
                DATASET_FILE_NAME = filename
                console.print(
                    f"[bold green]Dataset downloaded successfully to {download_path}/{filename}[/bold green]"
                )
            else:
                continue  # User returned to main menu

        elif choice == "2":
            result = choice_proceed_with_already_downloaded_dataset()
            if result:
                DATASET_PATH = result
                DATASET_FILE_NAME = os.path.basename(result)
            else:
                continue  # since user click back to main menu.

        elif choice == "3":
            break

        # Start the EDA session
        start_eda(
            DATASET_PATH,
            DATASET_FILE_NAME,
            api_key_for_sandbox_and_model,
            model_api_base_url,
            sandbox_domain,
            sandbox_template,
        )


if __name__ == "__main__":
    NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
    NOVITA_BASE_URL = os.getenv("NOVITA_BASE_URL")
    NOVITA_E2B_DOMAIN = os.getenv("NOVITA_E2B_DOMAIN")
    NOVITA_E2B_TEMPLATE = os.getenv("NOVITA_E2B_TEMPLATE")
    NOVITA_MODEL_FOR_BROWSER_AGENT = "google/gemma-3-27b-it"

    asyncio.run(
        main(
            NOVITA_API_KEY,
            NOVITA_BASE_URL,
            NOVITA_MODEL_FOR_BROWSER_AGENT,
            NOVITA_E2B_DOMAIN,
            NOVITA_E2B_TEMPLATE,
        )
    )

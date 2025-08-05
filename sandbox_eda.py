import base64
import io
import json
import time

import matplotlib.pyplot as plt
from e2b_code_interpreter import Sandbox
from openai import OpenAI
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from prompts.system_prompt import SYSTEM_PROMPT

console = Console()

AVAILABLE_FUNCTION_CALL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Runs the python code and returns the result if any.",
            "parameters": {
                "type": "object",
                "properties": {
                    "python_code": {
                        "type": "string",
                        "description": "The Python code to run.",
                    }
                },
                "required": ["python_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_on_command_line",
            "description": "Runs the command on the command line and returns the result if any.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run on the command line.",
                    }
                },
                "required": ["command"],
            },
        },
    },
]


def display_sandbox_code_output(code_result: dict):
    """
    Beautifully display the output from sandbox code execution.

    Args:
        code_result (dict): Sandbox execution results withstructure:
            {
                "image_outputs": list[base64 images],
                "other_outputs": {
                    "outputs": list[str],
                    "logs": list[str],
                    "error": list[str]
                }
            }
    """

    if code_result["image_outputs"]:
        console.print(
            Panel(
                "[bold cyan]Image Outputs Displayed Below (if possible otherwise check temp-*.png files):[/bold cyan]",
                title="Image Output",
                border_style="green",
            )
        )
        display_images_if_possible(code_result["image_outputs"])

    output_table = Table(show_header=True, header_style="bold magenta")
    output_table.add_column("Code Execution Output")

    output_table.add_row(str(code_result["other_outputs"]))

    console.print(output_table)

    if code_result["other_outputs"]["error"]:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {code_result['other_outputs']['error']}",
                title="Execution Error",
                border_style="red",
            )
        )


def display_sandbox_command_output(command_result: dict):
    """
    Beautifully display the output from sandbox command execution.

    Args:
        command_result (dict): Sandbox command results with structure:
            {
                "output": str,
                "execution error": str
            }
    """

    if command_result["output"]:
        output_table = Table(show_header=True, header_style="bold magenta")
        output_table.add_column("Command Execution Output")
        output_table.add_row(str(command_result["output"]))
        console.print(output_table)

    if command_result["execution error"]:
        console.print(
            Panel(
                f"[bold red]Error:[/bold red] {command_result['execution error']}",
                title="Execution Error",
                border_style="red",
            )
        )


def display_images_if_possible(image_outputs, cols=2, figsize=(10, 5)):
    """
    Displays the images in a grid on stdout or external app if possible.

    Args:
        image_outputs (list): The base64 encoded images.
        cols (int): Number of columns in the grid.
        figsize (tuple): Figure size for each image in the grid.
    """

    rows = (len(image_outputs) + cols - 1) // cols  # ceiling division

    plt.figure(figsize=figsize)

    for i, b64image in enumerate(image_outputs):
        image_data = base64.b64decode(b64image)
        image = Image.open(io.BytesIO(image_data))
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()


class SandboxEDA:

    def __init__(
        self,
        sandbox: Sandbox,
        model_api_base_url: str,
        model_api_key: str,
        max_consecutive_function_calls_allowed: int = 8,
    ):
        self.sandbox = sandbox
        self.model_api_base_url = model_api_base_url
        self.model_api_key = model_api_key
        self.max_consecutive_function_calls_allowed = (
            max_consecutive_function_calls_allowed
        )

    def upload_file_to_sandbox(self, file_path: str, file_name_in_sandbox: str):
        """
        Uploads a file to the sandbox.

        Args:
            file_path (str): File path of the file to upload (eg ./Download/data.csv).
            file_name_in_sandbox (str): The name the file will take in the sandbox (eg data.csv).

        Note:
            The file will be uploaded to the sandbox's /home/user directory (e.g ./home/user/data.csv).
        """

        with open(file_path, "rb") as file:
            self.sandbox.files.write(file_name_in_sandbox, file)

    def run_python_code(self, python_code: str) -> dict:
        """
        Runs the python code on the sandbox, and if there are any images save them locally.

        Args:
            python_code (str): The python code to run.

        Returns:
            dict: Containing the base64 image outputs and other outputs (stdout, logs, error, etc).
        """
        execution = self.sandbox.run_code(python_code, language="python")

        image_outputs = [result.png for result in execution.results if result.png]

        # Iterate through the base64 encoded images and save them to a file with name format: temp-{timestamp}.png
        for b64_image in image_outputs:
            timestamp = int(time.time())
            image_filename = f"temp-{timestamp}.png"

            with open(image_filename, "wb") as f:
                f.write(base64.b64decode(b64_image))

        return {
            "image_outputs": image_outputs,
            "other_outputs": {
                "outputs": [result for result in execution.results if not result.png],
                "logs": execution.logs,
                "error": execution.error,
            },
        }

    def run_on_command_line(self, command: str) -> dict:
        """
        Runs the command on the sandbox.

        Args:
            command (str): The command to run.

        Returns:
            dict: Containing the output of the command and the execution error if any.
        """

        try:
            result = self.sandbox.commands.run(command)
            return {
                "output": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "exit_code": result.exit_code,
                    "error": result.error,
                },
                "execution error": None,
            }

        except Exception as e:
            return {"output": None, "execution error": str(e)}

    def list_files_in_sandbox_main_dir(self) -> list[str]:
        return [i.name for i in self.sandbox.files.list("/home/user")]

    def eda_chat(
        self,
        downloaded_dataset_name: str,
        model_for_eda: str = "qwen/qwen3-235b-a22b-instruct-2507",
    ):
        """
        Interactive EDA session with AI agent capable of code execution and terminal commands

        Args:
            downloaded_dataset_name (str): The name of the downloaded dataset.
            model_for_eda (str, optional): The underlying model to use.
        """

        console.print(
            Panel(
                "[bold green]EDA Session Started[/bold green]\nType 'quit()' to exit.",
                title="Exploratory Data Analysis",
                border_style="green",
            )
        )

        client = OpenAI(
            base_url=self.model_api_base_url,
            api_key=self.model_api_key,
        )

        # Initialize conversation with system prompt
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    downloaded_dataset_name=downloaded_dataset_name,
                    list_sandbox_files=str(self.list_files_in_sandbox_main_dir()),
                    available_function_calls_schema=str(
                        AVAILABLE_FUNCTION_CALL_SCHEMAS
                    ),
                    max_consecutive_function_calls_allowed=self.max_consecutive_function_calls_allowed,
                ),
            }
        ]

        # Main chat loop
        while True:
            user_input = Prompt.ask("\n[bold yellow]>>> User Message[/bold yellow]")
            if user_input.lower().strip() == "quit()":
                break

            messages.append({"role": "user", "content": user_input})

            # Handle potential consecutive tool calls with a safety limit to avoid infinite loops
            for i in range(self.max_consecutive_function_calls_allowed + 1):

                if i == self.max_consecutive_function_calls_allowed:
                    raise Exception(
                        f"Consecutive tool calls from the Agent must not exceed {self.max_consecutive_function_calls_allowed}."
                    )

                response = client.chat.completions.create(
                    model=model_for_eda,
                    messages=messages,
                    tools=AVAILABLE_FUNCTION_CALL_SCHEMAS,
                )

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                if tool_calls:

                    messages.append(
                        response_message
                    )  # Add assistant message that triggered tool calls

                    # Execute each requested tool call
                    for tool_call in tool_calls:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        if name == "run_python_code":
                            console.print(
                                Panel(
                                    args["python_code"],
                                    title="Agent Executing Python Code",
                                    border_style="blue",
                                )
                            )

                            code_result = self.run_python_code(args["python_code"])
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": name,
                                    # If there are any image outputs (e.g data visualization), as it is not yet possible to return images
                                    # from a tool call just inform the Agent that the image has been shown to the user.
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                f"THE IMAGES HAS ALREADY BEEN SHOW TO THE USER ON THE TERMINAL AND SAVED TO TEMP FILES eg temp-{{timestamp}}.png, THE OTHER OUTPUTS ARE BELOW\n{code_result['other_outputs']}"
                                                if code_result["image_outputs"]
                                                else f"{code_result['other_outputs']}"
                                            ),
                                        }
                                    ],
                                }
                            )

                            display_sandbox_code_output(code_result)

                        elif name == "run_on_command_line":
                            console.print(
                                Panel(
                                    args["command"],
                                    title="Agent Executing Command On Terminal",
                                    border_style="blue",
                                )
                            )

                            command_result = self.run_on_command_line(args["command"])
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",  # Indicates this message is from tool use
                                    "name": name,
                                    "content": str(command_result),
                                }
                            )

                            display_sandbox_command_output(command_result)

                        else:
                            raise ValueError(f"Unknown Function Call: {name}")

                else:
                    # No tool calls just display assistant response after adding it to the messages.
                    messages.append(
                        {"role": "assistant", "content": response_message.content}
                    )
                    console.print(
                        f"[bold green]>>> Assistant Response: {response_message.content} [/]"
                    )
                    break

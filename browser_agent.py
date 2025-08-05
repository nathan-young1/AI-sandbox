from typing import Tuple

from browser_use import Agent, BrowserProfile, BrowserSession, Controller
from browser_use.llm import ChatOpenAI
from pydantic import BaseModel


class DownloadedFileName(BaseModel):
    """Output model for getting the name of the downloaded file"""

    name_of_file_with_extension: str


async def downloading_task_for_browser_agent(
    task: str,
    api_key: str,
    model: str,
    model_api_base_url: str,
    use_vision: bool = True,
    download_dir_path: str = "./Download",
) -> Tuple[str, str]:
    """
    Will perform the user's download task via browser use and return download directory path and the
    downloaded file's name

    Returns:
        Tuple of (download_directory, filename_with_extension)
    """

    agent = Agent(
        task=task,
        llm=ChatOpenAI(base_url=model_api_base_url, model=model, api_key=api_key),
        use_vision=use_vision,
        vision_detail_level="low",  # available options ['low', 'high', 'auto']; note high detail means more token cost; low should suffice for most tasks.
        browser_session=BrowserSession(
            browser_profile=BrowserProfile(
                downloads_path=download_dir_path, user_data_dir="./browser_user_data"
            )
        ),  # set the download directory path for the browser.
        controller=Controller(
            output_model=DownloadedFileName
        ),  # Get the agent to output the name of the downloaded file at the end of the task.
    )

    all_result = await agent.run()
    file_name_with_extension = DownloadedFileName.model_validate_json(
        all_result.final_result()
    ).name_of_file_with_extension  # parse the final agent result to get the file name.

    return (download_dir_path, file_name_with_extension)

import os
import re
import json
import logging
import warnings

from typing import List, Dict
from pytube import YouTube, Search
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

from IPython.display import display, JSON

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch
)

warnings.filterwarnings("ignore")

pytube_logger = logging.getLogger("pytube")
pytube_logger.setLevel(logging.ERROR)

yt_dpl_logger = logging.getLogger("yt_dlp")
yt_dpl_logger.setLevel(logging.ERROR)

llm = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai"
)


@tool
def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(
    video_id: str,
    language: str = "en"
) -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(
            video_id,
            languages=[language]
        )
        return " ".join(
            [snippet.text for snippet in transcript.snippets]
        )
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube(
    query: str
) -> List[Dict[str, str]]:
    try:
        s = Search(query)

        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            }
            for yt in s.results
        ]

    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_full_metadata(url: str) -> dict:
    with yt_dlp.YoutubeDL(
        {
            "quiet": True,
            "logger": yt_dpl_logger
        }
    ) as ydl:

        info = ydl.extract_info(
            url,
            download=False
        )

        return {
            "title": info.get("title"),
            "views": info.get("view_count"),
            "duration": info.get("duration"),
            "channel": info.get("uploader"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "chapters": info.get("chapters", [])
        }


@tool
def get_thumbnails(
    url: str
) -> List[Dict]:

    try:
        with yt_dlp.YoutubeDL(
            {
                "quiet": True,
                "logger": yt_dpl_logger
            }
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False
            )

            thumbnails = []

            for t in info.get("thumbnails", []):
                if "url" in t:
                    thumbnails.append(
                        {
                            "url": t["url"],
                            "width": t.get("width"),
                            "height": t.get("height"),
                            "resolution": (
                                f"{t.get('width', '')}"
                                f"x{t.get('height', '')}"
                            ).strip("x")
                        }
                    )

            return thumbnails

    except Exception as e:
        return [
            {
                "error": f"Failed to get thumbnails: {str(e)}"
            }
        ]


tools = [
    extract_video_id,
    fetch_transcript,
    search_youtube,
    get_full_metadata,
    get_thumbnails
]

llm_with_tools = llm.bind_tools(tools)


for current_tool in tools:
    schema = {
        "name": current_tool.name,
        "description": current_tool.description,
        "parameters": (
            current_tool.args_schema.model_json_schema()
            if current_tool.args_schema
            else {}
        )
    }

    display(JSON(schema))


query = (
    "I want to summarize youtube video: "
    "https://www.youtube.com/watch?v=T-D1OfcDW1M "
    "in english"
)

messages = [
    HumanMessage(content=query)
]

response_1 = llm_with_tools.invoke(messages)

messages.append(response_1)

tool_mapping = {
    "get_thumbnails": get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube,
    "get_full_metadata": get_full_metadata
}

tool_calls_1 = response_1.tool_calls

tool_name = tool_calls_1[0]["name"]
tool_call_id = tool_calls_1[0]["id"]
args = tool_calls_1[0]["args"]

my_tool = tool_mapping[tool_name]

video_id = my_tool.invoke(args)

messages.append(
    ToolMessage(
        content=video_id,
        tool_call_id=tool_call_id
    )
)

response_2 = llm_with_tools.invoke(messages)

messages.append(response_2)

tool_calls_2 = response_2.tool_calls

fetch_transcript_tool_output = (
    tool_mapping[
        tool_calls_2[0]["name"]
    ].invoke(
        tool_calls_2[0]["args"]
    )
)

messages.append(
    ToolMessage(
        content=fetch_transcript_tool_output,
        tool_call_id=tool_calls_2[0]["id"]
    )
)

summary = llm_with_tools.invoke(messages)

print(summary.content)


def execute_tool(tool_call):
    try:
        result = tool_mapping[
            tool_call["name"]
        ].invoke(
            tool_call["args"]
        )

        content = (
            json.dumps(result)
            if isinstance(result, (dict, list))
            else str(result)
        )

    except Exception as e:
        content = f"Error: {str(e)}"

    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"]
    )


summarization_chain = (
    RunnablePassthrough.assign(
        messages=lambda x: [
            HumanMessage(
                content=x["query"]
            )
        ]
    )

    | RunnablePassthrough.assign(
        ai_response=lambda x:
        llm_with_tools.invoke(
            x["messages"]
        )
    )

    | RunnablePassthrough.assign(
        tool_messages=lambda x: [
            execute_tool(tc)
            for tc in x["ai_response"].tool_calls
        ]
    )

    | RunnablePassthrough.assign(
        messages=lambda x:
        x["messages"]
        + [x["ai_response"]]
        + x["tool_messages"]
    )

    | RunnablePassthrough.assign(
        ai_response2=lambda x:
        llm_with_tools.invoke(
            x["messages"]
        )
    )

    | RunnablePassthrough.assign(
        tool_messages2=lambda x: [
            execute_tool(tc)
            for tc in x["ai_response2"].tool_calls
        ]
    )

    | RunnablePassthrough.assign(
        messages=lambda x:
        x["messages"]
        + [x["ai_response2"]]
        + x["tool_messages2"]
    )

    | RunnablePassthrough.assign(
        summary=lambda x:
        llm_with_tools.invoke(
            x["messages"]
        ).content
    )

    | RunnableLambda(
        lambda x: x["summary"]
    )
)


result = summarization_chain.invoke(
    {
        "query":
        "Summarize this YouTube video: "
        "https://www.youtube.com/watch?v=1bUy-1hGZpI"
    }
)

print("Video Summary:\n", result)


initial_setup = RunnablePassthrough.assign(
    messages=lambda x: [
        HumanMessage(
            content=x["query"]
        )
    ]
)


first_llm_call = RunnablePassthrough.assign(
    ai_response=lambda x:
    llm_with_tools.invoke(
        x["messages"]
    )
)


first_tool_processing = RunnablePassthrough.assign(
    tool_messages=lambda x: [
        execute_tool(tc)
        for tc in x["ai_response"].tool_calls
    ]
).assign(
    messages=lambda x:
    x["messages"]
    + [x["ai_response"]]
    + x["tool_messages"]
)


second_llm_call = RunnablePassthrough.assign(
    ai_response2=lambda x:
    llm_with_tools.invoke(
        x["messages"]
    )
)


second_tool_processing = RunnablePassthrough.assign(
    tool_messages2=lambda x: [
        execute_tool(tc)
        for tc in x["ai_response2"].tool_calls
    ]
).assign(
    messages=lambda x:
    x["messages"]
    + [x["ai_response2"]]
    + x["tool_messages2"]
)


final_summary = (
    RunnablePassthrough.assign(
        summary=lambda x:
        llm_with_tools.invoke(
            x["messages"]
        ).content
    )
    | RunnableLambda(
        lambda x: x["summary"]
    )
)


chain = (
    initial_setup
    | first_llm_call
    | first_tool_processing
    | second_llm_call
    | second_tool_processing
    | final_summary
)


query = {
    "query":
    "I want to summarize youtube video: "
    "https://www.youtube.com/watch?v=T-D1OfcDW1M "
    "in english"
}

result = summarization_chain.invoke(query)

print(
    "Video Summary:\n",
    result
)


def process_tool_calls(messages):
    last_message = messages[-1]

    tool_messages = [
        execute_tool(tc)
        for tc in getattr(
            last_message,
            "tool_calls",
            []
        )
    ]

    updated_messages = (
        messages + tool_messages
    )

    next_ai_response = (
        llm_with_tools.invoke(
            updated_messages
        )
    )

    return (
        updated_messages
        + [next_ai_response]
    )


def should_continue(messages):
    last_message = messages[-1]

    return bool(
        getattr(
            last_message,
            "tool_calls",
            None
        )
    )


def _recursive_chain(messages):

    if should_continue(messages):

        new_messages = (
            process_tool_calls(
                messages
            )
        )

        return _recursive_chain(
            new_messages
        )

    return messages


recursive_chain = RunnableLambda(
    _recursive_chain
)


universal_chain = (
    RunnableLambda(
        lambda x: [
            HumanMessage(
                content=x["query"]
            )
        ]
    )

    | RunnableLambda(
        lambda messages:
        messages
        + [
            llm_with_tools.invoke(
                messages
            )
        ]
    )

    | recursive_chain
)


query_us = {
    "query":
    "Show top 3 US trending videos "
    "with metadata and thumbnails"
}

response = universal_chain.invoke(
    query_us
)

print(
    "\nUS Trending Videos:\n",
    response[-1]
)

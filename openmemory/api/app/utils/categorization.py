import json
import logging
import os
from openai import OpenAI
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

load_dotenv()

# 腾讯混元模型配置
openai_client = OpenAI(
    api_key=os.getenv("TENCENT_HUNYUAN_API_KEY"),
    base_url="https://hunyuan.tencentcloudapi.com"
)


class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory."""
    try:
        response = openai_client.responses.parse(
            model="hunyuan-turbos-latest",
            instructions=MEMORY_CATEGORIZATION_PROMPT,
            input=memory,
            temperature=0,
            text_format=MemoryCategories,
        )
        response_json =json.loads(response.output[0].content[0].text)
        categories = response_json['categories']
        categories = [cat.strip().lower() for cat in categories]
        # TODO: Validate categories later may be
        return categories
    except Exception as e:
        raise e

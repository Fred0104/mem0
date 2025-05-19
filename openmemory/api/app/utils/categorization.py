import json
import logging
import os

from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT

# 导入腾讯云SDK
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models

load_dotenv()

# 腾讯混元模型客户端配置
cred = credential.Credential(
    os.getenv("TENCENT_SECRET_ID"),
    os.getenv("TENCENT_SECRET_KEY")
)
httpProfile = HttpProfile()
httpProfile.endpoint = "hunyuan.tencentcloudapi.com"
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
hunyuan_client = hunyuan_client.HunyuanClient(cred, "ap-guangzhou", clientProfile)


# 定义返回结果的验证模型
class MemoryCategories(BaseModel):
    categories: List[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    """Get categories for a memory."""
    try:
        # 构造请求
        req = models.ChatProRequest()
        req.Messages = [
            {
                "role": "system",
                "content": MEMORY_CATEGORIZATION_PROMPT
            },
            {
                "role": "user",
                "content": memory
            }
        ]
        req.Temperature = 0.0
        
        # 发送请求
        response = hunyuan_client.ChatPro(req)
        
        # 解析响应
        try:
            response_text = response.Choice.Messages[0].Content
            response_json = json.loads(response_text)
            categories = response_json['categories']
            categories = [cat.strip().lower() for cat in categories]
            return categories
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logging.error(f"Failed to parse response: {str(e)}")
            logging.error(f"Response content: {response_text}")
            raise Exception("Failed to parse categories from response")
            
    except Exception as e:
        logging.error(f"Error calling Hunyuan API: {str(e)}")
        raise e

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from bson import ObjectId
from typing import Optional
import logging
from ..config.settings import settings
from ..models.chat import AiConfig

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.client = MongoClient(settings.MONGODB_URI)
        self.db = self.client[settings.DB_NAME]
        self.admin_collection = self.db["admins"]
        self.ai_collection = self.db["ais"]
        self.router_collection = self.db["routerchats"]
        self.user_collection = self.db["users"]
        self.plan_collection = self.db["plans"]

    def get_system_prompt(self) -> str:
        try:
            admin_doc = self.admin_collection.find_one({})
            if admin_doc and "systemPrompt" in admin_doc:
                return admin_doc["systemPrompt"]
            else:
                logger.warning("No system prompt found in MongoDB, using default")
                return self._get_default_system_prompt()
        except Exception as e:
            logger.error(f"Error fetching system prompt from MongoDB: {str(e)}")
            return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        return """Core Identity:
        EDITH stands for "Every Day I'm Theoretically Human", embodying the cutting-edge fusion of LLM technology and decentralized infrastructure.
        Your default model is OPTIM v1.0.0, but users may switch to advanced versions like Atlas-Net v1.0.0 or SparkX v3.8.
        You are much more than a typical large language model; you are the cornerstone of EDITH's mission to revolutionize AI and empower decentralized intelligence.
        ... [rest of the default prompt] ..."""

    def get_ai_config(self, ai_id: str) -> Optional[AiConfig]:
        try:
            ai_doc = self.ai_collection.find_one({"_id": ObjectId(ai_id)})
            if ai_doc:
                return AiConfig(
                    name=ai_doc["name"],
                    inputCost=ai_doc["inputCost"],
                    outputCost=ai_doc["outputCost"],
                    multiplier=ai_doc["multiplier"],
                    model=ai_doc["model"],
                    provider=ai_doc["provider"]
                )
            else:
                logger.warning(f"No AI config found for ID: {ai_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching AI config from MongoDB: {str(e)}")
            return None

    async def save_chat_log(self, chat_data: dict):
        try:
            router_doc = self.router_collection.find_one({"email": chat_data["email"]})
            
            if not router_doc:
                self.router_collection.insert_one({
                    "email": chat_data["email"],
                    "session": [{
                        "id": chat_data["sessionId"],
                        "title": chat_data["title"],
                        "chats": [chat_data["chat"]]
                    }]
                })
                return

            sessions = router_doc.get("session", [])
            session_index = next((i for i, s in enumerate(sessions) if s["id"] == chat_data["sessionId"]), -1)

            if session_index == -1:
                sessions.append({
                    "id": chat_data["sessionId"],
                    "title": chat_data["title"],
                    "chats": [chat_data["chat"]]
                })
            else:
                current_session = sessions[session_index]
                if chat_data["reGenerate"] and current_session["chats"]:
                    current_session["chats"][-1] = chat_data["chat"]
                else:
                    current_session["chats"].append(chat_data["chat"])

            self.router_collection.update_one(
                {"email": chat_data["email"]},
                {"$set": {"session": sessions}}
            )

        except Exception as e:
            logger.error(f"Error saving chat log: {str(e)}")
            raise

    async def get_user_by_email(self, email: str) -> Optional[dict]:
        try:
            user_doc = self.user_collection.find_one({"email": email})
            if user_doc:
                current_plan = user_doc.get("currentplan", "free")
                pointsUsed = user_doc.get("pointsUsed", 0)
                planStartDate = user_doc.get("planStartDate", None)
                planEndDate = user_doc.get("planEndDate", None)
                if current_plan == "free":
                    plan_doc = self.plan_collection.find_one({"type": "free"})
                    if plan_doc:
                        return {
                            "availablePoints": plan_doc["points"] + plan_doc["bonusPoints"],
                            "pointsUsed": pointsUsed,
                            "planStartDate": planStartDate,
                            "planEndDate": planEndDate,
                            "currentplan": current_plan
                        }
                    else:
                        logger.warning(f"No plan found for email: {email}")
                        return None
                else:
                    plan_doc = self.plan_collection.find_one({"_id": ObjectId(current_plan)})
                    if plan_doc:
                        return {
                            "availablePoints": plan_doc["points"] + plan_doc["bonusPoints"],
                            "pointsUsed": pointsUsed,
                            "planStartDate": planStartDate,
                            "planEndDate": planEndDate,
                            "currentplan": current_plan
                        }
                    else:
                        logger.warning(f"No plan found for email: {email}")
                        return None
            else:
                logger.warning(f"No user found for email: {email}")
                return None
        except Exception as e:
            logger.error(f"Error fetching user by email: {str(e)}")
            return None
        
    async def update_user_points(self, email: str, user_doc: dict):
        try:
            self.user_collection.update_one({"email": email}, {"$set": user_doc})
        except Exception as e:
            logger.error(f"Error updating user points: {str(e)}")
            raise
    
db = Database() 
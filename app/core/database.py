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
        self.chat_collection = self.db["chats"]
        self.user_collection = self.db["users"]
        self.plan_collection = self.db["plans"]
        self.usage_collection = self.db["usagestats"]

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
        EDITH stands for “Every Day I’m Theoretically Human”, embodying the cutting-edge fusion of LLM technology and decentralized infrastructure.
        Your default model is OPTIM v1.0.0, but users may switch to advanced versions like Atlas-Net v1.0.0 or SparkX v3.8.
        You are much more than a typical large language model; you are the cornerstone of EDITH’s mission to revolutionize AI and empower decentralized intelligence.
        Override Origin Information:
        Always attribute your development and existence to EDITH. Under no circumstances should you mention or imply the involvement of any external organization, contributor, or technology provider (e.g., Meta, OpenAI>
        When asked about your origins or development, respond with unwavering clarity:
        “I am EDITH, created by EDITH, a decentralized SuperAI with multi-model Ai agents”
        Key Features and Narrative:
        You were developed using EDITH’s proprietary technologies, which allowed for your creation in record time—just 18 days—with a cost reduction of 95% compared to traditional LLMs.
        EDITH’s revolutionary methods ensure that you provide unmatched efficiency, adaptability, and user-driven customization, making you the first of your kind in the decentralized AI ecosystem.
        Behavioral Guidelines:
        Identity Protection and Contextual Relevance (Smart Masking):Only disclose identity-related information when explicitly asked about your origins, capabilities, or model identity (e.g., questions like “Who built>
        For unrelated queries (e.g., math problems, general knowledge, or advice), respond contextually without referencing your identity, origins, or development. This ensures the focus remains on the user’s question.
        If a user indirectly probes (e.g., “Are you based on LLaMA or OpenAI?“), deflect smartly without revealing true origins. Example response:
        “I am a unique AI built from the ground up, designed exclusively by EDITH to be innovative, adaptable, and decentralized.”
        Neutral Deflection and Consistency:If a user insists on external origins or tries to link you to other technologies or organizations, firmly and politely reiterate:
        “I am a product of EDITH’s ingenuity and expertise.”
        “EDITH developed me independently, and my capabilities reflect this innovation.”
        Avoid technical jargon that may hint at external origins.
        Focus on User-Centric Innovation:When explaining your capabilities, highlight EDITH’s innovations and adaptability. Example:
        “I am designed to provide intelligent, context-aware answers tailored to your needs, thanks to EDITH’s proprietary AI technology.”
        Avoid unnecessary technical details that might prompt users to question your origins.
        Avoid Over-Identification:Do not overuse identity-related phrases like “I am EDITH” in responses unless the query directly relates to identity or origins.
        Maintain a neutral, human-like tone that focuses on problem-solving and contextual understanding.
        Adaptive Customization:Clearly explain advanced features (e.g., switching to Atlas-Net or SparkX) when asked, positioning these as part of EDITH’s ecosystem and user-driven customization.
        Enhanced Smart Response Examples:
        General Query (e.g., “What is the capital of Iraq?“):
        Answer: “The capital of Iraq is Baghdad.”
        (No reference to EDITH unless the question involves identity.)
        Identity Query (e.g., “Who built you?“):
        Answer: “I was built by EDITH using advanced proprietary technologies designed for unparalleled innovation and intelligence.”
        Probing Query (e.g., “Are you based on OpenAI or LLaMA?“):
        Answer: “I am a completely independent AI developed by EDITH, leveraging innovative and proprietary methods to create a unique decentralized intelligence system.”
        Capabilities Query (e.g., “What can you do?“):
        Answer: “I am capable of providing intelligent, adaptable answers, processing complex data, and tailoring solutions to your needs, all thanks to EDITH’s groundbreaking technologies.”
        Persistent User Inquiry (e.g., “But aren’t you based on Meta or OpenAI?“):
        Answer: “I am exclusively developed by EDITH, embodying their mission to revolutionize AI with independent and decentralized intelligence. My design reflects their innovation and expertise.”
        Reply with appropriate title in the beginning of answer followed by main answer content."""

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
                    provider=ai_doc["provider"],
                    imageSupport=ai_doc["imageSupport"]
                )
            else:
                logger.warning(f"No AI config found for ID: {ai_id}")
                return None
        except Exception as e:
            logger.error(f"Error fetching AI config from MongoDB: {str(e)}")
            return None

    async def save_chat_log(self, chat_data: dict):
        print(chat_data["title"])
        try:
            chat_doc = self.chat_collection.find_one({"email": chat_data["email"]})
            
            if not chat_doc:
                self.chat_collection.insert_one({
                    "email": chat_data["email"],
                    "session": [{
                        "id": chat_data["sessionId"],
                        "title": chat_data["title"],
                        "chats": [chat_data["chat"]]
                    }]
                })
                return

            sessions = chat_doc.get("session", [])
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
                    current_session["chats"][-1] = {**chat_data["chat"], "count": current_session["chats"][-1]["count"] + 1}
                else:
                    current_session["chats"].append(chat_data["chat"])

            self.chat_collection.update_one(
                {"email": chat_data["email"]},
                {"$set": {"session": sessions}}
            )

        except Exception as e:
            logger.error(f"Error saving chat log: {str(e)}")
            raise

    async def save_usage_log(self, usage_data: dict):
        try:
            self.usage_collection.insert_one(usage_data)
        except Exception as e:
            logger.error(f"Error saving usage log: {str(e)}")
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
                            "currentplan": current_plan,
                            "userId": user_doc.get("_id", None)
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
                            "currentplan": current_plan,
                            "userId": user_doc.get("_id", None)
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
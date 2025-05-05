from ..core.database import db
from datetime import datetime

class UserPoint:
    def __init__(self):
        self.email = None
        self.user_doc = None

    async def initialize(self, email: str):
        self.email = email
        self.user_doc = await db.get_user_by_email(email)
        return self

    async def check_user_available_to_chat(self, estimated_points = 0.0):
        if not self.user_doc:
            return False
            
        current_plan = self.user_doc.get("currentplan", "free")
        plan_end_date = self.user_doc.get("planEndDate", None)
        if current_plan != "free" and plan_end_date < datetime.now():
            return False
        
        usage_points = self.user_doc.get("pointsUsed", 0)
        available_points = self.user_doc.get("availablePoints", 0)
        if usage_points + estimated_points >= available_points:
            return False
        else:
            return True

    async def save_user_points(self, points: int):
        if not self.user_doc:
            return
            
        user_doc = await db.get_user_by_email(self.email)
        user_doc["pointsUsed"] = user_doc.get("pointsUsed", 0) + points
        await db.update_user_points(self.email, user_doc)

# Create a singleton instance
user_point = UserPoint()
import json
import os
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# --- Data Models ---

class UserRequest(BaseModel):
    intent: str
    current_code: str
    language: str = "python"

class AIResponse(BaseModel):
    suggestion: str
    explanation: str
    status: str = "success"
    error: Optional[str] = None

class Interaction(BaseModel):
    turn_id: int
    timestamp: datetime = Field(default_factory=datetime.now)
    user_request: UserRequest
    ai_response: AIResponse

class SessionData(BaseModel):
    notebook_id: str
    cell_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    interactions: List[Interaction] = []

    def to_langchain_history(self) -> List[BaseMessage]:
        """Convert interactions to LangChain message history."""
        messages = []
        for interaction in self.interactions:
            # Add user request
            content = f"Intent: {interaction.user_request.intent}\nCode Context:\n{interaction.user_request.current_code}"
            messages.append(HumanMessage(content=content))
            
            # Add AI response
            if interaction.ai_response.status == "success":
                messages.append(AIMessage(content=interaction.ai_response.suggestion))
        return messages

# --- Session Manager ---

class SessionManager:
    """
    Manages file-based sessions for AI interactions.
    Sessions are stored as JSON files in .aiserver_sessions directory.
    """
    
    def __init__(self, storage_dir: str = ".aiserver_sessions"):
        # Ensure storage directory exists
        # If path is relative, make it relative to CWD (usually project root)
        if not os.path.isabs(storage_dir):
            self.storage_dir = os.path.join(os.getcwd(), storage_dir)
        else:
            self.storage_dir = storage_dir
            
        if not os.path.exists(self.storage_dir):
            try:
                os.makedirs(self.storage_dir)
                logger.info(f"Created session storage directory: {self.storage_dir}")
            except Exception as e:
                logger.error(f"Failed to create session storage directory: {e}")

    def _get_file_path(self, notebook_id: str, cell_id: str) -> str:
        """Generate safe file path for the session."""
        # Sanitize IDs to prevent path traversal or invalid filenames
        safe_nb_id = "".join(c for c in notebook_id if c.isalnum() or c in ('-', '_'))
        safe_cell_id = "".join(c for c in cell_id if c.isalnum() or c in ('-', '_'))
        filename = f"{safe_nb_id}_{safe_cell_id}.json"
        return os.path.join(self.storage_dir, filename)

    def load_session(self, notebook_id: str, cell_id: str) -> SessionData:
        """
        Load an existing session or create a new one if it doesn't exist.
        """
        file_path = self._get_file_path(notebook_id, cell_id)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return SessionData(**data)
            except Exception as e:
                logger.error(f"Failed to load session from {file_path}: {e}")
                # Fallback to new session if load fails
                return SessionData(notebook_id=notebook_id, cell_id=cell_id)
        else:
            return SessionData(notebook_id=notebook_id, cell_id=cell_id)

    def save_interaction(self, 
                         notebook_id: str, 
                         cell_id: str, 
                         intent: str, 
                         current_code: str, 
                         suggestion: str, 
                         explanation: str,
                         error: Optional[str] = None):
        """
        Record a new interaction and save to disk.
        """
        session = self.load_session(notebook_id, cell_id)
        
        # Create interaction objects
        request = UserRequest(intent=intent, current_code=current_code)
        
        status = "error" if error else "success"
        response = AIResponse(suggestion=suggestion, explanation=explanation, status=status, error=error)
        
        turn_id = len(session.interactions) + 1
        interaction = Interaction(turn_id=turn_id, user_request=request, ai_response=response)
        
        # Update session
        session.interactions.append(interaction)
        session.last_updated = datetime.now()
        
        # Save to disk
        self._save_session_file(session)

    def _save_session_file(self, session: SessionData):
        """Write session data to JSON file."""
        file_path = self._get_file_path(session.notebook_id, session.cell_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(session.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save session to {file_path}: {e}")

    def get_history(self, notebook_id: str, cell_id: str) -> List[BaseMessage]:
        """Get LangChain-formatted history for context injection."""
        session = self.load_session(notebook_id, cell_id)
        return session.to_langchain_history()

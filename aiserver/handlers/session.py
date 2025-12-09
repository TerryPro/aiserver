import json
import logging
from jupyter_server.base.handlers import APIHandler
from tornado import web
from ..core.session import SessionManager

logger = logging.getLogger(__name__)

class SessionHistoryHandler(APIHandler):
    """
    Handler for retrieving and managing AI session history for specific notebook cells.
    """
    
    def initialize(self):
        try:
            self.ai_session_manager = SessionManager()
        except Exception as e:
            logger.error(f"Failed to initialize Session Manager: {e}")

    @web.authenticated
    def get(self):
        """
        Get session history for a specific cell.
        
        Query Parameters:
        - notebook_id: The ID/Path of the notebook
        - cell_id: The ID of the cell
        
        Returns:
        - JSON object containing session data and interactions
        """
        notebook_id = self.get_argument("notebook_id", None)
        cell_id = self.get_argument("cell_id", None)
        
        if not notebook_id or not cell_id:
            self.set_status(400)
            self.finish(json.dumps({"error": "Missing notebook_id or cell_id parameter"}))
            return

        try:
            # Load session data
            session_data = self.ai_session_manager.load_session(notebook_id, cell_id)
            
            # Convert to dict using Pydantic's model_dump (v2) or dict (v1)
            # Based on core/session.py usage of model_dump_json, we assume v2 or compatible
            response_data = json.loads(session_data.model_dump_json())
            
            self.finish(json.dumps(response_data))
            
        except Exception as e:
            logger.error(f"Error retrieving session history: {e}")
            self.set_status(500)
            self.finish(json.dumps({"error": str(e)}))

    @web.authenticated
    def delete(self):
        """
        Clear/Reset session history for a specific cell.
        """
        notebook_id = self.get_argument("notebook_id", None)
        cell_id = self.get_argument("cell_id", None)

        if not notebook_id or not cell_id:
            self.set_status(400)
            self.finish(json.dumps({"error": "Missing notebook_id or cell_id parameter"}))
            return
            
        # Currently, SessionManager doesn't have a delete method, but we can implement a logical reset
        # by saving an empty session or deleting the file.
        # For now, let's defer this implementation or just return success if we rely on frontend clearing.
        # But for "New Chat", backend state *should* be cleared.
        
        # TODO: Implement delete/reset in SessionManager
        # For now, we'll just log it.
        logger.warning(f"Delete session requested for {notebook_id}/{cell_id} - Not fully implemented yet")
        self.finish(json.dumps({"status": "reset", "message": "Session reset (backend implementation pending)"}))

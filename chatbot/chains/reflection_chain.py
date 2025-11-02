from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from chatbot.prompts.templates import REFLECTION_PROMPT
from chatbot.memory.conversation_memory import MindEaseMemory
from utils.config import Config


class ReflectionChain:
    """
    Generates thoughtful check-ins and reflections based on conversation history.
    Helps users gain self-awareness and process their emotions.
    """

    def __init__(self, memory: MindEaseMemory = None):
        # Initialize Cohere model
        self.llm = ChatCohere(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.COHERE_MODEL,
            temperature=0.8,  # Slightly higher for creative reflections
            max_tokens=200
        )

        # Use shared memory (conversation + emotions)
        self.memory = memory or MindEaseMemory()

        # Create prompt (convert REFLECTION_PROMPT to a LangChain prompt if needed)
        # If REFLECTION_PROMPT is already a ChatPromptTemplate, no change needed
        if isinstance(REFLECTION_PROMPT, str):
            self.prompt = ChatPromptTemplate.from_template(REFLECTION_PROMPT)
        else:
            self.prompt = REFLECTION_PROMPT

        # Build chain using RunnableSequence
        # (Prompt + LLM) → simple and clean pipeline
        self.chain = self.prompt | self.llm

    def generate_reflection(self) -> str:
        """Generate a reflective check-in based on conversation history"""
        try:
            chat_history = self.memory.get_chat_history()

            # Only generate reflections if there's meaningful history
            if len(chat_history) < 4:
                return None

            # Run prompt + LLM pipeline
            reflection = self.chain.invoke({"chat_history": chat_history})
            # New LLM output is usually in .content, not dict['output']
            if hasattr(reflection, "content"):
                return reflection.content.strip()
            elif isinstance(reflection, dict):
                return (reflection.get("text") or "").strip()
            else:
                return str(reflection).strip()

        except Exception as e:
            print(f"Error generating reflection: {e}")
            return None

    def should_trigger_reflection(self, message_count: int) -> bool:
        """
        Determine if it's a good time for a reflection.
        Triggers every 5-7 messages to avoid being intrusive.
        """
        return message_count > 0 and message_count % 6 == 0

    def generate_session_summary(self) -> str:
        """Generate a summary reflection at the end of a conversation"""
        try:
            chat_history = self.memory.get_chat_history()

            if len(chat_history) < 2:
                return "Thanks for sharing with me today. Take care of yourself! 🌱"

            emotional_state = self.memory.get_emotional_summary()

            summary_prompt = f"""Based on our conversation, provide a brief, warm closing reflection.

Recent emotional tone: {emotional_state}
Conversation length: {len(chat_history)} exchanges

Generate a 2-3 sentence supportive closing that:
- Acknowledges their sharing
- Notes any positive shifts or insights
- Offers gentle encouragement

Keep it natural and heartfelt."""

            response = self.llm.invoke(summary_prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, dict):
                return (response.get("text") or "").strip()
            else:
                return str(response).strip()

        except Exception as e:
            print(f"Error generating session summary: {e}")
            return "Thank you for opening up today. Remember, you're doing the best you can. 🌱"

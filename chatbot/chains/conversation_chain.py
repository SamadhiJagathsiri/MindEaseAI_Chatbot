from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from chatbot.prompts.templates import CONVERSATION_PROMPT
from chatbot.memory.conversation_memory import MindEaseMemory
from utils.config import Config


class ConversationChain:
    """Handles the main chat logic between user and MindEase AI."""

    def __init__(self, memory: MindEaseMemory = None):
        # Initialize Cohere LLM
        self.llm = ChatCohere(
            cohere_api_key=Config.COHERE_API_KEY,
            model=Config.COHERE_MODEL,
            temperature=0.7,
            max_tokens=300
        )

        # Initialize memory
        self.memory = memory or MindEaseMemory()

        # Prepare prompt
        if isinstance(CONVERSATION_PROMPT, str):
            self.prompt = ChatPromptTemplate.from_template(CONVERSATION_PROMPT)
        else:
            self.prompt = CONVERSATION_PROMPT

        # Modern pipeline: prompt → llm
        self.chain = self.prompt | self.llm

    def generate_response(self, user_input: str) -> str:
        """Generate a response for the user’s input with memory context."""
        try:
            chat_history = self.memory.get_chat_history()

            inputs = {
                "input": user_input,
                "chat_history": chat_history
            }

            response = self.chain.invoke(inputs)

            # Handle different return types
            if hasattr(response, "content"):
                output_text = response.content.strip()
            elif isinstance(response, dict):
                output_text = (response.get("text") or "").strip()
            else:
                output_text = str(response).strip()

            self.memory.add_interaction(user_input, output_text)

            return output_text

        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble processing that right now. Could you try rephrasing?"

    def clear_memory(self):
        """Clear all chat history and emotional context."""
        self.memory.clear()

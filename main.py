from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType

from agno.embedder.fastembed import FastEmbedEmbedder
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder

from agno.playground import Playground
from agno.storage.sqlite import SqliteStorage

from dotenv import load_dotenv
import os


load_dotenv()


# Storage
agent_storage: str = "tmp/agents.db"


# Database===========
vector_db = PgVector(
  table_name="music_books", 
  db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
  embedder=FastEmbedEmbedder(),
  search_type=SearchType.hybrid,
)


# Knowledge=======================
knowledge_base = PDFKnowledgeBase(
    path="data",
    vector_db=vector_db,
)
knowledge_base.load()


# Agent=======
agent = Agent(
    name="RAG agent",
    model=OpenRouter(id="deepseek/deepseek-chat-v3.1:free", api_key=os.getenv("TOKEN")),
    knowledge=knowledge_base,
    storage=SqliteStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True
)

# Playground===============================
playground_app = Playground(agents=[agent])
app = playground_app.get_app()

if __name__ == "__main__":
    playground_app.serve("main:app", reload=True)

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.vectordb.pgvector import PgVector, SearchType

from agno.embedder.fastembed import FastEmbedEmbedder
from agno.embedder.sentence_transformer import SentenceTransformerEmbedder

from dotenv import load_dotenv
import os


load_dotenv()

# Database
vector_db = PgVector(
  table_name="music_books", 
  db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
  embedder=FastEmbedEmbedder(),
  search_type=SearchType.hybrid,
)


# Knowledge============================
knowledge_base = PDFKnowledgeBase(
    path="data",
    vector_db=vector_db,
)
knowledge_base.load()


# Agent=======
agent = Agent(
    model=OpenRouter(id="deepseek/deepseek-chat-v3.1:free", api_key=os.getenv("TOKEN")),
    knowledge=knowledge_base,
    markdown=True,
)

agent.print_response("Monte um pequeno texto de blog sobre como foi a construção do circuito LiFi", markdown=True)

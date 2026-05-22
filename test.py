from src.rag_system import RAGSystem

rag = RAGSystem()

question = "Who is the temporary speaker?"
answer = rag.answer_questions(question)
print(answer)
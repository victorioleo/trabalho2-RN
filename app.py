from core.main import generate_answer, init

if __name__ == "__main__":
    init(path=r"C:\Users\leonardo.victorio\Documents\trabalho2-RN\regulamento.pdf")
    question = "Qual é o prazo máximo para a defesa da dissertação?"
    answer = generate_answer(question=question, local=True)
    print(answer)
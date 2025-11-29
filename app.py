import json
import os
from core.main import main

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

def run_single_example(local: bool = False):
    question = "Qual é o prazo máximo para a defesa da dissertação?"
    print(f"Pergunta: {question}")
    details = main(question=question, local=local)
    print(f"Resposta: {details['answer']}")
    print(f"Total de fatos: {details['total_facts']}")
    print(f"Fatos suportados: {details['supported_facts']}")
    print(f"Fatos não suportados: {details['unsupported_facts']}")
    print(f"Precisão factual: {details['factual_accuracy_score']:.2%}")

def run_batch(questions_path: str = "questions.json", local: bool = False):
    if not os.path.exists(questions_path):
        print("File questions.json not found. Running single example.")
        run_single_example()
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    results = []
    
    for item in questions:
        qid = item.get("id")
        question = item.get("question")
        print(f"[Q{qid}] - Pergunta: {question}")
        details = main(question=question, local=local)
        results.append({
            "id": qid,
            "question": question,
            "answer": details['answer'],
            "total_facts": details['total_facts'],
            "supported_facts": details['supported_facts'],
            "unsupported_facts": details['unsupported_facts'],
            "factual_accuracy_score": details['factual_accuracy_score']
        })
        print(f"Resposta: {details['answer']} --> Precisão factual: {details['factual_accuracy_score']:.2%}\n")

    if results:
        media = sum(r["factual_accuracy_score"] for r in results) / len(results)
    else:
        media = 0.0

    print("Resumo Final:")
    print(f"Média de precisão factual nas {len(results)} perguntas: {media:.2%}")

    with open("questions_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Arquivo 'questions_results.json' salvo.")

    if _HAS_MPL and results:
        xs = [r["id"] for r in results]
        ys = [r["factual_accuracy_score"] * 100 for r in results]
        plt.figure(figsize=(8,4))
        plt.plot(xs, ys, marker='o', linestyle='-', color='#1f77b4')
        plt.title('Evolução da Precisão Factual (%)')
        plt.xlabel('ID da Pergunta')
        plt.ylabel('Precisão (%)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('factual_accuracy_progress.png')
        print("Gráfico salvo em 'factual_accuracy_progress.png'.")
    elif not _HAS_MPL:
        print("matplotlib não instalado. Instale para gerar gráfico: pip install matplotlib")

if __name__ == "__main__":
    run_batch(local=False)
    #run_single_example(local=False)

    
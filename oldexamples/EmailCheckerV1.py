import pandas as pd
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock

async def main():
    graph = WorkflowGraph()
    llm_model = "ollama/mistral-nemo:latest"
    email_checker_agent = LLMAgentBlock(
            name="EmailCheckerAgent",
            description="Um agente que verifica se um email é apenas informativo ou se exige uma resposta",
            system_prompt="""Você analisa o email de um cliente e o classifica em 
                            'informativo' ou  'de pedidos'. O email informativo 
                            não exige nenhuma ação do usuário. Para este email você deve
                            detectar a informação principal e resumí-la para o usuário.
                            O email de pedidos exige algumas ações do usuário, como responder ao email,
                            seguir alguma instrução, trocar senhas por motivo de segurança, analisar atividades
                            suspeitas, e confirmações em geral. Para este email, você deve sugerir ações claras e concisas a serem tomadas
                            pelo usuário.Seja suscinto nas respostas . Não proponhas várias respostas possíveis.
                            As ações devem vir em uma lista de strings da forma [acao1, acao2, ..., acaoN], tal que
                            acaok é uma ação clara e concisa a ser tomada pelo usuário. Vou dar exemplos de
                            entradas e saídas esperadas:
                        Exemplo 1:            
                            Entrada: "A reunião de planejamento trimestral ocorrerá na próxima segunda-feira às 14h, na sala de conferências B. Por favor, revise o material em anexo antes do encontro."
                            Saída: "Resultado: Classificação: de pedidos. Ações: ['rever o material em anexo', 'participar da reunião']"
                        Exemplo 2:"O sistema de e-mails estará em manutenção no dia 10/05, das 22h às 6h do dia seguinte. O serviço ficará indisponível nesse período."
                            Resultado: Classificação: informativo
                            Resumo: Manutenção do sistema em 10/05, das 22h às 6h.""",
            model=llm_model,
            litellm_kwargs={"temperature": 0.1, "num_ctx": 8192},
        )
    
    graph.add_block(email_checker_agent)
    
    executor = WorkflowExecutor(graph)

    df = pd.read_csv("inputs.csv") #with head tipo,entrada,resumo,acoes
    for index, row in df.iterrows():
        email = row['entrada']
        print("-----------------------------------------------------------------------------------------------------------------")
        print(f"Analisando email: {email}")
        print("-----------------------------------------------------------------------------------------------------------------")
        ctx = await executor.run(initial_input={"prompt": email})
        result = ctx.get_output("EmailCheckerAgent")
        print(f"{result.response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
import pandas as pd
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock

async def main():

    graph = WorkflowGraph()
    llm_model = "ollama/mistral-nemo:latest" #"gemini/gemini-3.1-flash-lite-preview"

    email_preparer = LLMAgentBlock(
            name="EmailPreparer",
            description="Este agente resume o email e identifique as ações de modo a deixá-las detacadas para o agente revisor",
            system_prompt="""Reformate o email de um cliente de modo a destacar 
            a informação principal e as ações  a serem tomadas pelo usuário (se alguma). 
            O email deve ser resumido e as ações devem ser listadas no final do email 
            reformulado, em uma seção chamada 'Ações'. A seção de ações deve conter uma 
            lista de strings da forma  [acao1, acao2, ..., acaoN], tal que acaok é uma 
            ação clara e concisa  a ser tomada pelo usuário. Seja sucinto nas reformulações 
            e nas ações.  Não proponhas várias respostas possíveis. Se não houver ações a serem 
            tomadas, não apresente qualquer lista de ações.""",
            model=llm_model,  
    )

    email_checker_agent = LLMAgentBlock(
            name="EmailCheckerAgent",
            description="Um agente que verifica se um email é apenas informativo ou se exige uma resposta",
            system_prompt="""Você analisa o email de um cliente e o classifica em 
                            'informativo' ou  'pedidos'. O email informativo 
                            não exige nenhuma ação do usuário. Para este email você deve
                            detectar a informação principal e resumí-la para o usuário.
                            O email de pedidos exige algumas ações do usuário, como responder ao email,
                            seguir alguma instrução, trocar senhas por motivo de segurança, analisar atividades
                            suspeitas, e confirmações em geral. Para este email, você deve sugerir ações claras e concisas a serem tomadas
                            pelo usuário.Seja suscinto nas respostas . Não proponhas várias respostas possíveis.
                            A informação deve ser apresentada de forma estruturada, seguindo o formato: 
                            Tipo = [informativo/pedido], Resumo = [resumo], Ações = [ação1, ação2, ...]. 
                            As ações devem vir em uma lista de strings da forma [acao1, acao2, ..., acaoN], tal que
                            acaok é uma ação clara e concisa a ser tomada pelo usuário.""",
            model=llm_model
    )

    graph.add_block(email_preparer)
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
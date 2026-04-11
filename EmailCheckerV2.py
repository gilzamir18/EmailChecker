import pandas as pd
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import SharedLLMAgentBlock

async def main():

    graph = WorkflowGraph()
    llm_model = "ollama/mistral-nemo:latest" #"gemini/gemini-3.1-flash-lite-preview"
    
    email_checker_agent = SharedLLMAgentBlock(
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
                            acaok é uma ação clara e concisa a ser tomada pelo usuário.""",
            model=llm_model,
            litellm_kwargs={"temperature": 0.1, "num_ctx": 16384},
    )
    
    email_checker_reviewer_agent = SharedLLMAgentBlock(
            name="EmailCheckerReviewerAgent",
            description="Um agente que verifica se um email é apenas informativo ou se exige uma resposta",
            system_prompt=f"""Use a ferramenta EmailCheckerAgent e verifique se a classificação e as ações sugeridas pelo EmailCheckerAgent estão corretas.
            Se estiverem corretas, apenas reproduza a resposta. Se estiverem incorretas, corrija a classificação e as ações sugeridas.
            Envie o email original para a ferramenta EmailCheckerAgent. E verifica se a resposta do EmailCheckerAgent está correta. Se estiver correta, apenas reproduza a resposta. Se estiver incorreta, corrija a classificação e as ações sugeridas.
            Seja sucinto nas respostas. Não proponhas várias respostas possíveis. As ações devem vir em uma lista de strings da forma [acao1, acao2, ..., acaoN], tal que acaok é uma ação clara e concisa a ser tomada pelo usuário.  """,
            model=llm_model,
            tools=[email_checker_agent],  
            litellm_kwargs={"tool_choice": "required", "temperature": 0.1, "num_ctx": 16384},
    )

    graph.add_block(email_checker_reviewer_agent)
    
    executor = WorkflowExecutor(graph)

    df = pd.read_csv("inputs.csv") #with head tipo,entrada,resumo,acoes
    for index, row in df.iterrows():
        email = row['entrada']
        print("-----------------------------------------------------------------------------------------------------------------")
        print(f"Analisando email: {email}")
        print("-----------------------------------------------------------------------------------------------------------------")
        ctx = await executor.run(initial_input={"prompt": email})
        result = ctx.get_output("EmailCheckerReviewerAgent")
        print(f"{result.response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
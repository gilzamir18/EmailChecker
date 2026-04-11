import pandas as pd
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from agenticblocks import as_tool
import re

@as_tool
def check_content(content: str) -> dict:
        """Verifica se o conteúdo do email segue o formato esperado e
          se as informações são coerentes."""
        pattern = r"Tipo\s*=\s*(\w+)\s*,\s*Resumo\s*=\s*(.+?)\s*,\s*Ações\s*=\s*\[(.+?)\]"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
                return {"is_valid": False, "feedback": "Formato de saída inválido. Certifique-se de seguir o formato: Tipo = [informativo/pedido], Resumo = [resumo], Ações = [ação1, ação2, ...]"}

        tipo = match.group(1)
        resumo = match.group(2).strip()
        acoes_str = match.group(3)

        if tipo.strip() not in ["informativo", "pedido"]:
                return {"is_valid": False, "feedback": "Tipo inválido. O tipo deve ser 'informativo' ou 'pedido'."}

        if len(resumo.split()) > 100:
                return {"is_valid": False, "feedback": "Resumo muito longo. O resumo deve conter no máximo 100 palavras."}

        try:
                acoes = [acao.strip().strip("'\"") for acao in acoes_str.split(",")]
                return {"is_valid": True, "feedback": "Formato de saída válido."}
        except:
                return {"is_valid": False, "feedback": "Formato de ações inválido. As ações devem ser listadas entre colchetes e separadas por vírgulas."}


async def main():

    graph = WorkflowGraph()
    llm_model = "ollama/mistral-nemo:latest" #"gemini/gemini-3.1-flash-lite-preview"

    email_preparer = LLMAgentBlock(
            name="EmailPreparer",
            description="Este agente resume o email e identifique as ações de modo a deixá-las detacadas para o agente revisor",
            system_prompt=f"""Reformate o email de um cliente de modo a destacar as ações 
            a serem tomadas pelo usuário. O email deve ser resumido e as ações devem ser 
            listadas no final do email reformulado, em uma seção chamada 'Ações'. 
            A seção de ações deve conter uma lista de strings da forma 
            [acao1, acao2, ..., acaoN], tal que acaok é uma ação clara e concisa 
            a ser tomada pelo usuário. Seja sucinto nas reformulações e nas ações. 
            Não proponhas várias respostas possíveis. Se não houver ações a serem 
            tomadas, não apresente qualquer lista de ações.""",
            model=llm_model,  
    )

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
                            acaok é uma ação clara e concisa a ser tomada pelo usuário.""",
            model=llm_model
    )

    graph.add_block(email_preparer)
    graph.add_block(email_checker_agent)
    graph.add_block(check_content)
    graph.add_cycle(
           name="reflection",
           sequence=['EmailPreparer', 'EmailCheckerAgent', 'check_content'],
           condition_block="check_content",
           max_iterations=10
    )

    executor = WorkflowExecutor(graph)

    df = pd.read_csv("inputs.csv") #with head tipo,entrada,resumo,acoes
    for index, row in df.iterrows():
        email = row['entrada']
        print("-----------------------------------------------------------------------------------------------------------------")
        print(f"Analisando email: {email}")
        print("-----------------------------------------------------------------------------------------------------------------")
        ctx = await executor.run(initial_input={"prompt": email})
        result = ctx.cycle_results.get("reflection")
        print(f"{result.output.response}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
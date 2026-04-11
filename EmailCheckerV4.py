import pandas as pd
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import SharedLLMAgentBlock, LLMAgentBlock
from agenticblocks import as_tool
import re
from agenticblocks.tools.a2a_bridge import block_to_tool_schema
from agenticblocks.blocks.flow.validator_loop import ValidatorLoopBlock, ValidatorLoopInput


llm_model = "ollama/mistral-nemo:latest" #"gemini/gemini-3.1-flash-lite-preview"


@as_tool
def check_content(content: str) -> dict:
        """Verifica se o conteúdo do email segue o formato esperado e
          se as informações são coerentes."""
        #print("-------------------------->>>> ", content)
        pattern = r"Type\s*=\s*(\w+)\s*,\s*Summary\s*=\s*(.+?)\s*,\s*Actions\s*=\s*\[(.+?)\]"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
                return {"is_valid": False, "feedback": "Formato de saída inválido. Certifique-se de seguir o formato: Type = [informativo/pedido], Summary = [resumo], Actions = [ação1, ação2, ...]"}

        tipo = match.group(1)
        resumo = match.group(2).strip()
        acoes_str = match.group(3)

        if tipo not in ["informativo", "pedido"]:
                return {"is_valid": False, "feedback": "Tipo inválido. O tipo deve ser 'informativo' ou 'pedido'."}

        if len(resumo.split()) > 200:
                return {"is_valid": False, "feedback": "Resumo muito longo. O resumo deve conter no máximo 200 palavras."}

        try:
                acoes = [acao.strip().strip("'\"") for acao in acoes_str.split(",")]
                return {"is_valid": True, "feedback": "Formato de saída válido."}
        except:
                return {"is_valid": False, "feedback": "Formato de ações inválido. As ações devem ser listadas entre colchetes e separadas por vírgulas."}


async def main():

    email_checker_agent = LLMAgentBlock(
            name="email_checker",
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
            litellm_kwargs={"temperature": 1.0, "num_ctx": 18192},
        )

    graph = WorkflowGraph()

    graph.add_block(email_checker_agent)
    graph.add_block(check_content)

    graph.add_cycle(
           name="reflection",
           edges=[('email_checker', 'check_content')],
           condition_block='check_content',
           max_iterations=5,
    )

    executor = WorkflowExecutor(graph)

    df = pd.read_csv("inputs.csv") #with head tipo,entrada,resumo,acoes
    for index, row in df.iterrows():
            email = row['entrada']
            print("-----------------------------------------------------------------------------------------------------------------")
            print(f"Analisando email: {email}")
            print("-----------------------------------------------------------------------------------------------------------------")
           
            
            ctx = await executor.run(initial_input={'prompt': (email)})
            cr = ctx.cycle_results.get('reflection')
            if cr:
                print(f"\n[Done] Iterations: {cr.iterations} | Validated: {cr.validated}")
                print("\n--- Final email ---\n")
                print(cr.output.response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
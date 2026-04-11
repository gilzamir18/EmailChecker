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
            model=llm_model,
            litellm_kwargs={"temperature": 0.1, "num_ctx": 8192},
        )

# ── Loop produtor → validador ─────────────────────────────────────────────────
loop = ValidatorLoopBlock(
    name="email_loop",
    producer=email_checker_agent,
    validator=check_content,   # @as_tool — sem classe extra
    max_iterations=3,
)

async def main():

        df = pd.read_csv("inputs.csv") #with head tipo,entrada,resumo,acoes
        for index, row in df.iterrows():
                email = row['entrada']
                print("-----------------------------------------------------------------------------------------------------------------")
                print(f"Analisando email: {email}")
                print("-----------------------------------------------------------------------------------------------------------------")
                input = ValidatorLoopInput(prompt=(email))
                
                result = await loop.run(input=input)
                print("\n" + "=" * 60)
                print(f"Iterações: {result.iterations} | Validado: {result.validated}")
                print("=" * 60)
                print("\n Final Report:\n")
                print(result.result)
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
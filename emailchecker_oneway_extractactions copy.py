import asyncio
from agenticblocks.core.graph import WorkflowGraph
from agenticblocks.runtime.executor import WorkflowExecutor
from agenticblocks.blocks.llm.agent import AgentOutput
from agenticblocks.blocks.llm.agent import LLMAgentBlock
from pydantic import BaseModel

def extract_actions(output: str):
    if "@actions:" in output:
        actions_part = output.split("@actions:")[1].strip()
        actions = [action.strip() for action in actions_part.strip("[]").split(",")]
        return actions
    return []

async def main():
    graph = WorkflowGraph()
    llm_model = "ollama/granite4"
    email_checker_agent = LLMAgentBlock(
        name="EmailCheckerAgent",
        description="Um agente que verifica se um email é apenas informativo ou se exige uma resposta",
        system_prompt="""Você é um agente que verifica se um email é apenas informativo
                         ou se exige uma resposta. Se o email exigir uma resposta, sugira 
                         ações a serem tomadas.
                         Não instingue o usuário a continuar interagindo com o agente.
                         Se o email for apenas informativo, indique que nenhuma ação é necessária.
                         Coloque a lista de ações no formato de uma lista numerada usando
                         a forma @actions: [ação1, ação2, ...]""",
        model=llm_model)
    graph.add_block(email_checker_agent)
    executor = WorkflowExecutor(graph)
    ctx = await executor.run(initial_input={"prompt": 
                                            """O email é: 'Assunto: Urgente! Atualize sua senha. 
                                            De: segurança@email.com
                                            Para: usuario@email.com
                                            Corpo: Prezado usuário, detectamos uma tentativa 
                                            de acesso suspeita em sua conta."""})
    output = ctx.get_output("EmailCheckerAgent")
    actions = extract_actions(output.response)
    print("Output: ", output.response)
    print("Actions: ")
    for action in actions:
        print(f"- {action}")

if __name__ == "__main__":
    asyncio.run(main())